# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Unified Account System for Kiro Gateway.

Manages multiple Kiro accounts with intelligent failover, sticky behavior,
and circuit breaker pattern for reliability.

Key features:
- Lazy initialization (only first working account at startup)
- Sticky behavior (prefer successful account)
- Circuit breaker with exponential backoff
- Probabilistic retry for "dead" accounts
- TTL-based model cache refresh (only when using account)
- Atomic state persistence
"""

import asyncio
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from loguru import logger

from kiro.auth import KiroAuthManager, AuthType
from kiro.cache import ModelInfoCache
from kiro.model_resolver import ModelResolver, normalize_model_name
from kiro.config import (
    HIDDEN_MODELS,
    MODEL_ALIASES,
    HIDDEN_FROM_LIST,
    ACCOUNT_RECOVERY_TIMEOUT,
    ACCOUNT_MAX_BACKOFF_MULTIPLIER,
    ACCOUNT_PROBABILISTIC_RETRY_CHANCE,
    ACCOUNT_CACHE_TTL,
    STATE_SAVE_INTERVAL_SECONDS,
    FALLBACK_MODELS,
    QUOTA_POLL_INTERVAL,
    QUOTA_POLL_INITIAL_DELAY,
    ACCOUNT_PRIORITY,
)
from kiro.quota import QUOTA_UNKNOWN_SENTINEL, QuotaInfo, query_quota
from kiro.utils import get_kiro_headers
from kiro.account_errors import ErrorType
from kiro.http_client import KiroHttpClient


def _is_runtime_endpoint(auth_manager: KiroAuthManager) -> bool:
    """
    Check if auth manager uses runtime endpoint that doesn't provide /ListAvailableModels.
    
    Runtime endpoint pattern: https://runtime.{region}.kiro.dev
    Old endpoint pattern: https://q.{region}.amazonaws.com
    
    Runtime endpoint does not provide /ListAvailableModels API (AWS limitation).
    
    Args:
        auth_manager: KiroAuthManager instance
    
    Returns:
        True if using runtime endpoint, False otherwise
    
    Examples:
        >>> auth_manager.api_host = "https://runtime.us-east-1.kiro.dev"
        >>> _is_runtime_endpoint(auth_manager)
        True
        >>> auth_manager.api_host = "https://runtime.eu-central-1.kiro.dev"
        >>> _is_runtime_endpoint(auth_manager)
        True
        >>> auth_manager.api_host = "https://q.us-east-1.amazonaws.com"
        >>> _is_runtime_endpoint(auth_manager)
        False
    """
    return "://runtime." in auth_manager.api_host


def _format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string (e.g., "30s", "5m", "2h", "1d")
    
    Examples:
        >>> _format_duration(30)
        '30s'
        >>> _format_duration(300)
        '5m'
        >>> _format_duration(7200)
        '2h'
        >>> _format_duration(86400)
        '1d'
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds / 60)}m"
    elif seconds < 86400:
        return f"{int(seconds / 3600)}h"
    else:
        return f"{int(seconds / 86400)}d"


@dataclass
class AccountStats:
    """
    Statistics for account usage.
    
    Tracks request counts for monitoring and future web UI.
    """
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0


@dataclass
class Account:
    """
    Complete account entity with all dependencies.
    
    Represents a single Kiro account with its authentication,
    model cache, resolver, and runtime state.
    
    Attributes:
        id: Unique identifier (path to credentials file)
        owner: Human-readable owner label (parent directory name when
            loaded from a recursive folder scan, or a hash for refresh tokens)
        auth_manager: Authentication manager (lazy initialized)
        model_cache: Model metadata cache (lazy initialized)
        model_resolver: Model resolver (lazy initialized)
        failures: Consecutive failure count (for Circuit Breaker)
        last_failure_time: Timestamp of last failure
        models_cached_at: Timestamp of last model cache update
        stats: Usage statistics
        quota_info: Latest quota information (lazy initialized by poller)
    """
    id: str
    owner: str = ""
    auth_manager: Optional[KiroAuthManager] = None
    model_cache: Optional[ModelInfoCache] = None
    model_resolver: Optional[ModelResolver] = None
    failures: int = 0
    last_failure_time: float = 0.0
    models_cached_at: float = 0.0
    stats: AccountStats = field(default_factory=AccountStats)
    quota_info: Optional[QuotaInfo] = None


@dataclass
class ModelAccountList:
    """
    List of accounts for a specific model.
    
    Attributes:
        accounts: List of account IDs that have this model
    
    Note: next_index removed - now using global _current_account_index
    """
    accounts: List[str] = field(default_factory=list)


class AccountManager:
    """
    Manages multiple Kiro accounts with intelligent failover.
    
    Responsibilities:
    - Load credentials from credentials.json
    - Lazy initialization of accounts
    - Select next available account (Circuit Breaker + Sticky)
    - Track statistics and failures
    - Persist state to state.json
    
    Example:
        >>> manager = AccountManager("credentials.json", "state.json")
        >>> await manager.load_credentials()
        >>> await manager.load_state()
        >>> account = await manager.get_next_account("claude-opus-4.5")
        >>> await manager.report_success(account.id, "claude-opus-4.5")
    """
    
    def __init__(self, credentials_file: str, state_file: str):
        """
        Initialize AccountManager.
        
        Args:
            credentials_file: Path to credentials.json
            state_file: Path to state.json
        """
        self._credentials_file = credentials_file
        self._state_file = state_file
        self._accounts: Dict[str, Account] = {}
        self._model_to_accounts: Dict[str, ModelAccountList] = {}
        self._lock = asyncio.Lock()
        self._dirty = False
        self._credentials_config: List[Dict] = []
        self._current_account_index: int = 0  # GLOBAL sticky index for all models
        # Per-account initialization locks prevent concurrent initialization
        # of the same account (e.g., get_next_account and poll_quota_once
        # both trying to init the same account simultaneously).
        self._init_locks: Dict[str, asyncio.Lock] = {}
    
    def _register_account(self, account_id: str, owner: str, source: str) -> None:
        """
        Register an account, warning if it silently replaces an existing one.

        Account identity is the resolved credential path (or refresh-token hash),
        so two credentials.json entries pointing at the same file collapse into a
        single account. Without a warning the user's second owner label would be
        applied silently and one entry would appear to vanish.

        Args:
            account_id: Unique account identifier (resolved path or token hash).
            owner: Owner label to assign (may be empty).
            source: Human-readable origin, used only for log messages.
        """
        existing = self._accounts.get(account_id)
        if existing is not None:
            logger.warning(
                f"Duplicate credential detected: {account_id} is already registered "
                f"(owner={existing.owner or '<none>'}). Overriding with owner="
                f"{owner or '<none>'}. Remove the duplicate entry from your "
                f"credentials.json or KIRO_CLI_DB to silence this warning."
            )

        self._accounts[account_id] = Account(id=account_id, owner=owner)
        logger.debug(f"Added account {source}: {account_id} (owner={owner or '<derived>'})")

    def _get_init_lock(self, account_id: str) -> asyncio.Lock:
        """
        Get or create a per-account initialization lock.

        This lock prevents concurrent initialization or model refresh of the
        same account by multiple callers (e.g., get_next_account and
        poll_quota_once both attempting to initialize the same account).

        Args:
            account_id: Account ID to get the lock for.

        Returns:
            An asyncio.Lock unique to this account.
        """
        if account_id not in self._init_locks:
            self._init_locks[account_id] = asyncio.Lock()
        return self._init_locks[account_id]
    
    async def load_credentials(self) -> None:
        """
        Load credentials from credentials.json.
        
        Validates each entry and creates Account objects.
        Invalid entries are skipped with warnings.
        Folders are scanned for credential files.
        """
        creds_path = Path(self._credentials_file).expanduser()
        
        if not creds_path.exists():
            logger.warning(f"Credentials file not found: {self._credentials_file}")
            return
        
        try:
            with open(creds_path, 'r', encoding='utf-8') as f:
                self._credentials_config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            return
        
        # Process each credential entry
        for entry in self._credentials_config:
            cred_type = entry.get("type")
            path = entry.get("path")
            enabled = entry.get("enabled", True)
            
            if not enabled:
                continue
            
            # Validate required fields based on type
            if not cred_type:
                logger.warning(f"Invalid credential entry (missing type): {entry}")
                continue
            
            # For json/sqlite types, path is required
            if cred_type in ("json", "sqlite") and not path:
                logger.warning(f"Invalid credential entry (type={cred_type} requires path): {entry}")
                continue
            
            # For refresh_token type, refresh_token field is required
            if cred_type == "refresh_token" and not entry.get("refresh_token"):
                logger.warning(f"Invalid credential entry (type=refresh_token requires refresh_token field): {entry}")
                continue
            
            # Handle refresh_token type (no path processing needed)
            if cred_type == "refresh_token":
                # Use deterministic hash for refresh_token (hash() is not deterministic between process restarts)
                token = entry.get('refresh_token', '')
                token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
                account_id = f"refresh_token_{token_hash}"
                self._register_account(account_id, entry.get("owner", ""), "from refresh_token")
                continue  # Skip path processing for refresh_token
            
            # Handle folder scanning for json/sqlite types
            expanded_path = Path(path).expanduser()
            if expanded_path.is_dir():
                recursive = entry.get("recursive", False)
                logger.info(f"Scanning folder for credentials: {path} (recursive={recursive})")
                for file_path in expanded_path.iterdir():
                    # Non-recursive: only top-level files
                    if not recursive and not file_path.is_file():
                        continue
                    
                    # Recursive: descend into subdirectories looking for credential files
                    if recursive and file_path.is_dir():
                        for sub_file in file_path.iterdir():
                            if not sub_file.is_file():
                                continue
                            self._try_add_credential_file(sub_file, cred_type, entry)
                        continue
                    
                    if recursive and not file_path.is_file():
                        continue
                    
                    self._try_add_credential_file(file_path, cred_type, entry)
            elif expanded_path.is_file():
                # Single credential file (refresh_token was already handled above)
                resolved_path = expanded_path.resolve()
                account_id = str(resolved_path)
                # Owner label: prefer the explicit "owner" field; otherwise derive it
                # from the parent directory name. This matches both the folder-scan
                # behavior in _try_add_credential_file and the documented KIRO_CLI_DB
                # contract (e.g. kiro-cli-db-file/wangmingrong1/data.sqlite3 ->
                # "wangmingrong1"). Without derivation the owner would fall back to the
                # full credential path in quota polling and the status dashboard.
                owner = entry.get("owner") or resolved_path.parent.name
                self._register_account(account_id, owner, "from file")
            else:
                logger.warning(f"Credential path not found: {path}")
        
        logger.info(f"Loaded {len(self._accounts)} account(s) from credentials")
    
    def _try_add_credential_file(self, file_path: Path, cred_type: str, entry: Dict) -> None:
        """
        Validate and register a single credential file as an account.
        
        Determines the owner label from the parent directory name when the
        file is nested inside a scanned folder (e.g.
        kiro-cli-db-file/wangmingrong1/data.sqlite3 → owner "wangmingrong1").
        
        Args:
            file_path: Path to the credential file.
            cred_type: Credential type ("json" or "sqlite").
            entry: The credentials.json entry (for potential future use).
        """
        account_id = str(file_path.resolve())
        is_valid = False
        
        # Try JSON validation
        if cred_type == "json":
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'refreshToken' in data or 'clientId' in data:
                        is_valid = True
            except Exception as e:
                logger.warning(f"Invalid JSON credentials file {file_path.name}: {e}")
        
        # Try SQLite validation
        elif cred_type == "sqlite":
            try:
                import sqlite3
                conn = sqlite3.connect(str(file_path))
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='auth_kv'")
                if cursor.fetchone():
                    is_valid = True
                conn.close()
            except Exception as e:
                logger.warning(f"Invalid SQLite database file {file_path.name}: {e}")
        
        if is_valid:
            # Derive owner label: prefer explicit "owner" field from credentials.json
            # entry; otherwise fall back to the parent directory name (for recursive
            # scans, e.g. kiro-cli-db-file/wangmingrong1/data.sqlite3 -> "wangmingrong1").
            owner = entry.get("owner") or file_path.parent.name
            self._register_account(account_id, owner, "from folder")
        else:
            logger.warning(f"Skipping invalid credentials file: {file_path.name}")
    
    async def load_state(self) -> None:
        """
        Load runtime state from state.json.
        
        Restores model_to_accounts mapping and account runtime state.
        Creates empty state if file doesn't exist.
        """
        state_path = Path(self._state_file)
        
        if not state_path.exists():
            logger.debug("State file not found, starting with empty state")
            return
        
        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            # Restore global current_account_index
            self._current_account_index = state_data.get("current_account_index", 0)
            
            # Restore model_to_accounts mapping (without next_index)
            for model, data in state_data.get("model_to_accounts", {}).items():
                self._model_to_accounts[model] = ModelAccountList(
                    accounts=data.get("accounts", [])
                )
            
            # Restore account runtime state
            for account_id, data in state_data.get("accounts", {}).items():
                if account_id in self._accounts:
                    account = self._accounts[account_id]
                    # Restore owner if not already set (from credentials.json)
                    if not account.owner:
                        account.owner = data.get("owner", "")
                    account.failures = data.get("failures", 0)
                    account.last_failure_time = data.get("last_failure_time", 0.0)
                    account.models_cached_at = data.get("models_cached_at", 0.0)
                    
                    # Restore quota info if present
                    quota_data = data.get("quota_info")
                    if quota_data:
                        account.quota_info = QuotaInfo(
                            owner=account.owner,
                            account_id=account_id,
                            subscription=quota_data.get("subscription", ""),
                            current_usage=quota_data.get("current_usage", 0),
                            usage_limit=quota_data.get("usage_limit", 0),
                            current_overages=quota_data.get("current_overages", 0),
                            overage_cap=quota_data.get("overage_cap", 0),
                            free_remaining=quota_data.get("free_remaining", 0),
                            overage_remaining=quota_data.get("overage_remaining", 0),
                            total_remaining=quota_data.get("total_remaining", 0),
                            next_reset_epoch=quota_data.get("next_reset_epoch", 0.0),
                            is_exhausted=quota_data.get("is_exhausted", True),
                            last_updated=quota_data.get("last_updated", 0.0),
                            last_error=quota_data.get("last_error"),
                        )
                    
                    stats_data = data.get("stats", {})
                    account.stats = AccountStats(
                        total_requests=stats_data.get("total_requests", 0),
                        successful_requests=stats_data.get("successful_requests", 0),
                        failed_requests=stats_data.get("failed_requests", 0)
                    )
            
            logger.info(f"Loaded state: {len(self._model_to_accounts)} model mappings, {len(self._accounts)} accounts")
        
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    async def _save_state(self) -> None:
        """
        Save runtime state to state.json atomically.
        
        Uses tmp file + rename for atomic write.
        """
        state_data = {
            "current_account_index": self._current_account_index,
            "accounts": {
                account_id: {
                    "owner": account.owner,
                    "failures": account.failures,
                    "last_failure_time": account.last_failure_time,
                    "models_cached_at": account.models_cached_at,
                    "quota_info": account.quota_info.to_dict() if account.quota_info else None,
                    "stats": {
                        "total_requests": account.stats.total_requests,
                        "successful_requests": account.stats.successful_requests,
                        "failed_requests": account.stats.failed_requests
                    }
                }
                for account_id, account in self._accounts.items()
            },
            "model_to_accounts": {
                model: {
                    "accounts": mal.accounts
                }
                for model, mal in self._model_to_accounts.items()
            }
        }
        
        state_path = Path(self._state_file)
        tmp_path = state_path.with_suffix('.json.tmp')
        
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            tmp_path.replace(state_path)
            logger.debug("State saved successfully")
        
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
    
    async def save_state_periodically(self) -> None:
        """
        Background task for periodic state saving.
        
        Saves state every STATE_SAVE_INTERVAL_SECONDS if dirty flag is set.
        """
        while True:
            await asyncio.sleep(STATE_SAVE_INTERVAL_SECONDS)
            
            if self._dirty:
                async with self._lock:
                    await self._save_state()
                    self._dirty = False
    
    async def _initialize_account(self, account_id: str) -> bool:
        """
        Initialize account (lazy initialization).
        
        Creates auth_manager, fetches models, creates cache and resolver.
        Uses a per-account lock to prevent concurrent initialization by
        multiple callers (e.g., get_next_account and poll_quota_once).
        
        Args:
            account_id: Account ID to initialize
        
        Returns:
            True if successful (or already initialized), False otherwise
        """
        account = self._accounts.get(account_id)
        if not account:
            return False
        
        # Fast path: already initialized by another caller
        if account.auth_manager is not None:
            return True
        
        # Per-account lock prevents concurrent initialization
        # (e.g., get_next_account and poll_quota_once both trying)
        init_lock = self._get_init_lock(account_id)
        async with init_lock:
            # Double-check after acquiring lock (another caller may have initialized)
            if account.auth_manager is not None:
                return True
            
            try:
                # Find credentials config for this account
                creds_config = None
                for entry in self._credentials_config:
                    path = entry.get("path", "")
                    expanded_path = Path(path).expanduser()
                    
                    if entry.get("type") == "refresh_token":
                        # Match by deterministic hash for refresh_token type
                        token = entry.get('refresh_token', '')
                        token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
                        if account_id == f"refresh_token_{token_hash}":
                            creds_config = entry
                            break
                    elif str(expanded_path.resolve()) == account_id or (expanded_path.is_dir() and account_id.startswith(str(expanded_path.resolve()) + os.sep)):
                        creds_config = entry
                        break
                
                if not creds_config:
                    logger.error(f"No credentials config found for account: {account_id}")
                    return False
                
                # Create KiroAuthManager based on type
                cred_type = creds_config.get("type")
                if cred_type == "json":
                    auth_manager = KiroAuthManager(
                        creds_file=account_id,
                        profile_arn=creds_config.get("profile_arn"),
                        region=creds_config.get("region", "us-east-1"),
                        api_region=creds_config.get("api_region")
                    )
                elif cred_type == "sqlite":
                    auth_manager = KiroAuthManager(
                        sqlite_db=account_id,
                        profile_arn=creds_config.get("profile_arn"),
                        region=creds_config.get("region", "us-east-1"),
                        api_region=creds_config.get("api_region")
                    )
                elif cred_type == "refresh_token":
                    auth_manager = KiroAuthManager(
                        refresh_token=creds_config.get("refresh_token"),
                        profile_arn=creds_config.get("profile_arn"),
                        region=creds_config.get("region", "us-east-1"),
                        api_region=creds_config.get("api_region")
                    )
                else:
                    logger.error(f"Unknown credential type: {cred_type}")
                    return False
                
                # Get token to verify credentials
                token = await auth_manager.get_access_token()
                
                # Determine if we should fetch models or use static list
                if _is_runtime_endpoint(auth_manager):
                    # New runtime endpoint does not provide /ListAvailableModels (AWS limitation)
                    # Use static list without attempting request
                    logger.debug(f"Account {account_id}: Using static model list for runtime.kiro.dev endpoint")
                    models_list = FALLBACK_MODELS
                else:
                    # Old endpoint - attempt to fetch dynamic model list
                    # Fetch models list with retry + fallback
                    params = {"origin": "AI_EDITOR"}
                    if auth_manager.auth_type == AuthType.KIRO_DESKTOP and auth_manager.profile_arn:
                        params["profileArn"] = auth_manager.profile_arn
                    
                    list_models_url = f"{auth_manager.q_host}/ListAvailableModels"
                    
                    # Use KiroHttpClient for retry logic (3 attempts with exponential backoff)
                    http_client = KiroHttpClient(auth_manager, shared_client=None)
                    
                    try:
                        response = await http_client.request_with_retry(
                            method="GET",
                            url=list_models_url,
                            json_data=None,
                            params=params,
                            stream=False
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            models_list = data.get("models", [])
                        else:
                            # Shouldn't happen (retry handles non-200), but keep for safety
                            raise Exception(f"HTTP {response.status_code}")
                    
                    except Exception as e:
                        # All retries exhausted - use fallback
                        logger.error(f"Failed to fetch models for {account_id} after retries: {e}")
                        logger.warning("Using pre-configured fallback models. Models will be refreshed on next TTL cycle when network recovers.")
                        models_list = FALLBACK_MODELS
                    
                    finally:
                        await http_client.close()
                
                # Create model cache and update
                model_cache = ModelInfoCache()
                await model_cache.update(models_list)
                
                # Add hidden models
                for display_name, internal_id in HIDDEN_MODELS.items():
                    model_cache.add_hidden_model(display_name, internal_id)
                
                # Create model resolver
                model_resolver = ModelResolver(
                    cache=model_cache,
                    hidden_models=HIDDEN_MODELS,
                    aliases=MODEL_ALIASES,
                    hidden_from_list=HIDDEN_FROM_LIST
                )
                
                # Update account
                account.auth_manager = auth_manager
                account.model_cache = model_cache
                account.model_resolver = model_resolver
                account.models_cached_at = time.time()
                
                # Update model_to_accounts mapping
                available_models = model_resolver.get_available_models()
                for model in available_models:
                    if model not in self._model_to_accounts:
                        self._model_to_accounts[model] = ModelAccountList()
                    if account_id not in self._model_to_accounts[model].accounts:
                        self._model_to_accounts[model].accounts.append(account_id)
                
                logger.info(f"Initialized account: {account_id} ({len(available_models)} models)")
                self._dirty = True
                return True
            
            except Exception as e:
                logger.error(f"Failed to initialize account {account_id}: {e}")
                return False
    
    async def _refresh_account_models(self, account_id: str) -> None:
        """
        Refresh model cache for account (TTL refresh).
        
        Uses a per-account lock to prevent concurrent refresh by multiple
        callers. Includes a double-check to skip refresh if another caller
        already refreshed while we waited for the lock.
        
        Args:
            account_id: Account ID to refresh
        """
        account = self._accounts.get(account_id)
        if not account or not account.auth_manager:
            return
        
        # Fast path: cache is still fresh (no refresh needed)
        if account.models_cached_at > 0:
            age = time.time() - account.models_cached_at
            if age <= ACCOUNT_CACHE_TTL:
                return
        
        # Per-account lock prevents concurrent refresh
        init_lock = self._get_init_lock(account_id)
        async with init_lock:
            # Double-check: another caller may have refreshed while we waited
            if account.models_cached_at > 0:
                age = time.time() - account.models_cached_at
                if age <= ACCOUNT_CACHE_TTL:
                    return
            
            # Check if using runtime endpoint (no dynamic model list available)
            if _is_runtime_endpoint(account.auth_manager):
                # Runtime endpoint does not provide /ListAvailableModels
                # Use static list and update cache timestamp
                logger.debug(f"Account {account_id}: Skipping model refresh for runtime.kiro.dev endpoint (using static list)")
                await account.model_cache.update(FALLBACK_MODELS)
                account.models_cached_at = time.time()
                self._dirty = True
                return
            
            # Old endpoint - attempt to fetch dynamic model list
            # Use KiroHttpClient for retry logic
            http_client = KiroHttpClient(account.auth_manager, shared_client=None)
            
            try:
                params = {"origin": "AI_EDITOR"}
                if account.auth_manager.auth_type == AuthType.KIRO_DESKTOP and account.auth_manager.profile_arn:
                    params["profileArn"] = account.auth_manager.profile_arn
                
                list_models_url = f"{account.auth_manager.q_host}/ListAvailableModels"
                
                response = await http_client.request_with_retry(
                    method="GET",
                    url=list_models_url,
                    json_data=None,
                    params=params,
                    stream=False
                )
                
                if response.status_code == 200:
                    data = response.json()
                    models_list = data.get("models", [])
                    await account.model_cache.update(models_list)
                    account.models_cached_at = time.time()
                    
                    # Update model_to_accounts mapping (new models may have appeared)
                    available_models = account.model_resolver.get_available_models()
                    for model in available_models:
                        if model not in self._model_to_accounts:
                            self._model_to_accounts[model] = ModelAccountList()
                        if account_id not in self._model_to_accounts[model].accounts:
                            self._model_to_accounts[model].accounts.append(account_id)
                    
                    logger.debug(f"Refreshed models for {account_id}")
                    self._dirty = True
            
            except Exception as e:
                # All retries exhausted - keep using stale cache
                logger.warning(f"Failed to refresh models for {account_id} after retries: {e}")
            
            finally:
                await http_client.close()
    
    def _is_account_available(self, account: Account) -> bool:
        """
        Check if an account is available for use based on quota.
        
        An account is considered unavailable ONLY when a successful quota query
        confirmed it is exhausted (total_remaining <= 0).
        
        The account is assumed available when:
        - quota_info is None (not polled yet) — we don't block on missing data.
        - the last quota query FAILED (is_quota_unknown) — a failed query yields
          is_exhausted=True as a placeholder, which must never be treated as a
          real "out of credits" signal. GetUsageLimits lives on the legacy Q host
          (q.{region}.amazonaws.com) while inference runs on runtime.kiro.dev, so
          the quota endpoint can be unreachable while requests still succeed.
          Failing closed here would reject every request despite healthy accounts.
        
        Args:
            account: Account to check.
        
        Returns:
            True if account has quota remaining, or quota is unknown/unverified.
        """
        quota = account.quota_info
        if quota is None:
            return True
        if quota.is_quota_unknown:
            return True
        return not quota.is_exhausted
    
    def _get_quota_remaining(self, account: Account) -> int:
        """
        Get the remaining quota for an account, for load-balancing ordering.
        
        Returns a very large sentinel when quota is unknown, so unknown-quota
        accounts are tried first during balancing — they haven't been confirmed
        exhausted yet. Quota counts as unknown when it was never polled, or when
        the last poll failed (a failed query reports total_remaining=0, which
        would otherwise rank a perfectly healthy account last).
        
        Args:
            account: Account to check.
        
        Returns:
            Remaining total credits, or a large sentinel if unknown.
        """
        quota = account.quota_info
        if quota is None or quota.is_quota_unknown:
            return QUOTA_UNKNOWN_SENTINEL
        return quota.total_remaining
    
    def _build_ordered_candidates(self, exclude_accounts: Optional[set] = None) -> List[Account]:
        """
        Build an ordered list of candidate accounts for selection.
        
        Ordering: priority accounts first (in ACCOUNT_PRIORITY order),
        then balanced candidates sorted by remaining quota (descending).
        
        Excludes accounts that are exhausted or in the exclude set.
        Must be called while holding self._lock.
        
        Args:
            exclude_accounts: Set of account IDs to exclude.
        
        Returns:
            Ordered list of candidate Account objects.
        """
        all_accounts = list(self._accounts.values())
        
        priority_candidates: List[Account] = []
        balanced_candidates: List[Account] = []
        
        for account in all_accounts:
            if exclude_accounts and account.id in exclude_accounts:
                continue
            
            if not self._is_account_available(account):
                logger.debug(f"Skipping exhausted account: {account.owner}")
                continue
            
            if account.owner in ACCOUNT_PRIORITY:
                priority_candidates.append(account)
            else:
                balanced_candidates.append(account)
        
        # Sort priority candidates by their position in ACCOUNT_PRIORITY
        priority_candidates.sort(
            key=lambda a: ACCOUNT_PRIORITY.index(a.owner) if a.owner in ACCOUNT_PRIORITY else len(ACCOUNT_PRIORITY)
        )
        
        # Sort balanced candidates by remaining quota (descending = most remaining first)
        balanced_candidates.sort(key=lambda a: self._get_quota_remaining(a), reverse=True)
        
        return priority_candidates + balanced_candidates
    
    def _should_skip_for_circuit_breaker(self, account: Account) -> bool:
        """
        Check if an account should be skipped due to circuit breaker backoff.
        
        Uses exponential backoff with probabilistic retry. Returns True
        if the account is in backoff and should be skipped (subject to
        probabilistic retry chance).
        
        This is a read-only check and does not require holding self._lock.
        Slight staleness in account.failures/last_failure_time is acceptable
        — the worst case is trying an account that just failed (it will
        fail again quickly) or skipping one that just recovered.
        
        Args:
            account: Account to check.
        
        Returns:
            True if the account should be skipped, False if it can be tried.
        """
        if account.failures <= 0:
            return False
        
        time_since_failure = time.time() - account.last_failure_time
        backoff_multiplier = min(2 ** (account.failures - 1), ACCOUNT_MAX_BACKOFF_MULTIPLIER)
        effective_timeout = ACCOUNT_RECOVERY_TIMEOUT * backoff_multiplier
        
        if time_since_failure < effective_timeout:
            if random.random() > ACCOUNT_PROBABILISTIC_RETRY_CHANCE:
                return True
            else:
                logger.info(f"Probabilistic retry for broken account {account.owner}")
        else:
            logger.info(f"Half-Open state for {account.owner} (recovery timeout passed, effective={effective_timeout}s)")
        
        return False
    
    async def get_next_account(self, model: str, exclude_accounts: Optional[set] = None) -> Optional[Account]:
        """
        Get next available account (Priority + Quota-Balanced + Circuit Breaker).
        
        Selection strategy (in order):
        1. Priority accounts: try accounts listed in ACCOUNT_PRIORITY (by owner)
           in listed order. Use the first that is healthy and has quota.
        2. Balanced fallback: if no priority account is available, select the
           account with the most remaining quota (descending sort).
        3. If all accounts exhausted or unhealthy: return None.
        
        Each candidate also passes the Circuit Breaker (exponential backoff
        with probabilistic retry) and lazy initialization checks.
        
        IMPORTANT: The main lock (self._lock) is held ONLY during the fast
        in-memory candidate selection phase. Network I/O (initialization,
        model refresh) is performed OUTSIDE the lock to prevent blocking
        all concurrent requests when one account's init is slow.
        
        Per-account init locks (_get_init_lock) prevent concurrent
        initialization of the same account by multiple callers.
        
        Args:
            model: Model name (will be normalized)
            exclude_accounts: Set of account IDs to exclude (already tried in current failover loop)
        
        Returns:
            Account object or None if no accounts available
        """
        # Special case: single account - bypass Circuit Breaker and quota checks
        # User should see real Kiro API errors instead of generic "Account unavailable"
        if len(self._accounts) == 1:
            account_id = list(self._accounts.keys())[0]
            account = self._accounts[account_id]
            
            if exclude_accounts and account_id in exclude_accounts:
                return None
            
            # Initialize if needed (per-account lock, no main lock held during I/O)
            if account.auth_manager is None:
                success = await self._initialize_account(account_id)
                if not success:
                    return None
            
            # Refresh if needed (per-account lock, no main lock held during I/O)
            if account.models_cached_at > 0:
                age = time.time() - account.models_cached_at
                if age > ACCOUNT_CACHE_TTL:
                    try:
                        await self._refresh_account_models(account_id)
                    except Exception as e:
                        logger.warning(f"Failed to refresh models for {account_id}: {e}")
            
            return account
        
        # Multi-account: build candidate list under lock (fast, no I/O)
        async with self._lock:
            ordered_candidates = self._build_ordered_candidates(exclude_accounts)
        
        if not ordered_candidates:
            logger.warning("No available accounts: all exhausted or excluded")
            return None
        
        # Try candidates in order (OUTSIDE lock — may do network I/O)
        for account in ordered_candidates:
            # Check Circuit Breaker (read-only, slight staleness acceptable)
            if self._should_skip_for_circuit_breaker(account):
                continue
            
            # Lazy initialization (per-account lock, no main lock held during I/O)
            if account.auth_manager is None:
                success = await self._initialize_account(account.id)
                if not success:
                    # Brief lock to update failure state
                    async with self._lock:
                        account.failures += 1
                        self._dirty = True
                    continue
            
            # Check TTL and refresh if needed (per-account lock, no main lock held during I/O)
            if account.models_cached_at > 0:
                age = time.time() - account.models_cached_at
                if age > ACCOUNT_CACHE_TTL:
                    try:
                        await self._refresh_account_models(account.id)
                    except Exception as e:
                        logger.warning(f"Failed to refresh models for {account.id}: {e}")
            
            # Account is suitable!
            logger.debug(f"Selected account: {account.owner} (quota_remaining={self._get_quota_remaining(account)})")
            return account
        
        # All accounts unavailable
        return None
    
    async def report_success(self, account_id: str, model: str) -> None:
        """
        Report successful request (reset failures, update stats, sticky, dynamic learning).
        
        Args:
            account_id: Account ID
            model: Model name
        """
        async with self._lock:
            account = self._accounts.get(account_id)
            if not account:
                return
            
            # Reset failures
            if account.failures > 0:
                account.failures = 0
                self._dirty = True
            
            # Update stats
            account.stats.total_requests += 1
            account.stats.successful_requests += 1
            self._dirty = True
            
            # Dynamic learning: add model to mapping if successful
            # This allows system to learn about new models not in FALLBACK_MODELS
            normalized_model = normalize_model_name(model)
            if normalized_model not in self._model_to_accounts:
                self._model_to_accounts[normalized_model] = ModelAccountList()
                logger.debug(f"Dynamic learning: discovered new model '{normalized_model}'")
            if account_id not in self._model_to_accounts[normalized_model].accounts:
                self._model_to_accounts[normalized_model].accounts.append(account_id)
                logger.debug(f"Dynamic learning: model '{normalized_model}' works on account {account_id}")
                self._dirty = True
            
            # GLOBAL STICKY: Update global current_account_index
            all_account_ids = list(self._accounts.keys())
            try:
                successful_index = all_account_ids.index(account_id)
                if self._current_account_index != successful_index:
                    self._current_account_index = successful_index
                    self._dirty = True
            except ValueError:
                pass
    
    async def report_failure(
        self,
        account_id: str,
        model: str,
        error_type: ErrorType,
        status_code: int,
        reason: Optional[str]
    ) -> None:
        """
        Report failed request (update failures, stats, failover).
        
        Args:
            account_id: Account ID
            model: Model name
            error_type: Error classification (FATAL or RECOVERABLE)
            status_code: HTTP status code
            reason: Error reason from Kiro API
        """
        async with self._lock:
            account = self._accounts.get(account_id)
            if not account:
                return
            
            # Special case: INVALID_MODEL_ID is discovery process, not account failure
            # Account is healthy, model is just not available on this account
            # Log for user visibility but don't penalize account statistics
            if reason == "INVALID_MODEL_ID":
                account.stats.total_requests += 1
                self._dirty = True
                logger.warning(
                    f"Model '{model}' not available on account {account_id}: "
                    f"status={status_code}, reason={reason}"
                )
                return
            
            # Update failure count (only for RECOVERABLE)
            if error_type == ErrorType.RECOVERABLE:
                account.failures += 1
                account.last_failure_time = time.time()
                self._dirty = True
                
                # Calculate backoff for logging
                backoff_multiplier = min(2 ** (account.failures - 1), ACCOUNT_MAX_BACKOFF_MULTIPLIER)
                effective_timeout = ACCOUNT_RECOVERY_TIMEOUT * backoff_multiplier
                logger.warning(
                    f"Account {account_id} failure #{account.failures}: "
                    f"status={status_code}, reason={reason}, "
                    f"cooldown={_format_duration(effective_timeout)}"
                )
            
            # Update stats
            account.stats.total_requests += 1
            account.stats.failed_requests += 1
            self._dirty = True
            
            # GLOBAL STICKY: Do NOT change _current_account_index on failure
            # It only changes on success (GLOBAL sticky behavior)
            # Failover happens through exclude_accounts in get_next_account()
    
    def get_first_account(self) -> Account:
        """
        Get first initialized account (for legacy mode).
        
        Returns:
            First initialized account
        
        Raises:
            RuntimeError: If no initialized accounts available
        """
        for account in self._accounts.values():
            if account.auth_manager is not None:
                return account
        raise RuntimeError("No initialized accounts available")
    
    def get_all_available_models(self) -> List[str]:
        """
        Collect unique models from all initialized accounts.
        
        Used by /v1/models endpoint in account system to show
        all available models across all accounts.
        
        Returns:
            Sorted list of unique model IDs
        """
        all_models = set()
        for account in self._accounts.values():
            if account.model_resolver:
                all_models.update(account.model_resolver.get_available_models())
        return sorted(all_models)
    
    def get_all_quota_info(self) -> List[QuotaInfo]:
        """
        Collect quota info from all accounts for status reporting.
        
        Returns QuotaInfo for each account that has been polled.
        Accounts that haven't been polled yet are not included.
        
        Returns:
            List of QuotaInfo objects, sorted by owner name.
        """
        result = []
        for account in self._accounts.values():
            if account.quota_info is not None:
                result.append(account.quota_info)
        result.sort(key=lambda q: q.owner)
        return result
    
    def get_account_owners(self) -> List[str]:
        """
        Get all account owner labels.
        
        Returns:
            Sorted list of owner labels.
        """
        return sorted(a.owner for a in self._accounts.values() if a.owner)
    
    async def poll_quota_once(self) -> None:
        """
        Query quota for all accounts (single poll cycle).
        
        Ensures every account is initialized (lazy init) before querying,
        so that quota data is available for all accounts — not just the
        one selected at startup. Does not acquire the main lock during
        HTTP queries (only during state updates) to avoid blocking
        request handling.
        """
        # Collect all accounts (snapshot under lock)
        async with self._lock:
            all_accounts = list(self._accounts.items())
        
        if not all_accounts:
            return
        
        logger.debug(f"Polling quota for {len(all_accounts)} account(s)")
        
        # Initialize uninitialized accounts so we can query their quota
        for account_id, account in all_accounts:
            if account.auth_manager is None:
                logger.info(f"Initializing account for quota poll: {account.owner or account_id}")
                try:
                    success = await self._initialize_account(account_id)
                    if not success:
                        logger.warning(f"Failed to initialize {account.owner} for quota poll")
                except Exception as e:
                    logger.warning(f"Error initializing {account.owner} for quota poll: {e}")
        
        # Re-collect initialized accounts
        async with self._lock:
            to_query = [
                (account_id, account)
                for account_id, account in self._accounts.items()
                if account.auth_manager is not None
            ]
        
        # Query each account (no lock held during HTTP)
        for account_id, account in to_query:
            owner = account.owner or account_id
            try:
                quota_info = await query_quota(account.auth_manager, owner, account_id)
                async with self._lock:
                    if account_id in self._accounts:
                        self._accounts[account_id].quota_info = quota_info
                        self._dirty = True
                    if quota_info.is_quota_unknown:
                        # Query failed: figures are placeholders, not a real
                        # "out of credits" signal. The account stays selectable.
                        logger.warning(
                            f"Account {owner} quota unknown (query failed): "
                            f"{quota_info.last_error}. Account remains available; "
                            f"quota limits cannot be enforced until the next "
                            f"successful poll."
                        )
                    elif quota_info.is_exhausted:
                        logger.warning(f"Account {owner} quota exhausted: {quota_info.total_remaining} remaining")
            except Exception as e:
                logger.error(f"Failed to poll quota for {owner}: {e}")
        
        logger.debug("Quota poll cycle complete")
    
    async def poll_quota_periodically(self) -> None:
        """
        Background task for periodic quota polling.
        
        Waits QUOTA_POLL_INITIAL_DELAY seconds before first poll, then
        polls every QUOTA_POLL_INTERVAL seconds. Runs indefinitely.
        """
        logger.info(f"Quota poller started: initial_delay={QUOTA_POLL_INITIAL_DELAY}s, interval={QUOTA_POLL_INTERVAL}s")
        await asyncio.sleep(QUOTA_POLL_INITIAL_DELAY)
        
        while True:
            try:
                await self.poll_quota_once()
            except Exception as e:
                logger.error(f"Quota poll cycle error: {e}")
            
            await asyncio.sleep(QUOTA_POLL_INTERVAL)

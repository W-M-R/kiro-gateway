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
Kiro API quota (usage limits) client.

Queries the AWS CodeWhisperer GetUsageLimits API to retrieve per-account
quota information, including free usage limits and overage caps.

The GetUsageLimits endpoint is only available on the legacy Q API host
(https://q.{region}.amazonaws.com), NOT on the new runtime.kiro.dev host.

API details (reverse-engineered from kiro-cli binary):
    - Endpoint: POST https://q.{region}.amazonaws.com/GetUsageLimits
    - Header:   x-amz-target: AmazonCodeWhispererService.GetUsageLimits
    - Body:     {"profileArn": "..."} (optional for some accounts)
    - Response: {
        "usageBreakdownList": [{
            "currentUsage": int,
            "usageLimit": int,
            "currentOverages": int,
            "overageCap": int,
            ...
        }],
        "subscriptionInfo": {"subscriptionTitle": str, ...},
        "nextDateReset": float (epoch seconds)
      }
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from kiro.auth import KiroAuthManager
from kiro.config import get_httpx_verify_config
from kiro.utils import _build_kiro_user_agent_headers


# AWS operation target header for GetUsageLimits.
# Note: This is the non-streaming CodeWhisperer service, not the streaming one.
_GET_USAGE_LIMITS_TARGET = "AmazonCodeWhispererService.GetUsageLimits"

# Path appended to the Q API host for the GetUsageLimits operation.
_GET_USAGE_LIMITS_PATH = "/GetUsageLimits"

# Request timeout for quota queries (seconds). Quota API is lightweight.
_QUOTA_QUERY_TIMEOUT = 30.0

# Sentinel used as "remaining quota" when an account's quota is unknown (never
# polled, or the last poll failed). Large enough to sort such accounts ahead of
# any account with real quota figures, since they are not confirmed exhausted.
QUOTA_UNKNOWN_SENTINEL: int = 10 ** 12


def _build_q_host(q_host: str) -> str:
    """
    Build the legacy Q API host URL from a runtime/legacy host.

    GetUsageLimits is NOT available on runtime.{region}.kiro.dev.
    It must be queried via the legacy host https://q.{region}.amazonaws.com.

    Args:
        q_host: The auth manager's q_host value (may be runtime or legacy).

    Returns:
        Legacy Q API host URL (https://q.{region}.amazonaws.com).

    Examples:
        >>> _build_q_host("https://runtime.us-east-1.kiro.dev")
        'https://q.us-east-1.amazonaws.com'
        >>> _build_q_host("https://q.us-east-1.amazonaws.com")
        'https://q.us-east-1.amazonaws.com'
    """
    if "://runtime." in q_host:
        # runtime.{region}.kiro.dev → q.{region}.amazonaws.com
        return q_host.replace("://runtime.", "://q.").replace(".kiro.dev", ".amazonaws.com")
    return q_host


@dataclass
class QuotaInfo:
    """
    Parsed quota information for a single Kiro account.

    Attributes:
        owner: Human-readable account owner label (e.g. "wangmingrong1").
        account_id: Internal account ID (file path or hash).
        subscription: Subscription plan title (e.g. "KIRO PRO+").
        current_usage: Current free-tier usage count.
        usage_limit: Free-tier usage limit (included credits).
        current_overages: Current overage usage count (beyond free tier).
        overage_cap: Maximum allowed overage usage.
        free_remaining: Remaining free-tier credits: max(0, usage_limit - current_usage).
        overage_remaining: Remaining overage credits: max(0, overage_cap - current_overages).
        total_remaining: Total remaining credits: free_remaining + overage_remaining.
        next_reset_epoch: Epoch timestamp (seconds) when quota resets.
        is_exhausted: True if total_remaining <= 0.
        last_updated: Epoch timestamp of last successful query.
        last_error: Error message if last query failed, None otherwise.
    """
    owner: str
    account_id: str
    subscription: str = ""
    current_usage: int = 0
    usage_limit: int = 0
    current_overages: int = 0
    overage_cap: int = 0
    free_remaining: int = 0
    overage_remaining: int = 0
    total_remaining: int = 0
    next_reset_epoch: float = 0.0
    is_exhausted: bool = True
    last_updated: float = 0.0
    last_error: Optional[str] = None

    @property
    def is_quota_unknown(self) -> bool:
        """
        Whether the quota figures are untrustworthy because the query failed.

        On a failed GetUsageLimits call, query_quota() returns a QuotaInfo with
        last_error set and is_exhausted=True. That is NOT a real "out of credits"
        signal, so callers must not use it to block account selection: the quota
        endpoint lives on the legacy Q host and can be unreachable while the
        inference endpoint works fine.

        Returns:
            True if the last quota query failed (figures are meaningless).
        """
        return self.last_error is not None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dictionary for API responses."""
        return {
            "owner": self.owner,
            "account_id": self.account_id,
            "subscription": self.subscription,
            "current_usage": self.current_usage,
            "usage_limit": self.usage_limit,
            "current_overages": self.current_overages,
            "overage_cap": self.overage_cap,
            "free_remaining": self.free_remaining,
            "overage_remaining": self.overage_remaining,
            "total_remaining": self.total_remaining,
            "next_reset_epoch": self.next_reset_epoch,
            "next_reset_iso": (
                datetime.fromtimestamp(self.next_reset_epoch, tz=timezone.utc).isoformat()
                if self.next_reset_epoch > 0 else None
            ),
            "is_exhausted": self.is_exhausted,
            "is_quota_unknown": self.is_quota_unknown,
            "last_updated": self.last_updated,
            "last_error": self.last_error,
        }


def parse_usage_response(
    raw: Dict[str, Any],
    owner: str,
    account_id: str,
    now: float,
) -> QuotaInfo:
    """
    Parse a raw GetUsageLimits JSON response into a QuotaInfo object.

    Handles missing fields gracefully by defaulting to zero, and tolerates
    an empty usageBreakdownList (treats account as exhausted).

    Args:
        raw: Parsed JSON response from GetUsageLimits API.
        owner: Account owner label.
        account_id: Internal account ID.
        now: Current epoch timestamp for last_updated.

    Returns:
        QuotaInfo with computed remaining/exhausted fields.

    Examples:
        >>> raw = {
        ...     "usageBreakdownList": [{
        ...         "currentUsage": 100,
        ...         "usageLimit": 2000,
        ...         "currentOverages": 50,
        ...         "overageCap": 10000,
        ...     }],
        ...     "subscriptionInfo": {"subscriptionTitle": "KIRO PRO+"},
        ...     "nextDateReset": 1785542400.0,
        ... }
        >>> info = parse_usage_response(raw, "test", "/path", 1700000000.0)
        >>> info.free_remaining
        1900
        >>> info.overage_remaining
        9950
        >>> info.total_remaining
        11850
        >>> info.is_exhausted
        False
    """
    breakdown_list: List[Dict[str, Any]] = raw.get("usageBreakdownList", [])
    breakdown: Dict[str, Any] = breakdown_list[0] if breakdown_list else {}

    current_usage = int(breakdown.get("currentUsage", 0))
    usage_limit = int(breakdown.get("usageLimit", 0))
    current_overages = int(breakdown.get("currentOverages", 0))
    overage_cap = int(breakdown.get("overageCap", 0))

    free_remaining = max(0, usage_limit - current_usage)
    overage_remaining = max(0, overage_cap - current_overages)
    total_remaining = free_remaining + overage_remaining

    subscription_info: Dict[str, Any] = raw.get("subscriptionInfo", {})
    subscription = str(subscription_info.get("subscriptionTitle", ""))

    next_reset = float(raw.get("nextDateReset", 0.0))

    return QuotaInfo(
        owner=owner,
        account_id=account_id,
        subscription=subscription,
        current_usage=current_usage,
        usage_limit=usage_limit,
        current_overages=current_overages,
        overage_cap=overage_cap,
        free_remaining=free_remaining,
        overage_remaining=overage_remaining,
        total_remaining=total_remaining,
        next_reset_epoch=next_reset,
        is_exhausted=(total_remaining <= 0),
        last_updated=now,
        last_error=None,
    )


async def query_quota(
    auth_manager: KiroAuthManager,
    owner: str,
    account_id: str,
) -> QuotaInfo:
    """
    Query GetUsageLimits API for a single account.

    Fetches an access token, builds the request, and parses the response.
    On failure, returns a QuotaInfo with last_error set and is_exhausted=True
    so that the account is skipped during load balancing.

    Args:
        auth_manager: Initialized KiroAuthManager with valid credentials.
        owner: Account owner label for the result.
        account_id: Internal account ID for the result.

    Returns:
        QuotaInfo with parsed quota data or error details.

    Raises:
        Never raises — all exceptions are caught and stored in last_error.
    """
    import time

    now = time.time()

    try:
        token = await auth_manager.get_access_token()
        base_host = _build_q_host(auth_manager.q_host)
        url = f"{base_host}{_GET_USAGE_LIMITS_PATH}"

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-amz-json-1.0",
            "x-amz-target": _GET_USAGE_LIMITS_TARGET,
            "x-amzn-codewhisperer-optout": "true",
            "amz-sdk-invocation-id": str(uuid.uuid4()),
            "amz-sdk-request": "attempt=1; max=3",
        }
        headers.update(_build_kiro_user_agent_headers(auth_manager.fingerprint))

        body: Dict[str, Any] = {}
        if auth_manager.profile_arn:
            body["profileArn"] = auth_manager.profile_arn

        verify = get_httpx_verify_config()
        async with httpx.AsyncClient(verify=verify, timeout=_QUOTA_QUERY_TIMEOUT) as client:
            response = await client.post(url, headers=headers, json=body)

        if response.status_code != 200:
            error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            logger.warning(f"Quota query failed for {owner}: {error_msg}")
            return QuotaInfo(
                owner=owner,
                account_id=account_id,
                is_exhausted=True,
                last_updated=now,
                last_error=error_msg,
            )

        raw = response.json()
        info = parse_usage_response(raw, owner, account_id, now)
        logger.info(
            f"Quota [{owner}] {info.subscription}: "
            f"used={info.current_usage}/{info.usage_limit} (free), "
            f"overage={info.current_overages}/{info.overage_cap}, "
            f"remaining={info.total_remaining}, "
            f"exhausted={info.is_exhausted}"
        )
        return info

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(f"Quota query error for {owner}: {error_msg}")
        return QuotaInfo(
            owner=owner,
            account_id=account_id,
            is_exhausted=True,
            last_updated=now,
            last_error=error_msg,
        )

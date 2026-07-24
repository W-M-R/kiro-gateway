# -*- coding: utf-8 -*-

"""
Tests for quota-based load balancing in kiro/account_manager.py.

Covers the priority + balanced selection logic in get_next_account():
- Priority accounts are tried first (in ACCOUNT_PRIORITY order)
- When priority accounts exhausted, fall back to balanced (most remaining)
- Exhausted accounts (quota depleted) are skipped
- exclude_accounts interaction
- Circuit Breaker interaction
- Single account bypass
- All accounts exhausted returns None
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kiro.account_manager import Account, AccountManager, AccountStats
from kiro.quota import QuotaInfo


def _make_account(
    account_id: str,
    owner: str,
    initialized: bool = True,
    total_remaining: int = 5000,
    exhausted: bool = False,
    failures: int = 0,
) -> Account:
    """
    Create a test Account with optional pre-set quota info.
    
    Args:
        account_id: Account ID string.
        owner: Owner label.
        initialized: If True, auth_manager is set (non-None).
        total_remaining: Total remaining quota to simulate.
        exhausted: If True, override total_remaining to 0 and is_exhausted to True.
        failures: Number of consecutive failures for Circuit Breaker.
    
    Returns:
        Configured Account object.
    """
    account = Account(
        id=account_id,
        owner=owner,
        auth_manager=MagicMock() if initialized else None,
        failures=failures,
        last_failure_time=time.time() if failures > 0 else 0.0,  # Recent failure for Circuit Breaker
        models_cached_at=time.time(),  # Fresh cache to avoid refresh attempts
    )
    if initialized:
        account.quota_info = QuotaInfo(
            owner=owner,
            account_id=account_id,
            total_remaining=0 if exhausted else total_remaining,
            is_exhausted=exhausted,
            current_usage=0,
            usage_limit=2000,
            current_overages=0,
            overage_cap=10000,
        )
    return account


def _make_manager(accounts_dict: dict) -> AccountManager:
    """
    Create an AccountManager with pre-populated accounts.
    
    Args:
        accounts_dict: {account_id: Account} mapping.
    
    Returns:
        AccountManager with accounts set (bypasses credential loading).
    """
    manager = AccountManager.__new__(AccountManager)
    manager._accounts = accounts_dict
    manager._model_to_accounts = {}
    manager._lock = __import__("asyncio").Lock()
    manager._dirty = False
    manager._credentials_config = []
    manager._current_account_index = 0
    return manager


class TestPrioritySelection:
    """Tests for priority account selection."""

    @pytest.mark.asyncio
    async def test_priority_account_selected_first(self):
        """Priority account should be selected over non-priority accounts."""
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", ["wangmingrong1"]):
            accounts = {
                "/a": _make_account("/a", "anjiahao", total_remaining=9000),
                "/w": _make_account("/w", "wangmingrong1", total_remaining=100),
            }
            manager = _make_manager(accounts)
            account = await manager.get_next_account("claude-sonnet-4.5")
            assert account is not None
            assert account.owner == "wangmingrong1"

    @pytest.mark.asyncio
    async def test_priority_order_respected(self):
        """Multiple priority accounts should be tried in listed order."""
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", ["wangmingrong1", "anjiahao"]):
            accounts = {
                "/a": _make_account("/a", "anjiahao", total_remaining=9000),
                "/w": _make_account("/w", "wangmingrong1", total_remaining=100),
            }
            manager = _make_manager(accounts)
            account = await manager.get_next_account("claude-sonnet-4.5")
            assert account.owner == "wangmingrong1"

    @pytest.mark.asyncio
    async def test_priority_exhausted_falls_to_next_priority(self):
        """Exhausted priority account should fall to next priority account."""
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", ["wangmingrong1", "anjiahao"]):
            accounts = {
                "/a": _make_account("/a", "anjiahao", total_remaining=5000),
                "/w": _make_account("/w", "wangmingrong1", exhausted=True),
            }
            manager = _make_manager(accounts)
            account = await manager.get_next_account("claude-sonnet-4.5")
            assert account.owner == "anjiahao"


class TestBalancedSelection:
    """Tests for quota-balanced selection (fallback mode)."""

    @pytest.mark.asyncio
    async def test_balanced_selects_most_remaining(self):
        """When no priority accounts, select account with most remaining quota."""
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", []):
            accounts = {
                "/a": _make_account("/a", "anjiahao", total_remaining=3000),
                "/m": _make_account("/m", "mazhuang", total_remaining=9000),
                "/g": _make_account("/g", "gaohedong", total_remaining=5000),
            }
            manager = _make_manager(accounts)
            account = await manager.get_next_account("claude-sonnet-4.5")
            assert account.owner == "mazhuang"  # Most remaining

    @pytest.mark.asyncio
    async def test_priority_all_exhausted_falls_to_balanced(self):
        """When all priority accounts exhausted, fall back to balanced."""
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", ["wangmingrong1"]):
            accounts = {
                "/w": _make_account("/w", "wangmingrong1", exhausted=True),
                "/a": _make_account("/a", "anjiahao", total_remaining=3000),
                "/m": _make_account("/m", "mazhuang", total_remaining=9000),
            }
            manager = _make_manager(accounts)
            account = await manager.get_next_account("claude-sonnet-4.5")
            assert account.owner == "mazhuang"  # Most remaining among balanced


class TestExhaustedSkipping:
    """Tests for skipping exhausted accounts."""

    @pytest.mark.asyncio
    async def test_exhausted_account_skipped(self):
        """Exhausted accounts should be skipped entirely."""
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", []):
            accounts = {
                "/a": _make_account("/a", "anjiahao", exhausted=True),
                "/m": _make_account("/m", "mazhuang", total_remaining=5000),
            }
            manager = _make_manager(accounts)
            account = await manager.get_next_account("claude-sonnet-4.5")
            assert account.owner == "mazhuang"

    @pytest.mark.asyncio
    async def test_all_exhausted_returns_none(self):
        """When all accounts are exhausted, return None."""
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", []):
            accounts = {
                "/a": _make_account("/a", "anjiahao", exhausted=True),
                "/m": _make_account("/m", "mazhuang", exhausted=True),
            }
            manager = _make_manager(accounts)
            account = await manager.get_next_account("claude-sonnet-4.5")
            assert account is None

    @pytest.mark.asyncio
    async def test_no_quota_info_treated_as_available(self):
        """Accounts without quota_info (not yet polled) should be available."""
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", []):
            account_no_quota = Account(
                id="/nq",
                owner="not_polled",
                auth_manager=MagicMock(),
                models_cached_at=time.time(),
            )
            accounts = {
                "/nq": account_no_quota,
                "/a": _make_account("/a", "anjiahao", total_remaining=5000),
            }
            manager = _make_manager(accounts)
            # No-quota account should be tried first (treated as infinite remaining)
            account = await manager.get_next_account("claude-sonnet-4.5")
            assert account.owner == "not_polled"


class TestExcludeAccounts:
    """Tests for exclude_accounts interaction."""

    @pytest.mark.asyncio
    async def test_exclude_skips_accounts(self):
        """Excluded accounts should be skipped during selection."""
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", []):
            accounts = {
                "/a": _make_account("/a", "anjiahao", total_remaining=9000),
                "/m": _make_account("/m", "mazhuang", total_remaining=5000),
            }
            manager = _make_manager(accounts)
            account = await manager.get_next_account(
                "claude-sonnet-4.5",
                exclude_accounts={"/a"}
            )
            assert account.owner == "mazhuang"

    @pytest.mark.asyncio
    async def test_all_excluded_returns_none(self):
        """When all accounts are excluded, return None."""
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", []):
            accounts = {
                "/a": _make_account("/a", "anjiahao", total_remaining=9000),
            }
            manager = _make_manager(accounts)
            account = await manager.get_next_account(
                "claude-sonnet-4.5",
                exclude_accounts={"/a"}
            )
            assert account is None


class TestSingleAccountBypass:
    """Tests for single account bypass (no Circuit Breaker, no quota check)."""

    @pytest.mark.asyncio
    async def test_single_account_bypasses_quota_check(self):
        """Single account should be returned even if exhausted."""
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", []):
            accounts = {
                "/a": _make_account("/a", "anjiahao", exhausted=True),
            }
            manager = _make_manager(accounts)
            account = await manager.get_next_account("claude-sonnet-4.5")
            assert account is not None
            assert account.owner == "anjiahao"

    @pytest.mark.asyncio
    async def test_single_account_uninitialized_tries_init(self):
        """Single uninitialized account should attempt initialization."""
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", []):
            accounts = {
                "/a": Account(id="/a", owner="anjiahao"),
            }
            manager = _make_manager(accounts)
            with patch.object(manager, "_initialize_account", new=AsyncMock(return_value=True)):
                account = await manager.get_next_account("claude-sonnet-4.5")
                assert account is not None

    @pytest.mark.asyncio
    async def test_single_account_init_fails_returns_none(self):
        """Single account that fails initialization should return None."""
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", []):
            accounts = {
                "/a": Account(id="/a", owner="anjiahao"),
            }
            manager = _make_manager(accounts)
            with patch.object(manager, "_initialize_account", new=AsyncMock(return_value=False)):
                account = await manager.get_next_account("claude-sonnet-4.5")
                assert account is None


class TestCircuitBreakerInteraction:
    """Tests for Circuit Breaker interaction with quota selection."""

    @pytest.mark.asyncio
    async def test_account_with_failures_skipped_probabilistically(self):
        """Account with recent failures should be skipped (probabilistic retry low)."""
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", []), \
             patch("kiro.account_manager.ACCOUNT_PROBABILISTIC_RETRY_CHANCE", 0.0):
            accounts = {
                "/a": _make_account("/a", "anjiahao", total_remaining=9000, failures=5),
                "/m": _make_account("/m", "mazhuang", total_remaining=5000, failures=0),
            }
            manager = _make_manager(accounts)
            account = await manager.get_next_account("claude-sonnet-4.5")
            # anjiahao has more quota but has failures — should skip to mazhuang
            assert account.owner == "mazhuang"

    @pytest.mark.asyncio
    async def test_account_with_failures_but_most_quota_may_retry(self):
        """Account with failures but 100% probabilistic retry should be tried."""
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", []), \
             patch("kiro.account_manager.ACCOUNT_PROBABILISTIC_RETRY_CHANCE", 1.0):
            accounts = {
                "/a": _make_account("/a", "anjiahao", total_remaining=9000, failures=1),
                "/m": _make_account("/m", "mazhuang", total_remaining=5000, failures=0),
            }
            manager = _make_manager(accounts)
            account = await manager.get_next_account("claude-sonnet-4.5")
            # With 100% retry chance, anjiahao (more quota) should be selected
            assert account.owner == "anjiahao"


class TestQuotaPolling:
    """Tests for quota polling methods."""

    @pytest.mark.asyncio
    async def test_poll_quota_once_updates_quota_info(self):
        """poll_quota_once should update quota_info for initialized accounts."""
        account = Account(
            id="/a",
            owner="anjiahao",
            auth_manager=MagicMock(),
            models_cached_at=time.time(),
        )
        manager = _make_manager({"/a": account})

        mock_quota = QuotaInfo(
            owner="anjiahao",
            account_id="/a",
            total_remaining=5000,
            is_exhausted=False,
        )

        with patch("kiro.account_manager.query_quota", new=AsyncMock(return_value=mock_quota)):
            await manager.poll_quota_once()

        assert manager._accounts["/a"].quota_info is not None
        assert manager._accounts["/a"].quota_info.total_remaining == 5000
        assert manager._dirty is True

    @pytest.mark.asyncio
    async def test_poll_quota_once_skips_uninitialized(self):
        """poll_quota_once should skip accounts without auth_manager."""
        account = Account(id="/a", owner="anjiahao")
        manager = _make_manager({"/a": account})

        with patch("kiro.account_manager.query_quota", new=AsyncMock()) as mock_query:
            await manager.poll_quota_once()
            mock_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_poll_quota_once_handles_errors(self):
        """poll_quota_once should not crash on query errors."""
        account = Account(
            id="/a",
            owner="anjiahao",
            auth_manager=MagicMock(),
            models_cached_at=time.time(),
        )
        manager = _make_manager({"/a": account})

        with patch("kiro.account_manager.query_quota", new=AsyncMock(side_effect=Exception("Network error"))):
            # Should not raise
            await manager.poll_quota_once()
            # Account quota_info should remain None (not updated)
            assert manager._accounts["/a"].quota_info is None


class TestQuotaInfoGetters:
    """Tests for get_all_quota_info and get_account_owners."""

    def test_get_all_quota_info_returns_sorted(self):
        """get_all_quota_info should return quota infos sorted by owner."""
        accounts = {
            "/z": _make_account("/z", "zoe", total_remaining=100),
            "/a": _make_account("/a", "alice", total_remaining=200),
            "/m": _make_account("/m", "mike", total_remaining=300),
        }
        manager = _make_manager(accounts)
        result = manager.get_all_quota_info()
        assert len(result) == 3
        assert result[0].owner == "alice"
        assert result[1].owner == "mike"
        assert result[2].owner == "zoe"

    def test_get_all_quota_info_excludes_unpolled(self):
        """get_all_quota_info should exclude accounts without quota_info."""
        accounts = {
            "/a": _make_account("/a", "alice", total_remaining=200),
            "/nq": Account(id="/nq", owner="not_polled", auth_manager=MagicMock()),
        }
        manager = _make_manager(accounts)
        result = manager.get_all_quota_info()
        assert len(result) == 1
        assert result[0].owner == "alice"

    def test_get_account_owners_returns_sorted(self):
        """get_account_owners should return sorted owner labels."""
        accounts = {
            "/z": _make_account("/z", "zoe", total_remaining=100),
            "/a": _make_account("/a", "alice", total_remaining=200),
            "/empty": Account(id="/empty", owner=""),
        }
        manager = _make_manager(accounts)
        owners = manager.get_account_owners()
        assert owners == ["alice", "zoe"]

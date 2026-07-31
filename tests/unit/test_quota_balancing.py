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

Also covers the lock-free-during-I/O refactor:
- get_next_account does NOT hold self._lock during network I/O
- Per-account init locks prevent concurrent initialization
- Double-check pattern skips redundant init/refresh
- Concurrent requests proceed when one account's init is slow
"""

import asyncio
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
    quota_error: str = None,
    no_quota_info: bool = False,
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
        quota_error: If set, simulates a FAILED quota query. query_quota() returns
            is_exhausted=True with last_error populated on failure, so this also
            forces total_remaining=0 and is_exhausted=True to mirror reality.
        no_quota_info: If True, leave quota_info as None (never polled).
    
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
    if initialized and not no_quota_info:
        if quota_error is not None:
            # Mirror query_quota()'s failure return value exactly
            account.quota_info = QuotaInfo(
                owner=owner,
                account_id=account_id,
                total_remaining=0,
                is_exhausted=True,
                last_error=quota_error,
            )
        else:
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
    manager._init_locks = {}
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


class TestQuotaQueryFailureFailsOpen:
    """
    Tests that a FAILED quota query never blocks account selection.

    query_quota() returns is_exhausted=True with last_error set when the
    GetUsageLimits call fails. That placeholder must not be treated as a real
    "out of credits" signal: GetUsageLimits only exists on the legacy Q host
    (q.{region}.amazonaws.com) while inference runs on runtime.kiro.dev, so the
    quota endpoint can be unreachable while requests still succeed. Failing
    closed here would reject every request despite healthy accounts.
    """

    @pytest.mark.asyncio
    async def test_failed_quota_query_keeps_account_available(self):
        """An account whose quota query failed must remain selectable."""
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", []):
            accounts = {
                "/a": _make_account("/a", "anjiahao", quota_error="HTTP 500: internal error"),
            }
            manager = _make_manager(accounts)
            account = await manager.get_next_account("claude-sonnet-4.5")
            assert account is not None
            assert account.owner == "anjiahao"

    @pytest.mark.asyncio
    async def test_all_quota_queries_failed_still_serves_requests(self):
        """
        Regression: total quota-API outage must not take the gateway down.

        Previously every account got is_exhausted=True and selection returned
        None, rejecting all traffic despite healthy accounts.
        """
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", []):
            accounts = {
                f"/{n}": _make_account(f"/{n}", n, quota_error="ConnectTimeout")
                for n in ["wangmingrong1", "anjiahao", "gaohedong", "guoshengyuan", "mazhuang"]
            }
            manager = _make_manager(accounts)
            account = await manager.get_next_account("claude-sonnet-4.5")
            assert account is not None, "quota-API outage must not reject all requests"

    @pytest.mark.asyncio
    async def test_real_exhaustion_still_skipped(self):
        """
        Guard against over-correction: a SUCCESSFUL query reporting zero quota
        must still exclude the account.
        """
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", []):
            accounts = {
                "/a": _make_account("/a", "anjiahao", exhausted=True),
                "/m": _make_account("/m", "mazhuang", total_remaining=5000),
            }
            manager = _make_manager(accounts)
            account = await manager.get_next_account("claude-sonnet-4.5")
            assert account.owner == "mazhuang"

    @pytest.mark.asyncio
    async def test_all_really_exhausted_still_returns_none(self):
        """Genuine exhaustion of every account must still return None."""
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", []):
            accounts = {
                "/a": _make_account("/a", "anjiahao", exhausted=True),
                "/m": _make_account("/m", "mazhuang", exhausted=True),
            }
            manager = _make_manager(accounts)
            assert await manager.get_next_account("claude-sonnet-4.5") is None

    @pytest.mark.asyncio
    async def test_mixed_failed_exhausted_and_healthy(self):
        """
        Mixed state: failed-query accounts stay candidates, the genuinely
        exhausted one is dropped.
        """
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", []):
            accounts = {
                "/f1": _make_account("/f1", "failed1", quota_error="timeout"),
                "/f2": _make_account("/f2", "failed2", quota_error="timeout"),
                "/ex": _make_account("/ex", "really_exhausted", exhausted=True),
                "/ok": _make_account("/ok", "healthy", total_remaining=5000),
            }
            manager = _make_manager(accounts)
            candidates = manager._build_ordered_candidates(None)
            owners = [c.owner for c in candidates]
            assert "really_exhausted" not in owners
            assert set(owners) == {"failed1", "failed2", "healthy"}

    def test_get_quota_remaining_uses_sentinel_for_failed_query(self):
        """
        A failed query reports total_remaining=0, which would rank a healthy
        account last. It must use the unknown sentinel instead.
        """
        from kiro.quota import QUOTA_UNKNOWN_SENTINEL

        manager = _make_manager({})
        failed = _make_account("/f", "failed", quota_error="HTTP 503")
        assert manager._get_quota_remaining(failed) == QUOTA_UNKNOWN_SENTINEL

    def test_get_quota_remaining_uses_sentinel_when_never_polled(self):
        """An unpolled account also has unknown quota."""
        from kiro.quota import QUOTA_UNKNOWN_SENTINEL

        manager = _make_manager({})
        unpolled = _make_account("/n", "unpolled", no_quota_info=True)
        assert manager._get_quota_remaining(unpolled) == QUOTA_UNKNOWN_SENTINEL

    def test_get_quota_remaining_uses_real_value_on_success(self):
        """A successful query must use its actual remaining figure."""
        manager = _make_manager({})
        healthy = _make_account("/h", "healthy", total_remaining=1234)
        assert manager._get_quota_remaining(healthy) == 1234

    def test_is_account_available_matrix(self):
        """Availability across all quota states."""
        manager = _make_manager({})
        cases = [
            (_make_account("/n", "unpolled", no_quota_info=True), True, "never polled"),
            (_make_account("/f", "failed", quota_error="err"), True, "query failed"),
            (_make_account("/e", "exhausted", exhausted=True), False, "really exhausted"),
            (_make_account("/h", "healthy", total_remaining=100), True, "healthy"),
        ]
        for account, expected, label in cases:
            assert manager._is_account_available(account) is expected, label

    @pytest.mark.asyncio
    async def test_failed_query_account_ranks_before_low_quota_account(self):
        """
        Unknown-quota accounts are not confirmed exhausted, so they sort ahead
        of accounts with real (low) remaining quota.
        """
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", []):
            accounts = {
                "/low": _make_account("/low", "low_quota", total_remaining=1),
                "/f": _make_account("/f", "failed", quota_error="timeout"),
            }
            manager = _make_manager(accounts)
            account = await manager.get_next_account("claude-sonnet-4.5")
            assert account.owner == "failed"

    @pytest.mark.asyncio
    async def test_poll_logs_unknown_not_exhausted_on_failure(self):
        """
        A failed poll must not log the misleading "quota exhausted: 0 remaining"
        message, which masks the real cause.
        """
        accounts = {"/a": _make_account("/a", "anjiahao", total_remaining=5000)}
        manager = _make_manager(accounts)
        failure = QuotaInfo(
            owner="anjiahao",
            account_id="/a",
            is_exhausted=True,
            last_error="HTTP 500: boom",
        )
        with patch("kiro.account_manager.query_quota", new=AsyncMock(return_value=failure)), \
             patch("kiro.account_manager.logger") as mock_logger:
            await manager.poll_quota_once()
            messages = " ".join(str(c) for c in mock_logger.warning.call_args_list)

        assert "quota unknown" in messages
        assert "HTTP 500: boom" in messages
        assert "quota exhausted" not in messages


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


# ==============================================================================
# Lock-Free-During-I/O Tests
# ==============================================================================

class TestLockNotHeldDuringInit:
    """Tests that get_next_account does NOT hold self._lock during network I/O."""

    @pytest.mark.asyncio
    async def test_lock_released_during_initialize_account(self):
        """
        The main lock must NOT be held while _initialize_account runs.

        What it does: Starts get_next_account with an uninitialized account,
        then attempts to acquire the lock from another task while init is
        in progress. The lock should be available (not held by get_next_account).
        """
        account = Account(id="/a", owner="alice", auth_manager=None,
                          models_cached_at=0.0)
        manager = _make_manager({"/a": account})

        init_started = asyncio.Event()
        lock_acquired_during_init = asyncio.Event()

        async def slow_init(account_id):
            init_started.set()
            # Give the other task time to try acquiring the lock
            await asyncio.sleep(0.1)
            # Simulate successful init
            account.auth_manager = MagicMock()
            account.models_cached_at = time.time()
            return True

        async def try_acquire_lock():
            await init_started.wait()
            async with manager._lock:
                lock_acquired_during_init.set()

        with patch.object(manager, "_initialize_account", side_effect=slow_init):
            task1 = asyncio.create_task(manager.get_next_account("claude-sonnet-4.5"))
            task2 = asyncio.create_task(try_acquire_lock())

            await asyncio.wait_for(asyncio.gather(task1, task2), timeout=5.0)

        assert lock_acquired_during_init.is_set(), \
            "Lock was held during _initialize_account — concurrent requests would be blocked!"

    @pytest.mark.asyncio
    async def test_lock_released_during_refresh_models(self):
        """
        The main lock must NOT be held while _refresh_account_models runs.

        What it does: Creates an account with an expired cache TTL, starts
        get_next_account, then attempts to acquire the lock from another task
        while refresh is in progress.
        """
        old_time = time.time() - 50000  # Expired (TTL is 43200s = 12h)
        account = Account(
            id="/a", owner="alice",
            auth_manager=MagicMock(),
            models_cached_at=old_time,
        )
        account.quota_info = QuotaInfo(
            owner="alice", account_id="/a", total_remaining=5000,
            is_exhausted=False, current_usage=0, usage_limit=2000,
            current_overages=0, overage_cap=10000,
        )
        manager = _make_manager({"/a": account})

        refresh_started = asyncio.Event()
        lock_acquired_during_refresh = asyncio.Event()

        async def slow_refresh(account_id):
            refresh_started.set()
            await asyncio.sleep(0.1)
            account.models_cached_at = time.time()
            return

        async def try_acquire_lock():
            await refresh_started.wait()
            async with manager._lock:
                lock_acquired_during_refresh.set()

        with patch.object(manager, "_refresh_account_models", side_effect=slow_refresh):
            task1 = asyncio.create_task(manager.get_next_account("claude-sonnet-4.5"))
            task2 = asyncio.create_task(try_acquire_lock())

            await asyncio.wait_for(asyncio.gather(task1, task2), timeout=5.0)

        assert lock_acquired_during_refresh.is_set(), \
            "Lock was held during _refresh_account_models — concurrent requests would be blocked!"

    @pytest.mark.asyncio
    async def test_concurrent_get_next_account_both_proceed(self):
        """
        Two concurrent get_next_account calls should both complete quickly,
        even if one triggers initialization.

        What it does: Two accounts, one initialized and one not. Two concurrent
        get_next_account calls should both return without blocking each other.
        """
        account_a = Account(id="/a", owner="alice", auth_manager=MagicMock(),
                            models_cached_at=time.time())
        account_a.quota_info = QuotaInfo(
            owner="alice", account_id="/a", total_remaining=5000,
            is_exhausted=False, current_usage=0, usage_limit=2000,
            current_overages=0, overage_cap=10000,
        )
        account_b = Account(id="/b", owner="bob", auth_manager=None,
                            models_cached_at=0.0)
        account_b.quota_info = QuotaInfo(
            owner="bob", account_id="/b", total_remaining=3000,
            is_exhausted=False, current_usage=0, usage_limit=2000,
            current_overages=0, overage_cap=10000,
        )
        manager = _make_manager({"/a": account_a, "/b": account_b})

        async def slow_init(account_id):
            await asyncio.sleep(0.2)
            account_b.auth_manager = MagicMock()
            account_b.models_cached_at = time.time()
            return True

        with patch.object(manager, "_initialize_account", side_effect=slow_init):
            # Both calls should complete within 1 second (not 2x0.2s)
            task1 = asyncio.create_task(manager.get_next_account("claude-sonnet-4.5"))
            task2 = asyncio.create_task(manager.get_next_account("claude-sonnet-4.5"))

            results = await asyncio.wait_for(asyncio.gather(task1, task2), timeout=1.0)

        # At least one should have returned the initialized account immediately
        assert any(r is not None for r in results)


class TestInitializeAccountDoubleCheck:
    """Tests for the per-account lock and double-check pattern in _initialize_account."""

    @pytest.mark.asyncio
    async def test_already_initialized_returns_true_immediately(self):
        """
        If auth_manager is already set, _initialize_account should return True
        without doing any work (fast path).
        """
        account = Account(id="/a", owner="alice", auth_manager=MagicMock(),
                          models_cached_at=time.time())
        manager = _make_manager({"/a": account})

        result = await manager._initialize_account("/a")

        assert result is True

    @pytest.mark.asyncio
    async def test_init_failure_returns_false(self):
        """If initialization fails (no credentials config), should return False."""
        account = Account(id="/a", owner="alice", auth_manager=None,
                          models_cached_at=0.0)
        manager = _make_manager({"/a": account})

        with patch.object(manager, "_credentials_config", []):
            result = await manager._initialize_account("/a")

        assert result is False
        assert account.auth_manager is None

    @pytest.mark.asyncio
    async def test_concurrent_init_same_account_does_not_deadlock(self):
        """
        Two concurrent _initialize_account calls for the same account
        should complete without deadlock (per-account lock allows
        serialized access).
        """
        account = Account(id="/a", owner="alice", auth_manager=None,
                          models_cached_at=0.0)
        manager = _make_manager({"/a": account})

        with patch.object(manager, "_credentials_config", []):
            task1 = asyncio.create_task(manager._initialize_account("/a"))
            task2 = asyncio.create_task(manager._initialize_account("/a"))

            results = await asyncio.wait_for(asyncio.gather(task1, task2), timeout=5.0)

        # Both should return False (no creds config), but no deadlock
        assert all(r is False for r in results)


class TestRefreshAccountModelsDoubleCheck:
    """Tests for the per-account lock and double-check in _refresh_account_models."""

    @pytest.mark.asyncio
    async def test_fresh_cache_skips_refresh(self):
        """
        If the cache is still fresh (within TTL), _refresh_account_models
        should return immediately without doing any work.
        """
        account = Account(id="/a", owner="alice", auth_manager=MagicMock(),
                          models_cached_at=time.time())
        manager = _make_manager({"/a": account})

        original_cached_at = account.models_cached_at
        await manager._refresh_account_models("/a")

        # Timestamp should NOT have changed (no refresh occurred)
        assert account.models_cached_at == original_cached_at

    @pytest.mark.asyncio
    async def test_expired_cache_triggers_refresh(self):
        """
        If the cache is expired (past TTL), _refresh_account_models should
        attempt to refresh (call HTTP client).
        """
        old_time = time.time() - 50000  # Expired (TTL is 43200s = 12h)
        account = Account(id="/a", owner="alice", auth_manager=MagicMock(),
                          models_cached_at=old_time)
        account.model_cache = MagicMock()
        account.model_cache.update = AsyncMock()
        account.model_resolver = MagicMock()
        account.model_resolver.get_available_models.return_value = []
        manager = _make_manager({"/a": account})

        # Mock _is_runtime_endpoint to avoid real HTTP calls
        with patch("kiro.account_manager._is_runtime_endpoint", return_value=True):
            await manager._refresh_account_models("/a")

        # Timestamp should have been updated (refresh occurred)
        assert account.models_cached_at > old_time

    @pytest.mark.asyncio
    async def test_concurrent_refresh_same_account_does_not_deadlock(self):
        """
        Two concurrent _refresh_account_models calls for the same account
        should complete without deadlock (per-account lock).
        """
        old_time = time.time() - 50000
        account = Account(id="/a", owner="alice", auth_manager=MagicMock(),
                          models_cached_at=old_time)
        account.model_cache = MagicMock()
        account.model_cache.update = AsyncMock()
        account.model_resolver = MagicMock()
        account.model_resolver.get_available_models.return_value = []
        manager = _make_manager({"/a": account})

        with patch("kiro.account_manager._is_runtime_endpoint", return_value=True):
            task1 = asyncio.create_task(manager._refresh_account_models("/a"))
            task2 = asyncio.create_task(manager._refresh_account_models("/a"))

            await asyncio.wait_for(asyncio.gather(task1, task2), timeout=5.0)

        # Both should complete without error


class TestGetInitLock:
    """Tests for the _get_init_lock per-account lock factory."""

    def test_same_account_returns_same_lock(self):
        """Same account_id should return the same lock object."""
        manager = _make_manager({"/a": Account(id="/a", owner="alice")})
        lock1 = manager._get_init_lock("/a")
        lock2 = manager._get_init_lock("/a")
        assert lock1 is lock2

    def test_different_accounts_return_different_locks(self):
        """Different account_ids should return different lock objects."""
        manager = _make_manager({
            "/a": Account(id="/a", owner="alice"),
            "/b": Account(id="/b", owner="bob"),
        })
        lock1 = manager._get_init_lock("/a")
        lock2 = manager._get_init_lock("/b")
        assert lock1 is not lock2

    def test_lock_is_asyncio_lock(self):
        """The returned lock should be an asyncio.Lock."""
        manager = _make_manager({"/a": Account(id="/a", owner="alice")})
        lock = manager._get_init_lock("/a")
        assert isinstance(lock, asyncio.Lock)


class TestBuildOrderedCandidates:
    """Tests for the _build_ordered_candidates helper method."""

    def test_excludes_exhausted_accounts(self):
        """Exhausted accounts should not appear in candidates."""
        accounts = {
            "/a": _make_account("/a", "alice", total_remaining=5000),
            "/e": _make_account("/e", "exhausted", exhausted=True),
        }
        manager = _make_manager(accounts)
        candidates = manager._build_ordered_candidates()
        assert len(candidates) == 1
        assert candidates[0].owner == "alice"

    def test_excludes_specified_accounts(self):
        """Accounts in exclude_accounts should not appear in candidates."""
        accounts = {
            "/a": _make_account("/a", "alice", total_remaining=5000),
            "/b": _make_account("/b", "bob", total_remaining=3000),
        }
        manager = _make_manager(accounts)
        candidates = manager._build_ordered_candidates(exclude_accounts={"/a"})
        assert len(candidates) == 1
        assert candidates[0].owner == "bob"

    def test_priority_sorted_before_balanced(self):
        """Priority accounts should come before balanced accounts."""
        with patch("kiro.account_manager.ACCOUNT_PRIORITY", ["bob"]):
            accounts = {
                "/a": _make_account("/a", "alice", total_remaining=9000),
                "/b": _make_account("/b", "bob", total_remaining=100),
            }
            manager = _make_manager(accounts)
            candidates = manager._build_ordered_candidates()
            assert candidates[0].owner == "bob"
            assert candidates[1].owner == "alice"

    def test_balanced_sorted_by_quota_descending(self):
        """Balanced candidates should be sorted by remaining quota (most first)."""
        accounts = {
            "/low": _make_account("/low", "low", total_remaining=100),
            "/high": _make_account("/high", "high", total_remaining=9000),
            "/mid": _make_account("/mid", "mid", total_remaining=5000),
        }
        manager = _make_manager(accounts)
        candidates = manager._build_ordered_candidates()
        assert candidates[0].owner == "high"
        assert candidates[1].owner == "mid"
        assert candidates[2].owner == "low"

    def test_empty_when_all_exhausted(self):
        """Should return empty list when all accounts are exhausted."""
        accounts = {
            "/a": _make_account("/a", "alice", exhausted=True),
            "/b": _make_account("/b", "bob", exhausted=True),
        }
        manager = _make_manager(accounts)
        candidates = manager._build_ordered_candidates()
        assert len(candidates) == 0


class TestShouldSkipForCircuitBreaker:
    """Tests for the _should_skip_for_circuit_breaker helper method."""

    def test_no_failures_returns_false(self):
        """Account with no failures should not be skipped."""
        account = Account(id="/a", owner="alice", failures=0)
        manager = _make_manager({"/a": account})
        assert manager._should_skip_for_circuit_breaker(account) is False

    def test_recent_failure_skipped(self):
        """Account with recent failure should be skipped (high probability)."""
        account = Account(
            id="/a", owner="alice",
            failures=3,
            last_failure_time=time.time(),  # Just failed
        )
        manager = _make_manager({"/a": account})
        with patch("kiro.account_manager.ACCOUNT_PROBABILISTIC_RETRY_CHANCE", 0.0):
            assert manager._should_skip_for_circuit_breaker(account) is True

    def test_old_failure_not_skipped(self):
        """Account with old failure (past recovery timeout) should not be skipped."""
        account = Account(
            id="/a", owner="alice",
            failures=1,
            last_failure_time=time.time() - 99999,  # Very old
        )
        manager = _make_manager({"/a": account})
        assert manager._should_skip_for_circuit_breaker(account) is False

    def test_probabilistic_retry_can_pass(self):
        """Account in backoff can pass with probabilistic retry (chance=1.0)."""
        account = Account(
            id="/a", owner="alice",
            failures=3,
            last_failure_time=time.time(),  # Recent failure
        )
        manager = _make_manager({"/a": account})
        with patch("kiro.account_manager.ACCOUNT_PROBABILISTIC_RETRY_CHANCE", 1.0):
            assert manager._should_skip_for_circuit_breaker(account) is False

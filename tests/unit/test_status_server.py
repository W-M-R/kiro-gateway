# -*- coding: utf-8 -*-

"""
Tests for kiro/status_server.py - status dashboard and quota monitoring API.

Covers:
- HTML injection prevention (owner, subscription, last_error are all externally
  influenced and must be escaped before embedding in the dashboard HTML)
- Quota-unknown state rendering (failed queries must show UNKNOWN, not EXHAUSTED)
- API endpoints (/api/quota, /api/status) return correct structure and
  classify accounts consistently with the actual selection logic
- HTML structural validity (error rows use valid tr/td, not bare br after </tr>)
"""

import re
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from kiro.account_manager import Account, AccountStats
from kiro.quota import QuotaInfo
from kiro.status_server import create_status_app, _render_dashboard


def _make_quota(
    owner: str = "alice",
    account_id: str = "/a",
    subscription: str = "KIRO PRO",
    total_remaining: int = 5000,
    is_exhausted: bool = False,
    last_error: str = None,
    usage_limit: int = 2000,
    overage_cap: int = 10000,
) -> QuotaInfo:
    """Build a QuotaInfo with sensible defaults for dashboard tests."""
    return QuotaInfo(
        owner=owner,
        account_id=account_id,
        subscription=subscription,
        total_remaining=total_remaining,
        is_exhausted=is_exhausted,
        last_error=last_error,
        current_usage=100,
        usage_limit=usage_limit,
        current_overages=0,
        overage_cap=overage_cap,
        last_updated=time.time(),
    )


def _make_app(quotas: list, accounts_meta: dict = None) -> SimpleNamespace:
    """
    Build a minimal app-like object with the state _render_dashboard expects.

    Args:
        quotas: List of QuotaInfo objects.
        accounts_meta: Optional dict of {account_id: Account} to control
            _accounts directly (defaults to building from quotas).

    Returns:
        SimpleNamespace with app.state.account_manager and app.state.start_time.
    """
    accounts = {}
    if accounts_meta is not None:
        accounts = accounts_meta
    else:
        for q in quotas:
            a = Account(id=q.account_id, owner=q.owner)
            a.auth_manager = MagicMock()
            a.stats = AccountStats()
            a.quota_info = q
            accounts[a.id] = a

    am = SimpleNamespace(
        _accounts=accounts,
        get_all_quota_info=lambda: quotas,
    )
    return SimpleNamespace(
        state=SimpleNamespace(account_manager=am, start_time=time.time() - 100)
    )


class TestHtmlInjectionPrevention:
    """
    The dashboard embeds externally-influenced strings into HTML.

    owner: from directory name or credentials.json / KIRO_CLI_DB config
    subscription: from upstream GetUsageLimits JSON response
    last_error: from upstream HTTP response body (response.text[:200])

    All must be HTML-escaped to prevent injection, especially since the status
    server listens on 0.0.0.0 without authentication and last_error is
    attacker-controllable via a MITM proxy.
    """

    XSS_VECTORS = [
        '<script>alert(1)</script>',
        '<img src=x onerror=alert(1)>',
        '"></td><td>INJECTED</td>',
        "' onmouseover='alert(1)",
        '<svg/onload=alert(1)>',
    ]

    @pytest.mark.parametrize("payload", XSS_VECTORS)
    def test_owner_is_escaped(self, payload):
        """owner field must not produce executable HTML."""
        q = _make_quota(owner=payload)
        html = _render_dashboard(_make_app([q]))
        # No raw tag should survive — escaped form should be present instead
        assert "<script" not in html.lower() or "&lt;script" in html
        assert "<img" not in html.lower() or "&lt;img" in html
        assert "<svg" not in html.lower() or "&lt;svg" in html
        # The raw payload must not appear verbatim in a tag context
        assert payload not in html

    @pytest.mark.parametrize("payload", XSS_VECTORS)
    def test_subscription_is_escaped(self, payload):
        """subscription field must not break table structure or inject tags."""
        q = _make_quota(subscription=payload)
        html = _render_dashboard(_make_app([q]))
        assert payload not in html

    @pytest.mark.parametrize("payload", XSS_VECTORS)
    def test_last_error_is_escaped(self, payload):
        """last_error (from upstream response body) must be escaped."""
        q = _make_quota(last_error=f"HTTP 500: {payload}")
        html = _render_dashboard(_make_app([q]))
        assert payload not in html
        # The escaped version should be present
        from html import escape
        assert escape(payload) in html


class TestQuotaUnknownRendering:
    """
    A failed quota query reports is_exhausted=True with last_error set.

    The dashboard must show UNKNOWN (not EXHAUSTED) for such accounts, because
    they remain selectable — labeling them exhausted would mislead users into
    thinking the gateway is out of credits when it is still serving requests.
    """

    def test_failed_query_shows_unknown_not_exhausted(self):
        """Failed-query accounts display UNKNOWN, not EXHAUSTED."""
        q = _make_quota(last_error="HTTP 500: boom")
        html = _render_dashboard(_make_app([q]))
        assert "UNKNOWN" in html
        assert "EXHAUSTED" not in html

    def test_real_exhaustion_shows_exhausted(self):
        """Genuine exhaustion (no error) still shows EXHAUSTED."""
        q = _make_quota(total_remaining=0, is_exhausted=True)
        html = _render_dashboard(_make_app([q]))
        assert "EXHAUSTED" in html
        assert "UNKNOWN" not in html

    def test_healthy_account_shows_remaining_number(self):
        """A healthy account shows its remaining count, not a status label."""
        q = _make_quota(total_remaining=7777, is_exhausted=False)
        html = _render_dashboard(_make_app([q]))
        assert "7777" in html
        assert "EXHAUSTED" not in html
        assert "UNKNOWN" not in html

    def test_summary_counts_unknown_separately(self):
        """The summary must not count failed-query accounts as exhausted."""
        quotas = [
            _make_quota(owner="failed1", account_id="/f1", last_error="timeout"),
            _make_quota(owner="failed2", account_id="/f2", last_error="timeout"),
            _make_quota(owner="exhausted", account_id="/ex", total_remaining=0, is_exhausted=True),
            _make_quota(owner="healthy", account_id="/ok", total_remaining=5000),
        ]
        html = _render_dashboard(_make_app(quotas))

        # Extract stat-card values
        def extract(label):
            m = re.search(
                rf'{label}</div>\s*<div class="stat-value[^"]*">(\d+)', html
            )
            return int(m.group(1)) if m else None

        assert extract("Available") == 3, "2 failed + 1 healthy = 3 available"
        assert extract("Quota Unknown") == 2
        assert extract("Exhausted") == 1

    def test_failed_query_not_dimmed(self):
        """Failed-query rows must not get the 'exhausted' (dimmed) CSS class."""
        q = _make_quota(last_error="timeout")
        html = _render_dashboard(_make_app([q]))
        # The exhausted class is applied via class="exhausted" on <tr>
        assert 'class="exhausted"' not in html


class TestHtmlStructureValidity:
    """The dashboard must produce structurally valid HTML."""

    def test_error_row_uses_valid_tr_td(self):
        """Error display must be a proper <tr><td>, not a bare <br> after </tr>."""
        q = _make_quota(last_error="some error")
        html = _render_dashboard(_make_app([q]))
        # Must NOT have a bare <br> immediately after </tr>
        assert not re.search(r'</tr>\s*<br', html)
        # Must have the error in a colspan td
        assert 'colspan="6"' in html

    def test_tbody_tags_balanced(self):
        """All <tr> and <td> tags inside <tbody> must be balanced."""
        q = _make_quota(last_error="err")
        html = _render_dashboard(_make_app([q]))
        tb = re.search(r'<tbody>(.*?)</tbody>', html, re.S).group(1)
        assert tb.count("<tr") == tb.count("</tr>")
        assert tb.count("<td") == tb.count("</td>")

    def test_no_unescaped_angle_brackets_in_cells(self):
        """Cell content must not contain raw < or > from external data."""
        q = _make_quota(owner="a<b>c", subscription="d>e", last_error="f<g>h")
        html = _render_dashboard(_make_app([q]))
        tb = re.search(r'<tbody>(.*?)</tbody>', html, re.S).group(1)
        # Extract text content of cells — no raw <b>, <g> etc. should form tags
        assert "<b>" not in tb
        assert "<g>" not in tb


class TestApiQuotaEndpoint:
    """Tests for the /api/quota JSON endpoint."""

    def test_returns_all_quota_infos(self):
        """/api/quota should return all polled accounts."""
        quotas = [
            _make_quota(owner="alice", account_id="/a", total_remaining=100),
            _make_quota(owner="bob", account_id="/b", total_remaining=200),
        ]
        app = create_status_app(_make_app(quotas).state.account_manager)
        client = TestClient(app)
        resp = client.get("/api/quota")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_accounts"] == 2
        assert data["polled_accounts"] == 2
        owners = {a["owner"] for a in data["accounts"]}
        assert owners == {"alice", "bob"}

    def test_includes_is_quota_unknown_field(self):
        """/api/quota must expose is_quota_unknown so clients can distinguish."""
        quotas = [
            _make_quota(owner="ok", account_id="/ok", total_remaining=100),
            _make_quota(owner="fail", account_id="/fail", last_error="timeout"),
        ]
        app = create_status_app(_make_app(quotas).state.account_manager)
        client = TestClient(app)
        data = client.get("/api/quota").json()
        by_owner = {a["owner"]: a for a in data["accounts"]}
        assert by_owner["ok"]["is_quota_unknown"] is False
        assert by_owner["fail"]["is_quota_unknown"] is True


class TestApiStatusEndpoint:
    """Tests for the /api/status JSON endpoint."""

    def test_classifies_failed_query_as_available_not_exhausted(self):
        """/api/status must list failed-query accounts as available, not exhausted."""
        quotas = [
            _make_quota(owner="failed", account_id="/f", last_error="timeout"),
            _make_quota(owner="exhausted", account_id="/e", total_remaining=0, is_exhausted=True),
            _make_quota(owner="healthy", account_id="/h", total_remaining=5000),
        ]
        app = create_status_app(_make_app(quotas).state.account_manager)
        client = TestClient(app)
        data = client.get("/api/status").json()

        assert "failed" in data["available_accounts"]
        assert "healthy" in data["available_accounts"]
        assert "exhausted" in data["exhausted_accounts"]
        assert "failed" not in data["exhausted_accounts"]
        assert "failed" in data.get("quota_unknown_accounts", [])

    def test_returns_required_fields(self):
        """/api/status must include all documented summary fields."""
        app = create_status_app(_make_app([_make_quota()]).state.account_manager)
        client = TestClient(app)
        data = client.get("/api/status").json()
        for field in ("version", "uptime_seconds", "total_accounts",
                      "initialized_accounts", "polled_accounts",
                      "available_accounts", "exhausted_accounts",
                      "priority_order", "quota_poll_interval", "server_time"):
            assert field in data, f"missing field: {field}"

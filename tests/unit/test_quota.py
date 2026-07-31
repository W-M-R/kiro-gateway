# -*- coding: utf-8 -*-

"""
Tests for kiro/quota.py - GetUsageLimits API client and QuotaInfo parsing.

Covers:
- QuotaInfo dataclass creation and serialization
- parse_usage_response: all field combinations, edge cases, malformed input
- query_quota: success, HTTP errors, network exceptions
- _build_q_host: runtime vs legacy host conversion
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from kiro.quota import (
    QuotaInfo,
    _build_q_host,
    parse_usage_response,
    query_quota,
)


class TestQuotaInfoDataclass:
    """Tests for QuotaInfo dataclass creation and serialization."""

    def test_default_values_are_zero_and_exhausted(self):
        """Default QuotaInfo should be exhausted with zero remaining."""
        info = QuotaInfo(owner="test", account_id="/path")
        assert info.owner == "test"
        assert info.account_id == "/path"
        assert info.current_usage == 0
        assert info.usage_limit == 0
        assert info.total_remaining == 0
        assert info.is_exhausted is True
        assert info.last_error is None

    def test_to_dict_contains_all_fields(self):
        """to_dict should include all fields for JSON API responses."""
        info = QuotaInfo(
            owner="test",
            account_id="/path",
            subscription="KIRO PRO+",
            current_usage=100,
            usage_limit=2000,
            total_remaining=1900,
            is_exhausted=False,
        )
        d = info.to_dict()
        assert d["owner"] == "test"
        assert d["subscription"] == "KIRO PRO+"
        assert d["current_usage"] == 100
        assert d["usage_limit"] == 2000
        assert d["total_remaining"] == 1900
        assert d["is_exhausted"] is False
        assert "next_reset_iso" in d

    def test_to_dict_exposes_is_quota_unknown(self):
        """to_dict must expose is_quota_unknown so clients can tell failure apart."""
        ok = QuotaInfo(owner="t", account_id="/p", total_remaining=10, is_exhausted=False)
        failed = QuotaInfo(owner="t", account_id="/p", is_exhausted=True, last_error="HTTP 500")
        assert ok.to_dict()["is_quota_unknown"] is False
        assert failed.to_dict()["is_quota_unknown"] is True


class TestIsQuotaUnknown:
    """
    Tests for the is_quota_unknown property.

    A failed GetUsageLimits call returns is_exhausted=True as a placeholder.
    is_quota_unknown distinguishes that from genuine exhaustion so callers do
    not fail closed and reject traffic while accounts are actually healthy.
    """

    def test_false_when_query_succeeded(self):
        """Successful query (no last_error) means quota figures are trustworthy."""
        info = QuotaInfo(owner="t", account_id="/p", total_remaining=500, is_exhausted=False)
        assert info.is_quota_unknown is False

    def test_false_for_genuine_exhaustion(self):
        """A successful query reporting zero quota is real exhaustion, not unknown."""
        info = QuotaInfo(owner="t", account_id="/p", total_remaining=0, is_exhausted=True)
        assert info.is_quota_unknown is False

    def test_true_when_query_failed(self):
        """Any last_error means the figures are meaningless."""
        info = QuotaInfo(owner="t", account_id="/p", is_exhausted=True, last_error="HTTP 500")
        assert info.is_quota_unknown is True

    def test_default_quotainfo_is_not_unknown(self):
        """A bare QuotaInfo has no error, so quota counts as known (exhausted)."""
        info = QuotaInfo(owner="t", account_id="/p")
        assert info.is_quota_unknown is False
        assert info.is_exhausted is True

    def test_query_quota_failure_yields_unknown(self):
        """
        Integration with query_quota's failure contract: parse a failure-shaped
        QuotaInfo exactly as query_quota builds it.
        """
        failure = QuotaInfo(
            owner="alice",
            account_id="/a",
            is_exhausted=True,
            last_updated=123.0,
            last_error="ConnectTimeout: quota host unreachable",
        )
        assert failure.is_quota_unknown is True


class TestParseUsageResponse:
    """Tests for parse_usage_response function."""

    def test_normal_response_with_overage(self):
        """Normal response with both free and overage usage."""
        raw = {
            "usageBreakdownList": [{
                "currentUsage": 100,
                "usageLimit": 2000,
                "currentOverages": 50,
                "overageCap": 10000,
            }],
            "subscriptionInfo": {"subscriptionTitle": "KIRO PRO+"},
            "nextDateReset": 1785542400.0,
        }
        info = parse_usage_response(raw, "owner1", "/path", 1700000000.0)
        assert info.owner == "owner1"
        assert info.subscription == "KIRO PRO+"
        assert info.current_usage == 100
        assert info.usage_limit == 2000
        assert info.current_overages == 50
        assert info.overage_cap == 10000
        assert info.free_remaining == 1900
        assert info.overage_remaining == 9950
        assert info.total_remaining == 11850
        assert info.is_exhausted is False
        assert info.next_reset_epoch == 1785542400.0
        assert info.last_error is None

    def test_response_with_free_only_no_overage(self):
        """Response where only free tier is used, no overage."""
        raw = {
            "usageBreakdownList": [{
                "currentUsage": 500,
                "usageLimit": 2000,
                "currentOverages": 0,
                "overageCap": 10000,
            }],
            "subscriptionInfo": {"subscriptionTitle": "KIRO FREE"},
        }
        info = parse_usage_response(raw, "owner2", "/path", 1700000000.0)
        assert info.free_remaining == 1500
        assert info.overage_remaining == 10000
        assert info.total_remaining == 11500
        assert info.is_exhausted is False

    def test_response_free_exhausted_but_overage_available(self):
        """Free tier exhausted but overage still available."""
        raw = {
            "usageBreakdownList": [{
                "currentUsage": 2500,
                "usageLimit": 2000,
                "currentOverages": 100,
                "overageCap": 10000,
            }],
            "subscriptionInfo": {},
        }
        info = parse_usage_response(raw, "owner3", "/path", 1700000000.0)
        assert info.free_remaining == 0  # max(0, 2000-2500)
        assert info.overage_remaining == 9900
        assert info.total_remaining == 9900
        assert info.is_exhausted is False

    def test_response_fully_exhausted(self):
        """Both free and overage completely used up."""
        raw = {
            "usageBreakdownList": [{
                "currentUsage": 2000,
                "usageLimit": 2000,
                "currentOverages": 10000,
                "overageCap": 10000,
            }],
            "subscriptionInfo": {"subscriptionTitle": "KIRO PRO+"},
        }
        info = parse_usage_response(raw, "owner4", "/path", 1700000000.0)
        assert info.free_remaining == 0
        assert info.overage_remaining == 0
        assert info.total_remaining == 0
        assert info.is_exhausted is True

    def test_response_exhausted_beyond_limits(self):
        """Usage exceeds limits (negative remaining clamped to 0)."""
        raw = {
            "usageBreakdownList": [{
                "currentUsage": 3000,
                "usageLimit": 2000,
                "currentOverages": 15000,
                "overageCap": 10000,
            }],
            "subscriptionInfo": {},
        }
        info = parse_usage_response(raw, "owner5", "/path", 1700000000.0)
        assert info.free_remaining == 0
        assert info.overage_remaining == 0
        assert info.total_remaining == 0
        assert info.is_exhausted is True

    def test_empty_usage_breakdown_list(self):
        """Empty usageBreakdownList should result in exhausted account."""
        raw = {
            "usageBreakdownList": [],
            "subscriptionInfo": {"subscriptionTitle": "UNKNOWN"},
        }
        info = parse_usage_response(raw, "owner6", "/path", 1700000000.0)
        assert info.current_usage == 0
        assert info.usage_limit == 0
        assert info.total_remaining == 0
        assert info.is_exhausted is True

    def test_missing_usage_breakdown_list_key(self):
        """Missing usageBreakdownList key should not crash."""
        raw = {"subscriptionInfo": {}}
        info = parse_usage_response(raw, "owner7", "/path", 1700000000.0)
        assert info.is_exhausted is True
        assert info.total_remaining == 0

    def test_missing_fields_in_breakdown(self):
        """Missing individual fields in breakdown should default to 0."""
        raw = {
            "usageBreakdownList": [{}],
            "subscriptionInfo": {},
        }
        info = parse_usage_response(raw, "owner8", "/path", 1700000000.0)
        assert info.current_usage == 0
        assert info.usage_limit == 0
        assert info.current_overages == 0
        assert info.overage_cap == 0
        assert info.is_exhausted is True

    def test_missing_subscription_info(self):
        """Missing subscriptionInfo should result in empty subscription string."""
        raw = {
            "usageBreakdownList": [{
                "currentUsage": 0,
                "usageLimit": 2000,
                "currentOverages": 0,
                "overageCap": 10000,
            }],
        }
        info = parse_usage_response(raw, "owner9", "/path", 1700000000.0)
        assert info.subscription == ""

    def test_missing_subscription_title(self):
        """Missing subscriptionTitle field should default to empty string."""
        raw = {
            "usageBreakdownList": [{
                "currentUsage": 0,
                "usageLimit": 2000,
                "currentOverages": 0,
                "overageCap": 10000,
            }],
            "subscriptionInfo": {"otherField": "value"},
        }
        info = parse_usage_response(raw, "owner10", "/path", 1700000000.0)
        assert info.subscription == ""

    def test_last_updated_set_from_now_parameter(self):
        """last_updated should match the 'now' parameter passed in."""
        now = 1234567890.0
        raw = {"usageBreakdownList": [{}]}
        info = parse_usage_response(raw, "owner", "/path", now)
        assert info.last_updated == now


class TestBuildQHost:
    """Tests for _build_q_host helper."""

    def test_runtime_host_converted_to_legacy(self):
        """runtime.kiro.dev host should be converted to q.amazonaws.com."""
        result = _build_q_host("https://runtime.us-east-1.kiro.dev")
        assert result == "https://q.us-east-1.amazonaws.com"

    def test_runtime_host_eu_region(self):
        """runtime host with eu region should convert correctly."""
        result = _build_q_host("https://runtime.eu-central-1.kiro.dev")
        assert result == "https://q.eu-central-1.amazonaws.com"

    def test_legacy_host_unchanged(self):
        """Legacy q.amazonaws.com host should pass through unchanged."""
        result = _build_q_host("https://q.us-east-1.amazonaws.com")
        assert result == "https://q.us-east-1.amazonaws.com"

    def test_other_host_unchanged(self):
        """Non-runtime, non-legacy hosts should pass through unchanged."""
        result = _build_q_host("https://custom.example.com")
        assert result == "https://custom.example.com"


class TestQueryQuota:
    """Tests for query_quota async function."""

    @pytest.mark.asyncio
    async def test_successful_query(self):
        """Successful API response should return parsed QuotaInfo."""
        mock_auth = MagicMock()
        mock_auth.get_access_token = AsyncMock(return_value="fake-token")
        mock_auth.q_host = "https://runtime.us-east-1.kiro.dev"
        mock_auth.profile_arn = "arn:aws:codewhisperer:us-east-1:123:profile/ABC"
        mock_auth.fingerprint = "abc123"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "usageBreakdownList": [{
                "currentUsage": 100,
                "usageLimit": 2000,
                "currentOverages": 0,
                "overageCap": 10000,
            }],
            "subscriptionInfo": {"subscriptionTitle": "KIRO PRO+"},
            "nextDateReset": 1785542400.0,
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("kiro.quota.httpx.AsyncClient", return_value=mock_client):
            info = await query_quota(mock_auth, "test_owner", "/path/to/db")

        assert info.owner == "test_owner"
        assert info.subscription == "KIRO PRO+"
        assert info.current_usage == 100
        assert info.free_remaining == 1900
        assert info.is_exhausted is False
        assert info.last_error is None

        # Verify request was made to legacy Q host
        call_args = mock_client.post.call_args
        assert "q.us-east-1.amazonaws.com" in call_args[0][0]
        assert call_args[1]["json"]["profileArn"] == "arn:aws:codewhisperer:us-east-1:123:profile/ABC"

    @pytest.mark.asyncio
    async def test_http_error_returns_exhausted_with_error(self):
        """Non-200 HTTP response should return exhausted QuotaInfo with error."""
        mock_auth = MagicMock()
        mock_auth.get_access_token = AsyncMock(return_value="fake-token")
        mock_auth.q_host = "https://q.us-east-1.amazonaws.com"
        mock_auth.profile_arn = None
        mock_auth.fingerprint = "abc123"

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("kiro.quota.httpx.AsyncClient", return_value=mock_client):
            info = await query_quota(mock_auth, "err_owner", "/path")

        assert info.is_exhausted is True
        assert info.last_error is not None
        assert "403" in info.last_error

    @pytest.mark.asyncio
    async def test_network_exception_returns_exhausted_with_error(self):
        """Network exception should return exhausted QuotaInfo with error, not raise."""
        mock_auth = MagicMock()
        mock_auth.get_access_token = AsyncMock(return_value="fake-token")
        mock_auth.q_host = "https://q.us-east-1.amazonaws.com"
        mock_auth.profile_arn = None
        mock_auth.fingerprint = "abc123"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectTimeout("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("kiro.quota.httpx.AsyncClient", return_value=mock_client):
            info = await query_quota(mock_auth, "net_owner", "/path")

        assert info.is_exhausted is True
        assert info.last_error is not None
        assert "ConnectTimeout" in info.last_error

    @pytest.mark.asyncio
    async def test_token_refresh_failure_returns_exhausted(self):
        """If get_access_token fails, should return exhausted with error."""
        mock_auth = MagicMock()
        mock_auth.get_access_token = AsyncMock(side_effect=Exception("Token refresh failed"))
        mock_auth.q_host = "https://q.us-east-1.amazonaws.com"
        mock_auth.fingerprint = "abc123"

        info = await query_quota(mock_auth, "token_fail", "/path")

        assert info.is_exhausted is True
        assert info.last_error is not None
        assert "Token refresh failed" in info.last_error

    @pytest.mark.asyncio
    async def test_no_profile_arn_omits_field(self):
        """When profile_arn is None, request body should not include profileArn."""
        mock_auth = MagicMock()
        mock_auth.get_access_token = AsyncMock(return_value="fake-token")
        mock_auth.q_host = "https://q.us-east-1.amazonaws.com"
        mock_auth.profile_arn = None
        mock_auth.fingerprint = "abc123"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"usageBreakdownList": [{}]}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("kiro.quota.httpx.AsyncClient", return_value=mock_client):
            await query_quota(mock_auth, "no_arn", "/path")

        call_args = mock_client.post.call_args
        assert call_args[1]["json"] == {}  # Empty body, no profileArn

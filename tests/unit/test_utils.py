# -*- coding: utf-8 -*-

"""
Tests for kiro/utils.py - fingerprint, header construction, and ID generators.

Covers:
- get_machine_fingerprint: determinism and exception fallback
- _build_kiro_user_agent_headers: shared client-identification headers
- get_kiro_headers: GenerateAssistantResponse headers
- get_kiro_mcp_headers: MCP headers and how they intentionally differ
- generate_completion_id / generate_conversation_id / generate_tool_call_id
"""

import hashlib
import re
import uuid
from unittest.mock import MagicMock, patch

import pytest

from kiro.utils import (
    _build_kiro_user_agent_headers,
    generate_completion_id,
    generate_conversation_id,
    generate_tool_call_id,
    get_kiro_headers,
    get_kiro_mcp_headers,
    get_machine_fingerprint,
)


def _auth(fingerprint: str = "fp_abc123") -> MagicMock:
    """Build a minimal auth manager stub exposing only .fingerprint."""
    am = MagicMock()
    am.fingerprint = fingerprint
    return am


class TestGetMachineFingerprint:
    """Tests for get_machine_fingerprint."""

    def test_returns_sha256_hex(self):
        """
        What it does: Verifies the fingerprint is a SHA256 hex digest.
        Purpose: The value is embedded in User-Agent, so its shape must be stable.
        """
        fp = get_machine_fingerprint()
        assert isinstance(fp, str)
        assert len(fp) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", fp)

    def test_is_deterministic(self):
        """
        What it does: Verifies repeated calls return the same value.
        Purpose: The fingerprint identifies an installation; it must not drift
        between requests.
        """
        assert get_machine_fingerprint() == get_machine_fingerprint()

    def test_matches_expected_hash_of_hostname_and_username(self):
        """
        What it does: Verifies the exact hashed input format.
        Purpose: Pin the documented "{hostname}-{username}-kiro-gateway" contract.
        """
        with patch("socket.gethostname", return_value="host1"), \
             patch("getpass.getuser", return_value="user1"):
            expected = hashlib.sha256(b"host1-user1-kiro-gateway").hexdigest()
            assert get_machine_fingerprint() == expected

    def test_falls_back_to_default_hash_on_exception(self):
        """
        What it does: Verifies the fallback when hostname lookup fails.
        Purpose: Fingerprinting must never break request building; a container
        with a broken hostname should still produce a usable value.
        """
        with patch("socket.gethostname", side_effect=OSError("no hostname")):
            expected = hashlib.sha256(b"default-kiro-gateway").hexdigest()
            assert get_machine_fingerprint() == expected

    def test_falls_back_when_getuser_fails(self):
        """
        What it does: Verifies the fallback when username lookup fails.
        Purpose: getpass.getuser() raises when no passwd entry exists (common in
        minimal containers).
        """
        with patch("getpass.getuser", side_effect=KeyError("no user")):
            expected = hashlib.sha256(b"default-kiro-gateway").hexdigest()
            assert get_machine_fingerprint() == expected


class TestBuildKiroUserAgentHeaders:
    """Tests for _build_kiro_user_agent_headers."""

    def test_returns_both_user_agent_headers(self):
        """
        What it does: Verifies both UA header keys are produced.
        Purpose: Kiro rejects requests missing either header.
        """
        h = _build_kiro_user_agent_headers("fp_x")
        assert set(h) == {"User-Agent", "x-amz-user-agent"}

    def test_embeds_fingerprint_in_both_headers(self):
        """
        What it does: Verifies the fingerprint is interpolated into both values.
        Purpose: The fingerprint is how a gateway install identifies itself.
        """
        h = _build_kiro_user_agent_headers("fp_unique_999")
        assert "fp_unique_999" in h["User-Agent"]
        assert "fp_unique_999" in h["x-amz-user-agent"]

    def test_emulates_kiro_ide_client(self):
        """
        What it does: Verifies the UA still claims to be KiroIDE.
        Purpose: The gateway must emulate the IDE client for API compatibility;
        changing this silently breaks access.
        """
        h = _build_kiro_user_agent_headers("fp")
        assert "KiroIDE-0.7.45-fp" in h["User-Agent"]
        assert "KiroIDE-0.7.45-fp" in h["x-amz-user-agent"]
        assert "aws-sdk-js" in h["User-Agent"]

    def test_handles_empty_fingerprint(self):
        """
        What it does: Verifies no crash on an empty fingerprint.
        Purpose: Defensive: an auth manager with an unset fingerprint must not
        raise during header construction.
        """
        h = _build_kiro_user_agent_headers("")
        assert h["User-Agent"].endswith("KiroIDE-0.7.45-")


class TestGetKiroHeaders:
    """Tests for get_kiro_headers (GenerateAssistantResponse)."""

    def test_contains_all_required_headers(self):
        """
        What it does: Verifies the full required header set.
        Purpose: A missing header yields the opaque "Improperly formed request".
        """
        h = get_kiro_headers(_auth(), "tok123")
        for key in (
            "Authorization",
            "Content-Type",
            "x-amz-target",
            "x-amzn-codewhisperer-optout",
            "x-amzn-kiro-agent-mode",
            "amz-sdk-invocation-id",
            "amz-sdk-request",
            "User-Agent",
            "x-amz-user-agent",
        ):
            assert key in h, f"missing header: {key}"

    def test_authorization_uses_bearer_token(self):
        """
        What it does: Verifies Bearer prefix and token passthrough.
        Purpose: Auth failures are hard to diagnose; pin the exact format.
        """
        assert get_kiro_headers(_auth(), "tok123")["Authorization"] == "Bearer tok123"

    def test_uses_amz_json_content_type_and_target(self):
        """
        What it does: Verifies the AWS JSON protocol headers.
        Purpose: GenerateAssistantResponse requires x-amz-json-1.0 plus the
        streaming service target, unlike the MCP endpoint.
        """
        h = get_kiro_headers(_auth(), "t")
        assert h["Content-Type"] == "application/x-amz-json-1.0"
        assert h["x-amz-target"] == (
            "AmazonCodeWhispererStreamingService.GenerateAssistantResponse"
        )

    def test_opts_out_of_telemetry(self):
        """
        What it does: Verifies opt-out is enabled for inference requests.
        Purpose: User content must not be retained for service improvement.
        """
        assert get_kiro_headers(_auth(), "t")["x-amzn-codewhisperer-optout"] == "true"

    def test_invocation_id_is_unique_per_call(self):
        """
        What it does: Verifies amz-sdk-invocation-id differs across calls.
        Purpose: A reused invocation id can cause upstream request dedup.
        """
        a = get_kiro_headers(_auth(), "t")["amz-sdk-invocation-id"]
        b = get_kiro_headers(_auth(), "t")["amz-sdk-invocation-id"]
        assert a != b
        uuid.UUID(a)  # must parse as a UUID

    def test_uses_fingerprint_from_auth_manager(self):
        """
        What it does: Verifies the fingerprint comes from the auth manager.
        Purpose: Per-account managers may carry different fingerprints.
        """
        h = get_kiro_headers(_auth("fp_from_am"), "t")
        assert "fp_from_am" in h["User-Agent"]


class TestGetKiroMcpHeaders:
    """
    Tests for get_kiro_mcp_headers (web_search / MCP endpoint).

    MCP intentionally differs from GenerateAssistantResponse. Missing or wrong
    headers here cause 403 "User is not authorized to make this call".
    """

    def test_contains_all_required_headers(self):
        """
        What it does: Verifies the required MCP header set.
        Purpose: Guard the 403-causing regression documented in the source.
        """
        h = get_kiro_mcp_headers(_auth(), "tok")
        for key in (
            "Authorization",
            "Content-Type",
            "x-amzn-codewhisperer-optout",
            "x-amzn-kiro-agent-mode",
            "amz-sdk-invocation-id",
            "amz-sdk-request",
            "User-Agent",
            "x-amz-user-agent",
        ):
            assert key in h, f"missing header: {key}"

    def test_uses_plain_json_content_type(self):
        """
        What it does: Verifies MCP uses application/json.
        Purpose: MCP is JSON-RPC, not the AWS JSON protocol.
        """
        assert get_kiro_mcp_headers(_auth(), "t")["Content-Type"] == "application/json"

    def test_omits_x_amz_target(self):
        """
        What it does: Verifies x-amz-target is absent.
        Purpose: MCP is a separate endpoint; sending a target header is wrong.
        """
        assert "x-amz-target" not in get_kiro_mcp_headers(_auth(), "t")

    def test_optout_is_false_for_mcp(self):
        """
        What it does: Verifies MCP opt-in semantics (optout=false).
        Purpose: This differs from inference requests (true) and is load-bearing.
        """
        assert get_kiro_mcp_headers(_auth(), "t")["x-amzn-codewhisperer-optout"] == "false"

    def test_differs_from_inference_headers_only_as_documented(self):
        """
        What it does: Compares MCP and inference header sets.
        Purpose: Lock in the documented differences so neither drifts silently.
        """
        inference = get_kiro_headers(_auth(), "t")
        mcp = get_kiro_mcp_headers(_auth(), "t")

        assert set(inference) - set(mcp) == {"x-amz-target"}
        assert set(mcp) - set(inference) == set()
        assert inference["Content-Type"] != mcp["Content-Type"]
        assert (
            inference["x-amzn-codewhisperer-optout"]
            != mcp["x-amzn-codewhisperer-optout"]
        )
        # Client identification must stay consistent across both endpoints
        assert inference["User-Agent"] == mcp["User-Agent"]
        assert inference["x-amz-user-agent"] == mcp["x-amz-user-agent"]
        assert inference["x-amzn-kiro-agent-mode"] == mcp["x-amzn-kiro-agent-mode"]


class TestGenerateCompletionId:
    """Tests for generate_completion_id."""

    def test_format_is_chatcmpl_prefixed_hex(self):
        """
        What it does: Verifies the OpenAI-compatible id format.
        Purpose: Clients parse this prefix.
        """
        cid = generate_completion_id()
        assert cid.startswith("chatcmpl-")
        assert re.fullmatch(r"chatcmpl-[0-9a-f]{32}", cid)

    def test_is_unique(self):
        """
        What it does: Verifies ids do not repeat.
        Purpose: Duplicate completion ids confuse client-side bookkeeping.
        """
        assert len({generate_completion_id() for _ in range(100)}) == 100


class TestGenerateConversationId:
    """Tests for generate_conversation_id."""

    def test_returns_uuid_string(self):
        """
        What it does: Verifies the value is a parseable UUID string.
        Purpose: Sent as conversationState.conversationId in the Kiro payload.
        """
        cid = generate_conversation_id()
        assert len(cid) == 36
        uuid.UUID(cid)

    def test_is_unique_per_call(self):
        """
        What it does: Verifies a fresh id per call.
        Purpose: Kiro does not correlate on this id; randomness is intended.
        """
        assert len({generate_conversation_id() for _ in range(100)}) == 100

    def test_accepts_no_arguments(self):
        """
        What it does: Verifies the function takes zero arguments.
        Purpose: Regression guard. It previously accepted a `messages` argument
        and hashed the conversation to produce a supposedly "stable" id. That
        branch was unreachable (all call sites pass nothing) and its stability
        claim was false, since the hash included the last message and therefore
        changed every turn. Re-adding the parameter should be a deliberate,
        reviewed decision, not an accident.
        """
        with pytest.raises(TypeError):
            generate_conversation_id([{"role": "user", "content": "hi"}])


class TestGenerateToolCallId:
    """Tests for generate_tool_call_id."""

    def test_format_is_call_prefixed_short_hex(self):
        """
        What it does: Verifies the documented "call_{8 hex}" format.
        Purpose: Tool-call ids must round-trip through OpenAI-format clients.
        """
        tid = generate_tool_call_id()
        assert re.fullmatch(r"call_[0-9a-f]{8}", tid)

    def test_is_unique(self):
        """
        What it does: Verifies ids are distinct across many calls.
        Purpose: Colliding ids would mismatch tool results to calls. Only 8 hex
        chars (32 bits), so allow a tiny collision margin.
        """
        ids = {generate_tool_call_id() for _ in range(500)}
        assert len(ids) >= 499

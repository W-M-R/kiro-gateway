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
Utility functions for Kiro Gateway.

Contains functions for fingerprint generation, header formatting,
and other common utilities.
"""

import hashlib
import uuid
from typing import TYPE_CHECKING, Dict

from loguru import logger

if TYPE_CHECKING:
    from kiro.auth import KiroAuthManager


def get_machine_fingerprint() -> str:
    """
    Generates a unique machine fingerprint based on hostname and username.
    
    Used for User-Agent formation to identify a specific gateway installation.
    
    Returns:
        SHA256 hash of the string "{hostname}-{username}-kiro-gateway"
    """
    try:
        import socket
        import getpass
        
        hostname = socket.gethostname()
        username = getpass.getuser()
        unique_string = f"{hostname}-{username}-kiro-gateway"
        
        return hashlib.sha256(unique_string.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Failed to get machine fingerprint: {e}")
        return hashlib.sha256(b"default-kiro-gateway").hexdigest()


# User-Agent strings emulating the Kiro IDE client for API compatibility.
# Shared by all Kiro API requests (GenerateAssistantResponse and MCP).
_KIRO_USER_AGENT_TEMPLATE = (
    "aws-sdk-js/1.0.27 ua/2.1 os/win32#10.0.19044 lang/js md/nodejs#22.21.1 "
    "api/codewhispererstreaming#1.0.27 m/E KiroIDE-0.7.45-{fingerprint}"
)
_KIRO_X_AMZ_USER_AGENT_TEMPLATE = "aws-sdk-js/1.0.27 KiroIDE-0.7.45-{fingerprint}"


def _build_kiro_user_agent_headers(fingerprint: str) -> Dict[str, str]:
    """
    Build User-Agent and x-amz-user-agent headers for Kiro API requests.

    Centralises client-identification header construction so that
    GenerateAssistantResponse and MCP requests stay consistent.

    Args:
        fingerprint: Machine fingerprint for client identification.

    Returns:
        Dict with ``User-Agent`` and ``x-amz-user-agent`` keys.
    """
    return {
        "User-Agent": _KIRO_USER_AGENT_TEMPLATE.format(fingerprint=fingerprint),
        "x-amz-user-agent": _KIRO_X_AMZ_USER_AGENT_TEMPLATE.format(fingerprint=fingerprint),
    }


def get_kiro_headers(auth_manager: "KiroAuthManager", token: str) -> dict:
    """
    Builds headers for Kiro GenerateAssistantResponse API requests.

    Includes all necessary headers for authentication and identification:
    - Authorization with Bearer token
    - User-Agent with fingerprint
    - AWS CodeWhisperer specific headers (x-amz-target, opt-out, agent mode)

    Args:
        auth_manager: Authentication manager for obtaining fingerprint
        token: Access token for authorization

    Returns:
        Dictionary with headers for HTTP request
    """
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/x-amz-json-1.0",
        "x-amz-target": "AmazonCodeWhispererStreamingService.GenerateAssistantResponse",
        "x-amzn-codewhisperer-optout": "true",
        "x-amzn-kiro-agent-mode": "vibe",
        "amz-sdk-invocation-id": str(uuid.uuid4()),
        "amz-sdk-request": "attempt=1; max=3",
    }
    headers.update(_build_kiro_user_agent_headers(auth_manager.fingerprint))
    return headers


def get_kiro_mcp_headers(auth_manager: "KiroAuthManager", token: str) -> dict:
    """
    Builds headers for Kiro MCP API requests (e.g. web_search tool).

    MCP requests differ from GenerateAssistantResponse requests:
    - Content-Type: ``application/json`` (not ``application/x-amz-json-1.0``)
    - No ``x-amz-target`` header (MCP is a separate JSON-RPC endpoint)
    - ``x-amzn-codewhisperer-optout: false`` (MCP opt-in semantics)

    Otherwise shares User-Agent, x-amz-user-agent, x-amzn-kiro-agent-mode,
    and amz-sdk-* headers to authenticate and identify the client. Missing
    these headers causes the MCP endpoint to reject requests with 403
    "User is not authorized to make this call".

    Args:
        auth_manager: Authentication manager for obtaining fingerprint
        token: Access token for authorization

    Returns:
        Dictionary with headers for MCP HTTP request
    """
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-amzn-codewhisperer-optout": "false",
        "x-amzn-kiro-agent-mode": "vibe",
        "amz-sdk-invocation-id": str(uuid.uuid4()),
        "amz-sdk-request": "attempt=1; max=3",
    }
    headers.update(_build_kiro_user_agent_headers(auth_manager.fingerprint))
    return headers


def generate_completion_id() -> str:
    """
    Generates a unique ID for chat completion.
    
    Returns:
        ID in format "chatcmpl-{uuid_hex}"
    """
    return f"chatcmpl-{uuid.uuid4().hex}"


def generate_conversation_id() -> str:
    """
    Generates a random conversation ID for the Kiro API payload.

    The ID is sent as ``conversationState.conversationId``. Kiro does not use it
    to correlate requests, so a fresh random UUID per request is correct and no
    cross-request stability is required.

    Note for future maintainers: this function previously accepted a ``messages``
    argument and hashed the conversation to derive a "stable" ID. That branch was
    never reachable (every call site invokes it with no arguments) and its
    stability guarantee was false anyway, because the hash included the LAST
    message and therefore changed on every turn. Do not reintroduce it without a
    concrete requirement and a correct stability scheme.

    Returns:
        Random conversation ID as a UUID string.

    Example:
        >>> conv_id = generate_conversation_id()
        >>> len(conv_id)
        36
    """
    return str(uuid.uuid4())


def generate_tool_call_id() -> str:
    """
    Generates a unique ID for tool call.
    
    Returns:
        ID in format "call_{uuid_hex[:8]}"
    """
    return f"call_{uuid.uuid4().hex[:8]}"
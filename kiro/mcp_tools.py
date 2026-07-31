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
MCP Tools Support (WebSearch via Kiro MCP API).

Handles server-side tools that execute on Kiro infrastructure via MCP API.
This module provides:
- MCP API calls for web_search
- SSE response emulation in Anthropic/OpenAI formats
- Path A: Native Anthropic server-side tools (early return)
- Path B: MCP tool emulation (streaming interception)
"""

import json
import time
import uuid
import random
import string
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

import httpx
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from kiro.config import get_httpx_verify_config
from kiro.tokenizer import count_message_tokens, count_tokens
from kiro.utils import get_kiro_mcp_headers

# Import debug_logger
try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None

# Module-level import of parse_kiro_stream so it can be patched in tests. There
# is no circular import: streaming_core does not import mcp_tools.
from kiro.streaming_core import KiroEvent, parse_kiro_stream  # noqa: E402


# ==================================================================================================
# WebSearch Continuation Context
# ==================================================================================================

@dataclass
class WebSearchContinuation:
    """
    Context needed to feed web_search results back to the model (tool round-trip).

    When the model calls web_search mid-stream, the gateway executes the search
    and must issue a follow-up Kiro request so the model can consume the results
    and produce a final answer. This object carries everything the streaming
    layer needs to build and dispatch that follow-up request.

    Attributes:
        base_payload: The Kiro payload originally sent for this request. Used as
            the template for continuation payloads (see
            build_web_search_continuation_payload). Never mutated.
        send_request: Async callback that POSTs a Kiro payload and returns the
            streaming httpx.Response. Provided by the route layer so the
            continuation reuses the same HTTP client, auth, and retry logic.
            Must return an httpx.Response; a non-200 response triggers the
            raw-summary fallback rather than an exception.
        max_iterations: Maximum number of consecutive web_search rounds allowed
            for this request (from WEB_SEARCH_MAX_ITERATIONS). Once reached, the
            raw search summary is emitted to the client instead of continuing.
    """
    base_payload: Dict[str, Any]
    send_request: Callable[[Dict[str, Any]], Awaitable[httpx.Response]]
    max_iterations: int


def _extract_web_search_query(tool: Dict[str, Any]) -> Optional[str]:
    """
    Extract the search query from a web_search tool_use event.

    Handles both OpenAI-style (function.arguments) and Anthropic-style (input)
    tool structures, with arguments as either a JSON string or a dict.

    Args:
        tool: Tool-use dict from a KiroEvent (event.tool_use).

    Returns:
        The query string, or None if absent/unparseable.
    """
    tool_input = (tool.get("function", {}) or {}).get("arguments", None)
    if tool_input is None:
        tool_input = tool.get("input", {})
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except json.JSONDecodeError:
            return None
    if not isinstance(tool_input, dict):
        return None
    query = tool_input.get("query", "")
    return query.strip() if isinstance(query, str) and query.strip() else None


async def web_search_event_source(
    initial_response: "httpx.Response",
    continuation: Optional[WebSearchContinuation],
    auth_manager,
    first_token_timeout: float,
):
    """
    Yield Kiro events across one or more responses, resolving web_search calls.

    This async generator wraps ``parse_kiro_stream`` and transparently turns the
    "model requests web_search → gateway executes search → model consumes results"
    round-trip into a continuous event stream. Consumers (OpenAI/Anthropic
    streaming formatters) iterate it exactly like ``parse_kiro_stream`` and stay
    largely unchanged.

    Behaviour:
    - Non-web_search events (content, thinking, usage, context_usage, normal
      tool_use) pass through unchanged.
    - When a ``web_search`` tool_use is seen and ``continuation`` is provided:
        1. The search is executed via the MCP API.
        2. A synthetic ``web_search`` event is emitted carrying the query,
           tool_use_id, and results so the formatter can render transparency
           blocks (Anthropic) or ignore them (OpenAI).
        3. A follow-up Kiro request is dispatched with the tool result appended
           to history, and iteration continues over the new response — the
           model's follow-up text becomes the visible answer.
    - Fallbacks that emit the raw summary as ``content`` events (legacy
      behaviour) instead of continuing:
        * ``continuation`` is None (feature disabled).
        * The iteration limit (``max_iterations``) is reached.
        * The MCP call fails or returns no results (in which case the original
          tool_use passes through so the client still sees the attempt).
        * The follow-up Kiro request returns a non-200 response.

    Response lifecycle: the initial response is owned/closed by the caller; every
    continuation response opened here is closed here (on transition or via the
    finally block), preventing connection leaks.

    Args:
        initial_response: The already-validated (HTTP 200) first Kiro response.
        continuation: Continuation context, or None to disable continuation.
        auth_manager: Auth manager used for the MCP web_search call.
        first_token_timeout: First-token timeout forwarded to parse_kiro_stream.

    Yields:
        KiroEvent objects (including synthetic ``web_search`` events).
    """
    # Imported at module level (see above) so it can be patched in tests.
    current_response = initial_response
    owns_current = False  # initial response is owned by the caller
    iteration = 0

    # Payload used to build the NEXT continuation request. It starts as the
    # original request and accumulates every completed search round, so the model
    # keeps seeing all prior results.
    #
    # Rebuilding from ``continuation.base_payload`` on every round would drop the
    # earlier rounds' toolResults: the model would request a search, receive a
    # follow-up turn that no longer contains the result it just asked for, and
    # search again — burning through max_iterations and ending on the raw-summary
    # fallback instead of a synthesized answer.
    current_payload = continuation.base_payload if continuation is not None else None

    try:
        while True:
            assistant_text = ""
            pending: Optional[Tuple[str, str, Dict]] = None  # (query, tool_use_id, results)

            async for event in parse_kiro_stream(current_response, first_token_timeout):
                if event.type == "content":
                    assistant_text += event.content or ""
                    yield event
                    continue

                if event.type == "tool_use" and event.tool_use:
                    tool = event.tool_use
                    tool_name = (tool.get("function", {}) or {}).get("name", "") or tool.get("name", "")

                    if tool_name != "web_search" or continuation is None:
                        # Normal tool call (or continuation disabled) — pass through.
                        yield event
                        continue

                    query = _extract_web_search_query(tool)
                    if not query:
                        logger.warning("web_search called without query, skipping MCP call")
                        continue

                    logger.info("Intercepted web_search tool call (continuation flow)")
                    mcp_tool_use_id, results = await call_kiro_mcp_api(query, auth_manager)

                    if results is None:
                        # MCP failed — surface the original tool call so the
                        # client still sees the attempt (legacy passthrough).
                        logger.error("MCP API call failed for web_search, passing tool_use through")
                        yield event
                        continue

                    iteration += 1

                    # Emit transparency event (formatters decide how to render).
                    yield KiroEvent(
                        type="web_search",
                        web_search={
                            "query": query,
                            "tool_use_id": mcp_tool_use_id,
                            "results": results,
                        },
                    )

                    if iteration >= continuation.max_iterations:
                        # Iteration limit reached — emit raw summary and stop.
                        logger.warning(
                            f"web_search iteration limit ({continuation.max_iterations}) reached, "
                            f"emitting raw summary instead of continuing"
                        )
                        summary = generate_search_summary(query, results)
                        yield KiroEvent(type="content", content=summary)
                        return

                    pending = (query, mcp_tool_use_id, results)
                    break  # exit inner loop to dispatch continuation request

                # thinking / usage / context_usage / anything else
                yield event
            else:
                # Inner loop exhausted without a continuation — we're done.
                return

            # A web_search continuation was requested.
            if pending is None:
                return

            query, tool_use_id, results = pending
            try:
                # Build on top of the accumulated payload (not the original one)
                # so every previous round's toolResults stay in history.
                new_payload = build_web_search_continuation_payload(
                    current_payload, query, tool_use_id, results, assistant_text
                )
                new_response = await continuation.send_request(new_payload)
            except Exception as e:
                logger.error(
                    f"Failed to dispatch web_search continuation request: {e}, "
                    f"emitting raw summary as fallback",
                    exc_info=True,
                )
                yield KiroEvent(type="content", content=generate_search_summary(query, results))
                return

            # Non-200 from the continuation request — fall back to raw summary.
            if new_response.status_code != 200:
                logger.error(
                    f"web_search continuation request returned {new_response.status_code}, "
                    f"emitting raw summary as fallback"
                )
                try:
                    await new_response.aclose()
                except Exception:
                    pass
                yield KiroEvent(type="content", content=generate_search_summary(query, results))
                return

            # Transition to the new response, closing the previous continuation
            # response (never the caller-owned initial response).
            if owns_current and current_response is not None:
                try:
                    await current_response.aclose()
                except Exception:
                    pass
            current_response = new_response
            owns_current = True
            # Accumulate this round so the next continuation keeps its results.
            current_payload = new_payload
            # loop continues, iterating over the continuation response
    finally:
        # Close the final continuation response if we own it. The initial
        # response is closed by the caller's own finally block.
        if owns_current and current_response is not None:
            try:
                await current_response.aclose()
            except Exception:
                pass


# ==================================================================================================
# ID Generation
# ==================================================================================================

def generate_random_id(length: int) -> str:
    """
    Generate random alphanumeric string.
    
    Args:
        length: Length of string to generate
    
    Returns:
        Random string of specified length
    
    Example:
        >>> generate_random_id(22)
        'aBcD1234567890XyZ12345'
    """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


# ==================================================================================================
# MCP API Functions
# ==================================================================================================

async def call_kiro_mcp_api(
    query: str,
    auth_manager
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Call Kiro MCP API for web_search.
    
    URL: {auth_manager.q_host}/mcp
    Headers: Authorization, x-amzn-codewhisperer-optout, Content-Type
    Timeout: 60 seconds
    
    Args:
        query: Search query
        auth_manager: KiroAuthManager instance
    
    Returns:
        Tuple of (tool_use_id, results_dict) or (None, None) on error
    
    MCP Request Format:
        {
            "id": "web_search_tooluse_{22random}_{timestamp}_{8random}",
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "web_search",
                "arguments": {"query": "..."}
            }
        }
    
    MCP Response Format:
        {
            "id": "web_search_tooluse_...",
            "jsonrpc": "2.0",
            "result": {
                "content": [{
                    "type": "text",
                    "text": "{\"results\":[...],\"totalResults\":10,\"query\":\"...\"}"
                }],
                "isError": false
            }
        }
    
    CRITICAL: result.content[0].text is a JSON STRING, not a dict!
    """
    # Generate IDs
    random_22 = generate_random_id(22)
    timestamp = int(time.time() * 1000)
    random_8 = generate_random_id(8)
    request_id = f"web_search_tooluse_{random_22}_{timestamp}_{random_8}"
    tool_use_id = f"srvtoolu_{uuid.uuid4().hex[:32]}"
    
    # Build MCP request
    # profileArn is REQUIRED at the top level of the MCP request body — without
    # it the MCP endpoint rejects the call with 400 "profileArn is required for
    # this request." (unlike GenerateAssistantResponse where it sits in the
    # payload, MCP is JSON-RPC and expects profileArn alongside the RPC fields).
    mcp_request = {
        "id": request_id,
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "web_search",
            "arguments": {"query": query}
        },
    }
    if auth_manager.profile_arn:
        mcp_request["profileArn"] = auth_manager.profile_arn
    
    # Log MCP request
    try:
        mcp_request_json = json.dumps(mcp_request, ensure_ascii=False, indent=2).encode('utf-8')
        if debug_logger:
            debug_logger.log_raw_chunk(b"[MCP REQUEST]\n" + mcp_request_json)
    except Exception as e:
        logger.warning(f"Failed to log MCP request: {e}")
    
    try:
        token = await auth_manager.get_access_token()
        
        # Headers: MCP requires the full Kiro client header set (User-Agent,
        # x-amz-user-agent, x-amzn-kiro-agent-mode, amz-sdk-*) to authenticate;
        # with only Authorization the endpoint returns 403. Content-Type is
        # application/json (not the x-amz-json-1.0 used by streaming), and there
        # is no x-amz-target since MCP is a separate JSON-RPC endpoint.
        headers = get_kiro_mcp_headers(auth_manager, token)
        
        mcp_url = f"{auth_manager.q_host}/mcp"
        logger.debug(f"Calling MCP API: {mcp_url}")
        
        async with httpx.AsyncClient(timeout=60.0, verify=get_httpx_verify_config()) as client:
            response = await client.post(mcp_url, json=mcp_request, headers=headers)
            
            if response.status_code != 200:
                logger.error(
                    f"MCP API error: {response.status_code}, body={response.text[:300]}"
                )
                return None, None
            
            mcp_response = response.json()
            
            # Log MCP response
            try:
                mcp_response_json = json.dumps(mcp_response, ensure_ascii=False, indent=2).encode('utf-8')
                if debug_logger:
                    debug_logger.log_raw_chunk(b"[MCP RESPONSE]\n" + mcp_response_json)
            except Exception as e:
                logger.warning(f"Failed to log MCP response: {e}")
            
            # DEBUG: Log full MCP response to see what we actually got
            # logger.debug(f"MCP API full response: {json.dumps(mcp_response, ensure_ascii=False)}")
            
            if "error" in mcp_response and mcp_response["error"] is not None:
                logger.error(f"MCP API returned error: {mcp_response['error']}")
                return None, None
            
            # Parse results: result.content[0].text is JSON STRING (CRITICAL!)
            result_text = mcp_response.get("result", {}).get("content", [{}])[0].get("text", "{}")
            results = json.loads(result_text)  # Parse JSON string to dict
            
            logger.debug(f"MCP API returned {results.get('totalResults', 0)} results")
            return tool_use_id, results
            
    except httpx.TimeoutException as e:
        logger.error(f"MCP API timeout: {e}")
        return None, None
    except httpx.RequestError as e:
        logger.error(f"MCP API request error: {e}")
        return None, None
    except json.JSONDecodeError as e:
        logger.error(f"MCP API response JSON parse error: {e}")
        return None, None
    except Exception as e:
        logger.error(f"MCP API unexpected exception: {e}", exc_info=True)
        return None, None


def build_web_search_continuation_payload(
    base_payload: Dict[str, Any],
    query: str,
    tool_use_id: str,
    results: Dict,
    assistant_text: str,
) -> Dict[str, Any]:
    """
    Build a follow-up Kiro payload that feeds web_search results back to the model.

    When the model calls web_search, the assistant turn ends after emitting the
    tool_use request. To let the model actually consume the search results and
    produce a final answer, the gateway must issue a second Kiro request that
    contains the completed tool-use round-trip in the conversation history.

    This constructor is API-agnostic: the Kiro payload structure is identical for
    OpenAI and Anthropic clients, so both streaming paths share this logic.

    Transformation applied to a copy of ``base_payload``:
    1. The original ``currentMessage`` is appended to ``history`` (it was the
       user turn that triggered the search).
    2. A synthetic ``assistantResponseMessage`` carrying the web_search
       ``toolUses`` entry is appended (the assistant's decision to search).
    3. A new ``currentMessage`` is created: a user turn whose
       ``userInputMessageContext.toolResults`` holds the search summary, while
       preserving the ``tools`` definition so the model can search again.

    Args:
        base_payload: The original Kiro payload sent for this request. Not
            mutated — a deep copy is returned.
        query: The search query the model requested.
        tool_use_id: The tool-use identifier linking the toolUse to its result.
        results: Parsed MCP web_search response (dict with a "results" key).
        assistant_text: Any assistant text emitted before the tool call. Used as
            the assistant message content; falls back to a placeholder if empty
            (Kiro API requires non-empty content).

    Returns:
        A new Kiro payload dict ready to POST to /generateAssistantResponse.

    Raises:
        ValueError: If ``base_payload`` is missing the expected
            ``conversationState.currentMessage`` structure.
    """
    import copy

    conversation_state = base_payload.get("conversationState")
    if not isinstance(conversation_state, dict):
        raise ValueError("base_payload missing 'conversationState'")

    current_message = conversation_state.get("currentMessage")
    if not isinstance(current_message, dict) or "userInputMessage" not in current_message:
        raise ValueError("base_payload missing 'currentMessage.userInputMessage'")

    # Work on a deep copy so the caller's payload (used for retries) is untouched.
    new_payload = copy.deepcopy(base_payload)
    new_state = new_payload["conversationState"]

    # Preserve the model ID and tools from the original user message so the
    # follow-up turn stays consistent and the model can search again.
    original_user_input = current_message["userInputMessage"]
    model_id = original_user_input.get("modelId", "")
    original_context = original_user_input.get("userInputMessageContext", {})
    tools = original_context.get("tools") if isinstance(original_context, dict) else None

    # 1. Move the original current message into history.
    history = new_state.get("history")
    if not isinstance(history, list):
        history = []
    history.append(copy.deepcopy(current_message))

    # 2. Append the assistant's tool-use decision.
    assistant_content = assistant_text.strip() if assistant_text else ""
    if not assistant_content:
        assistant_content = "(empty placeholder)"
    history.append({
        "assistantResponseMessage": {
            "content": assistant_content,
            "toolUses": [{
                "name": "web_search",
                "input": {"query": query},
                "toolUseId": tool_use_id,
            }],
        }
    })

    # 3. Build the new current message carrying the tool result.
    summary = generate_search_summary(query, results)
    new_user_input: Dict[str, Any] = {
        "content": "(empty placeholder)",
        "modelId": model_id,
        "origin": "AI_EDITOR",
    }
    new_context: Dict[str, Any] = {
        "toolResults": [{
            "content": [{"text": summary}],
            "status": "success",
            "toolUseId": tool_use_id,
        }]
    }
    # Preserve tools so the model may call web_search again on the next round.
    if tools:
        new_context["tools"] = tools
    new_user_input["userInputMessageContext"] = new_context

    new_state["history"] = history
    new_state["currentMessage"] = {"userInputMessage": new_user_input}

    return new_payload


def generate_search_summary(query: str, results: Dict) -> str:
    """
    Generate human-readable summary from search results wrapped in XML tags.
    
    Wraps results in <web_search>...</web_search> tags to visually distinguish
    tool output from model's own text. Returns FULL snippets without truncation
    so the model has complete information.
    
    Format per result:
    - Title
    - Published date (converted from milliseconds timestamp)
    - URL
    - Full snippet (no truncation)
    
    Args:
        query: Original search query
        results: Parsed MCP response (dict with "results" key)
    
    Returns:
        Formatted summary text wrapped in XML tags with full snippets
    
    Example:
        '<web_search>\nSearch results for "Python tutorials":\n\n
        1. Title: **Learn Python - Official Tutorial**\n
           Published: 13 Mar 2025 14:23:45\n
           URL: https://python.org/tutorial\n
           [Full snippet text without truncation]\n\n
        </web_search>'
    """
    # Start with opening tag
    summary = f'\n<web_search>\nSearch results for "{query}":\n\n'
    
    if results and "results" in results:
        for i, result in enumerate(results["results"], 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            snippet = result.get("snippet", "")
            published_date_ms = result.get("publishedDate")
            
            # Format: Title
            summary += f"{i}. Title: **{title}**\n"
            
            # Format: Published date (convert from milliseconds timestamp)
            if published_date_ms:
                try:
                    # Convert milliseconds to seconds for datetime
                    dt = datetime.fromtimestamp(published_date_ms / 1000)
                    # Format as "13 Mar 2025 14:23:45"
                    date_str = dt.strftime("%d %b %Y %H:%M:%S")
                    summary += f"   Published: {date_str}\n"
                except (ValueError, OSError):
                    # Invalid timestamp - skip date
                    pass
            
            # Format: URL
            if url:
                summary += f"   URL: {url}\n"
            
            # Format: Snippet (NO truncation - model needs full information)
            if snippet:
                summary += f"   {snippet}\n"
            
            summary += "\n"
    else:
        summary += "No results found.\n"
    
    # Close with closing tag
    summary += "</web_search>\n"
    
    return summary


# ==================================================================================================
# SSE Emulation (Anthropic Format)
# ==================================================================================================

async def generate_anthropic_web_search_sse(
    model: str,
    query: str,
    tool_use_id: str,
    results: Dict,
    input_tokens: int
):
    """
    Generate Anthropic SSE stream for web_search response.
    
    Emulates 11 events (EXACT structure from architecture):
    1. message_start - with usage (input_tokens)
    2. content_block_start - server_tool_use
    3. content_block_delta - input_json_delta (query)
    4. content_block_stop - server_tool_use
    5. content_block_start - web_search_tool_result
    6. content_block_stop - web_search_tool_result
    7. content_block_start - text
    8-N. content_block_delta - text_delta (summary chunks)
    N+1. content_block_stop - text
    N+2. message_delta - stop_reason + output_tokens
    N+3. message_stop
    
    Args:
        model: Model name
        query: Search query
        tool_use_id: Tool use ID
        results: Search results dict
        input_tokens: Input token count
    
    Yields:
        SSE formatted strings
    """
    from kiro.streaming_anthropic import format_sse_event
    
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    summary = generate_search_summary(query, results)
    
    # Count output tokens WITHOUT Claude correction (MCP API response, not model generation)
    output_tokens = count_tokens(summary, apply_claude_correction=False)
    
    # Event 1: message_start
    yield format_sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0}
        }
    })
    
    # Event 2: content_block_start (server_tool_use)
    yield format_sse_event("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "id": tool_use_id,
            "type": "server_tool_use",
            "name": "web_search",
            "input": {}
        }
    })
    
    # Event 3: content_block_delta (input_json_delta)
    yield format_sse_event("content_block_delta", {
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "input_json_delta",
            "partial_json": json.dumps({"query": query})
        }
    })
    
    # Event 4: content_block_stop (server_tool_use)
    yield format_sse_event("content_block_stop", {
        "type": "content_block_stop",
        "index": 0
    })
    
    # Event 5: content_block_start (web_search_tool_result)
    search_content = []
    for r in results.get("results", []):
        search_content.append({
            "type": "web_search_result",
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "encrypted_content": r.get("snippet", ""),
            "page_age": None
        })
    
    yield format_sse_event("content_block_start", {
        "type": "content_block_start",
        "index": 1,
        "content_block": {
            "type": "web_search_tool_result",
            "tool_use_id": tool_use_id,
            "content": search_content
        }
    })
    
    # Event 6: content_block_stop (web_search_tool_result)
    yield format_sse_event("content_block_stop", {
        "type": "content_block_stop",
        "index": 1
    })
    
    # Event 7: content_block_start (text)
    yield format_sse_event("content_block_start", {
        "type": "content_block_start",
        "index": 2,
        "content_block": {"type": "text", "text": ""}
    })
    
    # Events 8-N: content_block_delta (text_delta) - stream summary in chunks
    chunk_size = 100
    for i in range(0, len(summary), chunk_size):
        chunk = summary[i:i + chunk_size]
        yield format_sse_event("content_block_delta", {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "text_delta", "text": chunk}
        })
    
    # Event N+1: content_block_stop (text)
    yield format_sse_event("content_block_stop", {
        "type": "content_block_stop",
        "index": 2
    })
    
    # Event N+2: message_delta
    yield format_sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": output_tokens}
    })
    
    # Event N+3: message_stop
    yield format_sse_event("message_stop", {
        "type": "message_stop"
    })


# ==================================================================================================
# SSE Emulation (OpenAI Format)
# ==================================================================================================

async def generate_openai_web_search_sse(
    model: str,
    query: str,
    tool_use_id: str,
    results: Dict,
    input_tokens: int
):
    """
    Generate OpenAI SSE stream for web_search response.
    
    CRITICAL: OpenAI format is COMPLETELY different from Anthropic:
    - data: {...} WITHOUT event: prefix
    - choices[0].delta structure
    - finish_reason instead of stop_reason
    - chat.completion.chunk object type
    
    Emulates server-side execution: returns summary directly as content,
    WITHOUT tool_calls flow (model doesn't call tool, MCP API already executed).
    
    Args:
        model: Model name
        query: Search query
        tool_use_id: Tool use ID (not used in OpenAI format)
        results: Search results dict
        input_tokens: Input token count
    
    Yields:
        SSE formatted strings (OpenAI format)
    
    Example OpenAI SSE:
        data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"claude-sonnet-4","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}
        
        data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"claude-sonnet-4","choices":[{"index":0,"delta":{"content":"Here"},"finish_reason":null}]}
        
        data: [DONE]
    """
    from kiro.utils import generate_completion_id
    
    completion_id = generate_completion_id()
    created_time = int(time.time())
    summary = generate_search_summary(query, results)
    
    # Count output tokens WITHOUT Claude correction (MCP API response, not model generation)
    output_tokens = count_tokens(summary, apply_claude_correction=False)
    
    # Chunk 1: role
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    
    # Chunks 2-N: content (stream summary in chunks)
    chunk_size = 100
    for i in range(0, len(summary), chunk_size):
        content_chunk = summary[i:i + chunk_size]
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": content_chunk},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    
    # Chunk N+1: finish_reason + usage
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
    }
    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    
    # Final: [DONE]
    yield "data: [DONE]\n\n"


# ==================================================================================================
# Path A: Native Anthropic Handler
# ==================================================================================================

def extract_query_from_messages(messages, api_format: str) -> Optional[str]:
    """
    Extract search query from first user message.
    
    IMPORTANT: messages is a list of Pydantic models, NOT dicts.
    Use hasattr() and getattr() to access fields.
    
    LIMITATION: Extracts query only from the FIRST message (single-turn).
    This is acceptable for web_search use case.
    
    Args:
        messages: List of messages from request (Pydantic models)
        api_format: "anthropic" or "openai"
    
    Returns:
        Search query string or None
    """
    if not messages:
        return None
    
    first_msg = messages[0]
    
    # Extract content (Pydantic model - always use hasattr/getattr)
    content = getattr(first_msg, 'content', None)
    if content is None:
        return None
    
    # Convert to text
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        # Extract text from content blocks (могут быть Pydantic модели или dict)
        text_parts = []
        for block in content:
            # Handle both Pydantic models and dicts
            if hasattr(block, 'type') and hasattr(block, 'text'):
                # Pydantic model (TextContentBlock)
                if block.type == "text":
                    text_parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                # Dict format
                text_parts.append(block.get("text", ""))
        text = "".join(text_parts)
    else:
        return None
    
    # Remove prefix if present (some clients add this)
    prefix = "Perform a web search for the query: "
    if text.startswith(prefix):
        query = text[len(prefix):]
    else:
        query = text
    
    return query.strip() if query.strip() else None


async def handle_native_web_search(
    request,
    request_data,
    auth_manager,
    api_format: str = "anthropic"
):
    """
    Handle native Anthropic web_search (Path A).
    
    This function bypasses /generateAssistantResponse entirely.
    Direct MCP API call → SSE emulation → return to client.
    
    Args:
        request: FastAPI Request
        request_data: Validated request (AnthropicMessagesRequest or ChatCompletionRequest)
        auth_manager: KiroAuthManager instance
        api_format: "anthropic" or "openai"
    
    Returns:
        StreamingResponse or JSONResponse
    """
    # Extract query from first user message
    query = extract_query_from_messages(request_data.messages, api_format)
    if not query:
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "Cannot extract search query from messages"
                }
            }
        )
    
    logger.info(f"WebSearch query (Path A - native): {query}")
    
    # Call MCP API
    tool_use_id, results = await call_kiro_mcp_api(query, auth_manager)
    
    if results is None:
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "Web search failed. Please try again."
                }
            }
        )
    
    # Count tokens WITHOUT Claude correction (MCP API, not model)
    input_tokens = count_message_tokens(
        [msg.model_dump() for msg in request_data.messages],
        apply_claude_correction=False
    )
    
    # Return response based on streaming mode and API format
    if request_data.stream:
        # Streaming mode - generate SSE
        logger.debug(f"Returning streaming web_search response (api_format={api_format})")
        
        if api_format == "openai":
            sse_generator = generate_openai_web_search_sse(
                request_data.model,
                query,
                tool_use_id,
                results,
                input_tokens
            )
        else:  # anthropic
            sse_generator = generate_anthropic_web_search_sse(
                request_data.model,
                query,
                tool_use_id,
                results,
                input_tokens
            )
        
        return StreamingResponse(
            sse_generator,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    else:
        # Non-streaming mode - return full JSON
        logger.debug(f"Returning non-streaming web_search response (api_format={api_format})")
        summary = generate_search_summary(query, results)
        
        # Count output tokens WITHOUT Claude correction (MCP API response)
        output_tokens = count_tokens(summary, apply_claude_correction=False)
        
        if api_format == "openai":
            # OpenAI format: chat.completion
            from kiro.utils import generate_completion_id
            completion_id = generate_completion_id()
            created_time = int(time.time())
            
            full_response = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created_time,
                "model": request_data.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": summary
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }
            }
        else:  # anthropic
            # Anthropic format: message
            message_id = f"msg_{uuid.uuid4().hex[:24]}"
            
            # Build search results content
            search_content = []
            for r in results.get("results", []):
                search_content.append({
                    "type": "web_search_result",
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "encrypted_content": r.get("snippet", ""),
                    "page_age": None
                })
            
            full_response = {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": tool_use_id,
                        "name": "web_search",
                        "input": {"query": query}
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": tool_use_id,
                        "content": search_content
                    },
                    {
                        "type": "text",
                        "text": summary
                    }
                ],
                "model": request_data.model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
            }
        
        return JSONResponse(content=full_response)

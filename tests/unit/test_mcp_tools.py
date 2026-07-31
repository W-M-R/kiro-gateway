# -*- coding: utf-8 -*-

"""
Tests for MCP Tools Support (WebSearch).

Tests cover:
- ID generation
- MCP API calls
- Search summary generation
- Query extraction from messages
- Native web_search handler (Path A)
- SSE emulation (Anthropic and OpenAI formats)
"""

import json
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime

from kiro.mcp_tools import (
    generate_random_id,
    call_kiro_mcp_api,
    generate_search_summary,
    extract_query_from_messages,
    handle_native_web_search,
    generate_anthropic_web_search_sse,
    generate_openai_web_search_sse,
    build_web_search_continuation_payload,
    web_search_event_source,
    WebSearchContinuation,
    _extract_web_search_query,
)
from kiro.streaming_core import KiroEvent


def _make_base_payload(with_history=False, with_tools=True):
    """
    Build a minimal Kiro payload for continuation tests.

    Args:
        with_history: Include a pre-existing history entry.
        with_tools: Include a tools definition in the current message context.

    Returns:
        A Kiro payload dict shaped like build_kiro_payload output.
    """
    context = {}
    if with_tools:
        context["tools"] = [{
            "toolSpecification": {
                "name": "web_search",
                "description": "Search the web",
                "inputSchema": {"json": {"type": "object"}}
            }
        }]
    current = {
        "userInputMessage": {
            "content": "What is the weather in Paris?",
            "modelId": "claude-sonnet-4.5",
            "origin": "AI_EDITOR",
        }
    }
    if context:
        current["userInputMessage"]["userInputMessageContext"] = context

    state = {
        "chatTriggerType": "MANUAL",
        "conversationId": "conv-123",
        "currentMessage": current,
    }
    if with_history:
        state["history"] = [{
            "userInputMessage": {"content": "hi", "modelId": "claude-sonnet-4.5"}
        }, {
            "assistantResponseMessage": {"content": "hello"}
        }]

    return {"conversationState": state, "profileArn": "arn:test"}


def _make_results(n=2):
    """Build a fake MCP web_search results dict with n entries."""
    return {
        "results": [
            {
                "title": f"Result {i}",
                "url": f"https://example.com/{i}",
                "snippet": f"Snippet {i}",
                "publishedDate": 1700000000000,
            }
            for i in range(n)
        ],
        "totalResults": n,
        "query": "test",
    }


def _make_ws_tool(query="paris weather", as_string=False, anthropic=False):
    """Build a web_search tool_use dict in OpenAI or Anthropic shape."""
    if anthropic:
        return {"id": "toolu_1", "name": "web_search", "input": {"query": query}}
    args = json.dumps({"query": query}) if as_string else {"query": query}
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": "web_search", "arguments": args},
    }


def _events_for(response, mapping):
    """
    Build a mock parse_kiro_stream that yields per-response event lists.

    Args:
        response: unused (signature compatibility)
        mapping: dict {response_obj: [KiroEvent, ...]}

    Returns:
        An async generator function suitable for patching parse_kiro_stream.
    """
    async def _mock(resp, *args, **kwargs):
        for ev in mapping.get(resp, []):
            yield ev
    return _mock


# ==================================================================================================
# Tests for ID Generation
# ==================================================================================================

class TestIDGeneration:
    """Tests for random ID generation."""
    
    def test_generate_random_id_length(self):
        """
        What it does: Verifies ID generation with exact length.
        Purpose: Ensure generate_random_id returns correct length.
        """
        print("Setup: Generating IDs of different lengths...")
        
        print("Action: Generate ID of length 22...")
        id_22 = generate_random_id(22)
        print(f"Comparing length: Expected 22, Got {len(id_22)}")
        assert len(id_22) == 22
        
        print("Action: Generate ID of length 8...")
        id_8 = generate_random_id(8)
        print(f"Comparing length: Expected 8, Got {len(id_8)}")
        assert len(id_8) == 8
        
        print("Action: Generate ID of length 100...")
        id_100 = generate_random_id(100)
        print(f"Comparing length: Expected 100, Got {len(id_100)}")
        assert len(id_100) == 100
    
    def test_generate_random_id_alphanumeric(self):
        """
        What it does: Verifies ID contains only alphanumeric characters.
        Purpose: Ensure no special characters in generated IDs.
        """
        print("Setup: Generating large ID to test character set...")
        
        print("Action: Generate ID of length 1000...")
        random_id = generate_random_id(1000)
        
        print(f"Checking if alphanumeric: {random_id[:50]}...")
        assert random_id.isalnum()
    
    def test_generate_random_id_uniqueness(self):
        """
        What it does: Verifies IDs are unique (probabilistically).
        Purpose: Ensure randomness works correctly.
        """
        print("Setup: Generating multiple IDs...")
        
        print("Action: Generate 100 IDs of length 22...")
        ids = [generate_random_id(22) for _ in range(100)]
        
        print(f"Comparing uniqueness: Generated {len(ids)} IDs, unique: {len(set(ids))}")
        assert len(set(ids)) == len(ids)  # All should be unique


# ==================================================================================================
# Tests for MCP API Call
# ==================================================================================================

class TestCallKiroMCPAPI:
    """Tests for MCP API calls."""
    
    @pytest.mark.asyncio
    async def test_mcp_api_success(self, mock_auth_manager):
        """
        What it does: Verifies successful MCP API call and result parsing.
        Purpose: Ensure MCP API integration works correctly.
        """
        print("Setup: Mocking successful MCP API response...")
        query = "Python tutorials"
        
        # Mock MCP response (CRITICAL: result.content[0].text is JSON STRING)
        mock_response_data = {
            "id": "web_search_tooluse_abc123_1234567890_xyz",
            "jsonrpc": "2.0",
            "result": {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "results": [
                            {
                                "title": "Python Tutorial",
                                "url": "https://python.org",
                                "snippet": "Learn Python programming",
                                "publishedDate": 1700000000000
                            }
                        ],
                        "totalResults": 1,
                        "query": "Python tutorials"
                    })
                }],
                "isError": False
            }
        }
        
        # Mock httpx.AsyncClient - CRITICAL: json() must be async
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value=mock_response_data)
        
        mock_post = AsyncMock(return_value=mock_response)
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = mock_post
        
        print("Action: Calling call_kiro_mcp_api...")
        with patch("kiro.mcp_tools.httpx.AsyncClient", return_value=mock_client):
            tool_use_id, results = await call_kiro_mcp_api(query, mock_auth_manager)
        
        print(f"Comparing tool_use_id: Got '{tool_use_id}'")
        assert tool_use_id is not None
        assert tool_use_id.startswith("srvtoolu_")
        
        print(f"Comparing results: Got {results}")
        assert results is not None
        assert results["totalResults"] == 1
        assert results["results"][0]["title"] == "Python Tutorial"
        assert results["results"][0]["url"] == "https://python.org"
    
    @pytest.mark.asyncio
    async def test_mcp_api_error_response(self, mock_auth_manager):
        """
        What it does: Verifies handling of MCP API error response.
        Purpose: Ensure errors are handled gracefully.
        """
        print("Setup: Mocking MCP API error response...")
        query = "test"
        
        # Mock error response
        mock_response_data = {
            "id": "web_search_tooluse_abc123_1234567890_xyz",
            "jsonrpc": "2.0",
            "error": {"code": -32600, "message": "Invalid request"}
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value=mock_response_data)
        
        mock_post = AsyncMock(return_value=mock_response)
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = mock_post
        
        print("Action: Calling call_kiro_mcp_api...")
        with patch("kiro.mcp_tools.httpx.AsyncClient", return_value=mock_client):
            tool_use_id, results = await call_kiro_mcp_api(query, mock_auth_manager)
        
        print(f"Comparing result: Expected (None, None), Got ({tool_use_id}, {results})")
        assert tool_use_id is None
        assert results is None
    
    @pytest.mark.asyncio
    async def test_mcp_api_http_error(self, mock_auth_manager):
        """
        What it does: Verifies handling of HTTP errors from MCP API.
        Purpose: Ensure non-200 status codes are handled.
        """
        print("Setup: Mocking HTTP 500 error...")
        query = "test"
        
        mock_response = Mock()
        mock_response.status_code = 500
        
        mock_post = AsyncMock(return_value=mock_response)
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = mock_post
        
        print("Action: Calling call_kiro_mcp_api...")
        with patch("kiro.mcp_tools.httpx.AsyncClient", return_value=mock_client):
            tool_use_id, results = await call_kiro_mcp_api(query, mock_auth_manager)
        
        print(f"Comparing result: Expected (None, None), Got ({tool_use_id}, {results})")
        assert tool_use_id is None
        assert results is None
    
    @pytest.mark.asyncio
    async def test_mcp_api_timeout(self, mock_auth_manager):
        """
        What it does: Verifies handling of MCP API timeout.
        Purpose: Ensure timeouts are handled gracefully.
        """
        print("Setup: Mocking timeout exception...")
        query = "test"
        
        import httpx
        
        mock_post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = mock_post
        
        print("Action: Calling call_kiro_mcp_api...")
        with patch("kiro.mcp_tools.httpx.AsyncClient", return_value=mock_client):
            tool_use_id, results = await call_kiro_mcp_api(query, mock_auth_manager)
        
        print(f"Comparing result: Expected (None, None), Got ({tool_use_id}, {results})")
        assert tool_use_id is None
        assert results is None
    
    @pytest.mark.asyncio
    async def test_mcp_api_json_decode_error(self, mock_auth_manager):
        """
        What it does: Verifies handling of malformed JSON in MCP response.
        Purpose: Ensure JSON parsing errors are handled.
        """
        print("Setup: Mocking malformed JSON response...")
        query = "test"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
        
        mock_post = AsyncMock(return_value=mock_response)
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = mock_post
        
        print("Action: Calling call_kiro_mcp_api...")
        with patch("kiro.mcp_tools.httpx.AsyncClient", return_value=mock_client):
            tool_use_id, results = await call_kiro_mcp_api(query, mock_auth_manager)
        
        print(f"Comparing result: Expected (None, None), Got ({tool_use_id}, {results})")
        assert tool_use_id is None
        assert results is None


# ==================================================================================================
# Tests for Search Summary Generation
# ==================================================================================================

class TestGenerateSearchSummary:
    """Tests for search summary formatting."""
    
    def test_generate_summary_with_results(self):
        """
        What it does: Verifies summary formatting with results.
        Purpose: Ensure XML tags and proper formatting.
        """
        print("Setup: Creating mock search results...")
        query = "Python"
        results = {
            "results": [
                {
                    "title": "Python.org",
                    "url": "https://python.org",
                    "snippet": "Official Python website with tutorials",
                    "publishedDate": 1700000000000
                },
                {
                    "title": "Python Tutorial",
                    "url": "https://docs.python.org",
                    "snippet": "Complete Python documentation",
                    "publishedDate": None  # No date
                }
            ],
            "totalResults": 2
        }
        
        print("Action: Generating summary...")
        summary = generate_search_summary(query, results)
        
        print(f"Checking XML tags...")
        assert "<web_search>" in summary
        assert "</web_search>" in summary
        
        print(f"Checking query in summary...")
        assert "Python" in summary
        
        print(f"Checking first result...")
        assert "Python.org" in summary
        assert "https://python.org" in summary
        assert "Official Python website with tutorials" in summary
        
        print(f"Checking second result...")
        assert "Python Tutorial" in summary
        assert "https://docs.python.org" in summary
        assert "Complete Python documentation" in summary
    
    def test_generate_summary_no_results(self):
        """
        What it does: Verifies summary with empty results list.
        Purpose: Ensure empty results are handled gracefully.
        """
        print("Setup: Creating empty results...")
        query = "nonexistent"
        results = {"results": [], "totalResults": 0}
        
        print("Action: Generating summary...")
        summary = generate_search_summary(query, results)
        
        print(f"Checking XML tags...")
        assert "<web_search>" in summary
        assert "</web_search>" in summary
        
        print(f"Checking query in summary...")
        assert "nonexistent" in summary
        
        print(f"Summary content: {repr(summary)}")
        # Empty results list produces empty content between tags (no "No results found")
        assert "Search results for" in summary
    
    def test_generate_summary_malformed_results(self):
        """
        What it does: Verifies handling of malformed results.
        Purpose: Ensure graceful handling of invalid data.
        """
        print("Setup: Creating malformed results...")
        query = "test"
        results = {"invalid": "structure"}
        
        print("Action: Generating summary...")
        summary = generate_search_summary(query, results)
        
        print(f"Checking for 'No results found'...")
        assert "No results found" in summary
    
    def test_generate_summary_date_formatting(self):
        """
        What it does: Verifies date formatting from milliseconds timestamp.
        Purpose: Ensure publishedDate is converted correctly.
        """
        print("Setup: Creating result with timestamp...")
        query = "test"
        # 1700000000000 ms = 2023-11-14 22:13:20 UTC
        results = {
            "results": [{
                "title": "Test",
                "url": "https://test.com",
                "snippet": "Test snippet",
                "publishedDate": 1700000000000
            }],
            "totalResults": 1
        }
        
        print("Action: Generating summary...")
        summary = generate_search_summary(query, results)
        
        print(f"Checking date format...")
        # Should contain formatted date like "14 Nov 2023"
        assert "Nov 2023" in summary or "Ноя 2023" in summary  # Depends on locale
    
    def test_generate_summary_full_snippet_no_truncation(self):
        """
        What it does: Verifies snippets are NOT truncated.
        Purpose: Ensure model gets full information.
        """
        print("Setup: Creating result with long snippet...")
        query = "test"
        long_snippet = "A" * 1000  # 1000 characters
        results = {
            "results": [{
                "title": "Test",
                "url": "https://test.com",
                "snippet": long_snippet,
                "publishedDate": None
            }],
            "totalResults": 1
        }
        
        print("Action: Generating summary...")
        summary = generate_search_summary(query, results)
        
        print(f"Checking snippet is NOT truncated...")
        assert long_snippet in summary
        assert len(long_snippet) == 1000  # Full length preserved


# ==================================================================================================
# Tests for Query Extraction
# ==================================================================================================

class TestExtractQueryFromMessages:
    """Tests for query extraction from messages."""
    
    def test_extract_query_anthropic_string_content(self):
        """
        What it does: Extracts query from Anthropic string content.
        Purpose: Ensure simple string messages work.
        """
        print("Setup: Creating Anthropic message with string content...")
        from kiro.models_anthropic import AnthropicMessage
        messages = [AnthropicMessage(role="user", content="Search for Python tutorials")]
        
        print("Action: Extracting query...")
        query = extract_query_from_messages(messages, "anthropic")
        
        print(f"Comparing query: Expected 'Search for Python tutorials', Got '{query}'")
        assert query == "Search for Python tutorials"
    
    def test_extract_query_anthropic_list_content(self):
        """
        What it does: Extracts query from Anthropic list content.
        Purpose: Ensure content blocks work.
        """
        print("Setup: Creating Anthropic message with list content...")
        from kiro.models_anthropic import AnthropicMessage, TextContentBlock
        messages = [AnthropicMessage(
            role="user",
            content=[TextContentBlock(type="text", text="Python tutorials")]
        )]
        
        print("Action: Extracting query...")
        query = extract_query_from_messages(messages, "anthropic")
        
        print(f"Comparing query: Expected 'Python tutorials', Got '{query}'")
        assert query == "Python tutorials"
    
    def test_extract_query_with_prefix(self):
        """
        What it does: Removes 'Perform a web search for the query:' prefix.
        Purpose: Ensure prefix is stripped correctly.
        """
        print("Setup: Creating message with prefix...")
        from kiro.models_anthropic import AnthropicMessage
        messages = [AnthropicMessage(
            role="user",
            content="Perform a web search for the query: Python"
        )]
        
        print("Action: Extracting query...")
        query = extract_query_from_messages(messages, "anthropic")
        
        print(f"Comparing query: Expected 'Python', Got '{query}'")
        assert query == "Python"
    
    def test_extract_query_empty_messages(self):
        """
        What it does: Handles empty messages list.
        Purpose: Ensure None is returned for empty input.
        """
        print("Setup: Creating empty messages list...")
        messages = []
        
        print("Action: Extracting query...")
        query = extract_query_from_messages(messages, "anthropic")
        
        print(f"Comparing query: Expected None, Got {query}")
        assert query is None
    
    def test_extract_query_no_text_content(self):
        """
        What it does: Handles messages without text content.
        Purpose: Ensure None is returned for non-text messages.
        """
        print("Setup: Creating message with image content...")
        from kiro.models_anthropic import AnthropicMessage, ImageContentBlock, Base64ImageSource
        messages = [AnthropicMessage(
            role="user",
            content=[ImageContentBlock(
                type="image",
                source=Base64ImageSource(
                    type="base64",
                    media_type="image/png",
                    data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                )
            )]
        )]
        
        print("Action: Extracting query...")
        query = extract_query_from_messages(messages, "anthropic")
        
        print(f"Comparing query: Expected None or empty, Got '{query}'")
        assert query is None or query == ""
    
    def test_extract_query_multiple_text_blocks(self):
        """
        What it does: Concatenates multiple text blocks.
        Purpose: Ensure all text is extracted.
        """
        print("Setup: Creating message with multiple text blocks...")
        from kiro.models_anthropic import AnthropicMessage, TextContentBlock
        messages = [AnthropicMessage(
            role="user",
            content=[
                TextContentBlock(type="text", text="Search for "),
                TextContentBlock(type="text", text="Python tutorials")
            ]
        )]
        
        print("Action: Extracting query...")
        query = extract_query_from_messages(messages, "anthropic")
        
        print(f"Comparing query: Expected 'Search for Python tutorials', Got '{query}'")
        assert query == "Search for Python tutorials"


# ==================================================================================================
# Tests for SSE Emulation
# ==================================================================================================

class TestAnthropicSSEEmulation:
    """Tests for Anthropic SSE stream generation."""
    
    @pytest.mark.asyncio
    async def test_generate_anthropic_sse_structure(self):
        """
        What it does: Verifies Anthropic SSE event structure.
        Purpose: Ensure all 11 events are generated correctly.
        """
        print("Setup: Preparing test data...")
        model = "claude-sonnet-4"
        query = "Python"
        tool_use_id = "srvtoolu_test123"
        results = {
            "results": [{"title": "Test", "url": "https://test.com", "snippet": "Test"}],
            "totalResults": 1
        }
        input_tokens = 100
        
        print("Action: Generating SSE stream...")
        events = []
        async for event in generate_anthropic_web_search_sse(model, query, tool_use_id, results, input_tokens):
            events.append(event)
        
        print(f"Comparing event count: Got {len(events)} events")
        assert len(events) >= 11  # At least 11 events (may have more text_delta chunks)
        
        print("Checking event types...")
        event_types = []
        for event in events:
            if "event:" in event:
                event_type = event.split("event:")[1].split("\n")[0].strip()
                event_types.append(event_type)
        
        print(f"Event types: {event_types}")
        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types


class TestOpenAISSEEmulation:
    """Tests for OpenAI SSE stream generation."""
    
    @pytest.mark.asyncio
    async def test_generate_openai_sse_structure(self):
        """
        What it does: Verifies OpenAI SSE event structure.
        Purpose: Ensure OpenAI format is correct.
        """
        print("Setup: Preparing test data...")
        model = "claude-sonnet-4"
        query = "Python"
        tool_use_id = "srvtoolu_test123"
        results = {
            "results": [{"title": "Test", "url": "https://test.com", "snippet": "Test"}],
            "totalResults": 1
        }
        input_tokens = 100
        
        print("Action: Generating SSE stream...")
        chunks = []
        async for chunk in generate_openai_web_search_sse(model, query, tool_use_id, results, input_tokens):
            chunks.append(chunk)
        
        print(f"Comparing chunk count: Got {len(chunks)} chunks")
        assert len(chunks) >= 3  # At least: role, content chunks, finish + [DONE]
        
        print("Checking for [DONE] marker...")
        assert any("[DONE]" in chunk for chunk in chunks)
        
        print("Checking for role delta (flexible matching)...")
        assert any('"role"' in chunk and '"assistant"' in chunk for chunk in chunks)
        
        print("Checking for finish_reason (flexible matching)...")
        assert any('"finish_reason"' in chunk and '"stop"' in chunk for chunk in chunks)
        
        print("Checking for data: prefix...")
        assert any(chunk.startswith("data:") for chunk in chunks)
        
        print("Checking for usage information...")
        assert any('"usage"' in chunk for chunk in chunks)


# ==================================================================================================
# Tests for web_search query extraction helper
# ==================================================================================================

class TestExtractWebSearchQuery:
    """Tests for _extract_web_search_query."""

    def test_openai_dict_arguments(self):
        """OpenAI-style tool with dict arguments extracts query."""
        tool = _make_ws_tool(query="paris weather")
        assert _extract_web_search_query(tool) == "paris weather"

    def test_openai_string_arguments(self):
        """OpenAI-style tool with JSON-string arguments extracts query."""
        tool = _make_ws_tool(query="london news", as_string=True)
        assert _extract_web_search_query(tool) == "london news"

    def test_anthropic_input_field(self):
        """Anthropic-style tool with input field extracts query."""
        tool = _make_ws_tool(query="tokyo weather", anthropic=True)
        assert _extract_web_search_query(tool) == "tokyo weather"

    def test_invalid_json_string_returns_none(self):
        """Malformed JSON string arguments yield None."""
        tool = {
            "id": "c1", "type": "function",
            "function": {"name": "web_search", "arguments": "{not json"},
        }
        assert _extract_web_search_query(tool) is None

    def test_missing_query_returns_none(self):
        """Arguments without query key yield None."""
        tool = {
            "id": "c1", "type": "function",
            "function": {"name": "web_search", "arguments": {"q": "x"}},
        }
        assert _extract_web_search_query(tool) is None

    def test_empty_query_returns_none(self):
        """Empty/whitespace query yields None."""
        tool = _make_ws_tool(query="   ")
        assert _extract_web_search_query(tool) is None

    def test_non_dict_arguments_returns_none(self):
        """Non-dict, non-string arguments yield None."""
        tool = {
            "id": "c1", "type": "function",
            "function": {"name": "web_search", "arguments": 123},
        }
        assert _extract_web_search_query(tool) is None


# ==================================================================================================
# Tests for build_web_search_continuation_payload
# ==================================================================================================

class TestBuildWebSearchContinuationPayload:
    """Tests for the continuation payload constructor."""

    def test_original_payload_not_mutated(self):
        """The caller's base_payload must remain untouched."""
        base = _make_base_payload()
        original = json.loads(json.dumps(base))
        build_web_search_continuation_payload(base, "q", "tid", _make_results(), "text")
        assert base == original

    def test_current_message_moved_to_history(self):
        """The original currentMessage is appended to history."""
        base = _make_base_payload()
        result = build_web_search_continuation_payload(base, "q", "tid", _make_results(), "text")
        history = result["conversationState"]["history"]
        assert history[-2]["userInputMessage"]["content"] == "What is the weather in Paris?"

    def test_assistant_tooluse_appended(self):
        """A synthetic assistantResponseMessage with web_search toolUses is appended."""
        base = _make_base_payload()
        result = build_web_search_continuation_payload(base, "paris", "tid", _make_results(), "")
        assistant = result["conversationState"]["history"][-1]["assistantResponseMessage"]
        assert assistant["toolUses"][0]["name"] == "web_search"
        assert assistant["toolUses"][0]["input"] == {"query": "paris"}
        assert assistant["toolUses"][0]["toolUseId"] == "tid"

    def test_empty_assistant_text_uses_placeholder(self):
        """Empty assistant_text falls back to placeholder (Kiro requires non-empty)."""
        base = _make_base_payload()
        result = build_web_search_continuation_payload(base, "q", "tid", _make_results(), "")
        assert result["conversationState"]["history"][-1]["assistantResponseMessage"]["content"] == "(empty placeholder)"

    def test_assistant_text_preserved(self):
        """Non-empty assistant_text is used as the assistant message content."""
        base = _make_base_payload()
        result = build_web_search_continuation_payload(base, "q", "tid", _make_results(), "Let me search.")
        assert result["conversationState"]["history"][-1]["assistantResponseMessage"]["content"] == "Let me search."

    def test_new_current_message_has_tool_results(self):
        """New currentMessage contains toolResults with the search summary."""
        base = _make_base_payload()
        result = build_web_search_continuation_payload(base, "q", "tid", _make_results(), "")
        current = result["conversationState"]["currentMessage"]["userInputMessage"]
        tool_results = current["userInputMessageContext"]["toolResults"]
        assert len(tool_results) == 1
        assert tool_results[0]["toolUseId"] == "tid"
        assert tool_results[0]["status"] == "success"
        assert "web_search" in tool_results[0]["content"][0]["text"]

    def test_tools_preserved_for_re_search(self):
        """The tools definition is preserved so the model can search again."""
        base = _make_base_payload(with_tools=True)
        result = build_web_search_continuation_payload(base, "q", "tid", _make_results(), "")
        context = result["conversationState"]["currentMessage"]["userInputMessage"]["userInputMessageContext"]
        assert "tools" in context
        assert context["tools"][0]["toolSpecification"]["name"] == "web_search"

    def test_no_tools_stays_absent(self):
        """When base payload has no tools, continuation also has no tools."""
        base = _make_base_payload(with_tools=False)
        result = build_web_search_continuation_payload(base, "q", "tid", _make_results(), "")
        context = result["conversationState"]["currentMessage"]["userInputMessage"]["userInputMessageContext"]
        assert "tools" not in context

    def test_model_id_preserved(self):
        """Model ID carries over from the original current message."""
        base = _make_base_payload()
        result = build_web_search_continuation_payload(base, "q", "tid", _make_results(), "")
        assert result["conversationState"]["currentMessage"]["userInputMessage"]["modelId"] == "claude-sonnet-4.5"

    def test_profile_arn_preserved(self):
        """profileArn survives the deep copy."""
        base = _make_base_payload()
        result = build_web_search_continuation_payload(base, "q", "tid", _make_results(), "")
        assert result["profileArn"] == "arn:test"

    def test_existing_history_extended_not_replaced(self):
        """Pre-existing history entries are preserved, not replaced."""
        base = _make_base_payload(with_history=True)
        result = build_web_search_continuation_payload(base, "q", "tid", _make_results(), "")
        history = result["conversationState"]["history"]
        # 2 original + 1 moved current + 1 synthetic assistant = 4
        assert len(history) == 4
        assert history[0]["userInputMessage"]["content"] == "hi"

    def test_invalid_payload_raises(self):
        """Missing conversationState raises ValueError."""
        with pytest.raises(ValueError):
            build_web_search_continuation_payload({}, "q", "tid", _make_results(), "text")

    def test_missing_current_message_raises(self):
        """Missing currentMessage.userInputMessage raises ValueError."""
        base = {"conversationState": {"currentMessage": {}}}
        with pytest.raises(ValueError):
            build_web_search_continuation_payload(base, "q", "tid", _make_results(), "text")


# ==================================================================================================
# Tests for web_search_event_source
# ==================================================================================================

class TestWebSearchEventSource:
    """Tests for the continuation event source."""

    @pytest.mark.asyncio
    async def test_no_web_search_passes_through(self):
        """Without web_search, all events pass through unchanged."""
        events = [
            KiroEvent(type="content", content="answer"),
            KiroEvent(type="usage", usage={"x": 1}),
        ]
        resp = Mock()
        with patch("kiro.mcp_tools.parse_kiro_stream", _events_for(resp, {resp: events})):
            collected = []
            async for ev in web_search_event_source(resp, None, None, 5.0):
                collected.append(ev)
        assert [e.type for e in collected] == ["content", "usage"]

    @pytest.mark.asyncio
    async def test_continuation_emits_web_search_event_then_model_text(self):
        """web_search triggers a synthetic event, then continuation model text."""
        resp1 = Mock()
        resp2 = Mock()
        resp2.status_code = 200
        events1 = [KiroEvent(type="tool_use", tool_use=_make_ws_tool())]
        events2 = [KiroEvent(type="content", content="It's sunny.")]

        async def send(payload):
            return resp2

        cont = WebSearchContinuation(base_payload=_make_base_payload(), send_request=send, max_iterations=5)

        with patch("kiro.mcp_tools.parse_kiro_stream", _events_for(resp1, {resp1: events1, resp2: events2})):
            with patch("kiro.mcp_tools.call_kiro_mcp_api", new=AsyncMock(return_value=("srvtoolu_1", _make_results()))):
                collected = []
                async for ev in web_search_event_source(resp1, cont, "auth", 5.0):
                    collected.append(ev)

        types = [e.type for e in collected]
        assert "web_search" in types
        assert "content" in types
        ws_event = next(e for e in collected if e.type == "web_search")
        assert ws_event.web_search["query"] == "paris weather"
        assert ws_event.web_search["tool_use_id"] == "srvtoolu_1"
        # The model's follow-up text is the visible answer
        content_events = [e for e in collected if e.type == "content"]
        assert any("sunny" in (e.content or "") for e in content_events)

    @pytest.mark.asyncio
    async def test_continuation_disabled_emits_raw_summary(self):
        """When continuation is None, web_search emits no synthetic event (passthrough)."""
        resp1 = Mock()
        events1 = [KiroEvent(type="tool_use", tool_use=_make_ws_tool())]

        with patch("kiro.mcp_tools.parse_kiro_stream", _events_for(resp1, {resp1: events1})):
            collected = []
            async for ev in web_search_event_source(resp1, None, None, 5.0):
                collected.append(ev)

        # With continuation disabled, the tool_use passes through as-is
        assert len(collected) == 1
        assert collected[0].type == "tool_use"

    @pytest.mark.asyncio
    async def test_mcp_failure_passes_tool_use_through(self):
        """When MCP call fails, the original tool_use event is passed through."""
        resp1 = Mock()
        events1 = [KiroEvent(type="tool_use", tool_use=_make_ws_tool())]

        async def send(payload):
            return Mock(status_code=200)

        cont = WebSearchContinuation(base_payload=_make_base_payload(), send_request=send, max_iterations=5)

        with patch("kiro.mcp_tools.parse_kiro_stream", _events_for(resp1, {resp1: events1})):
            with patch("kiro.mcp_tools.call_kiro_mcp_api", new=AsyncMock(return_value=(None, None))):
                collected = []
                async for ev in web_search_event_source(resp1, cont, "auth", 5.0):
                    collected.append(ev)

        assert len(collected) == 1
        assert collected[0].type == "tool_use"

    @pytest.mark.asyncio
    async def test_iteration_limit_emits_raw_summary(self):
        """Reaching max_iterations emits raw summary as content and stops."""
        resp1 = Mock()
        events1 = [KiroEvent(type="tool_use", tool_use=_make_ws_tool())]

        async def send(payload):
            return Mock(status_code=200)

        cont = WebSearchContinuation(base_payload=_make_base_payload(), send_request=send, max_iterations=1)

        with patch("kiro.mcp_tools.parse_kiro_stream", _events_for(resp1, {resp1: events1})):
            with patch("kiro.mcp_tools.call_kiro_mcp_api", new=AsyncMock(return_value=("srvtoolu_1", _make_results()))):
                collected = []
                async for ev in web_search_event_source(resp1, cont, "auth", 5.0):
                    collected.append(ev)

        types = [e.type for e in collected]
        assert "web_search" in types
        # Raw summary emitted as content (fallback at limit)
        content_events = [e for e in collected if e.type == "content"]
        assert any("web_search" in (e.content or "") for e in content_events)

    @pytest.mark.asyncio
    async def test_non_200_continuation_falls_back_to_raw_summary(self):
        """A non-200 continuation response emits raw summary as fallback."""
        resp1 = Mock()
        resp2 = Mock()
        resp2.status_code = 500
        events1 = [KiroEvent(type="tool_use", tool_use=_make_ws_tool())]

        async def send(payload):
            return resp2

        cont = WebSearchContinuation(base_payload=_make_base_payload(), send_request=send, max_iterations=5)

        with patch("kiro.mcp_tools.parse_kiro_stream", _events_for(resp1, {resp1: events1})):
            with patch("kiro.mcp_tools.call_kiro_mcp_api", new=AsyncMock(return_value=("srvtoolu_1", _make_results()))):
                collected = []
                async for ev in web_search_event_source(resp1, cont, "auth", 5.0):
                    collected.append(ev)

        content_events = [e for e in collected if e.type == "content"]
        assert any("web_search" in (e.content or "") for e in content_events)

    @pytest.mark.asyncio
    async def test_send_request_exception_falls_back_to_raw_summary(self):
        """send_request raising falls back to raw summary instead of crashing."""
        resp1 = Mock()
        events1 = [KiroEvent(type="tool_use", tool_use=_make_ws_tool())]

        async def send(payload):
            raise RuntimeError("boom")

        cont = WebSearchContinuation(base_payload=_make_base_payload(), send_request=send, max_iterations=5)

        with patch("kiro.mcp_tools.parse_kiro_stream", _events_for(resp1, {resp1: events1})):
            with patch("kiro.mcp_tools.call_kiro_mcp_api", new=AsyncMock(return_value=("srvtoolu_1", _make_results()))):
                collected = []
                async for ev in web_search_event_source(resp1, cont, "auth", 5.0):
                    collected.append(ev)

        content_events = [e for e in collected if e.type == "content"]
        assert any("web_search" in (e.content or "") for e in content_events)

    @pytest.mark.asyncio
    async def test_web_search_without_query_skipped(self):
        """A web_search tool_use without a query is skipped (no MCP call)."""
        resp1 = Mock()
        tool = {
            "id": "c1", "type": "function",
            "function": {"name": "web_search", "arguments": {}},
        }
        events1 = [KiroEvent(type="tool_use", tool_use=tool)]

        async def send(payload):
            return Mock(status_code=200)

        cont = WebSearchContinuation(base_payload=_make_base_payload(), send_request=send, max_iterations=5)

        with patch("kiro.mcp_tools.parse_kiro_stream", _events_for(resp1, {resp1: events1})):
            with patch("kiro.mcp_tools.call_kiro_mcp_api", new=AsyncMock()) as mock_mcp:
                collected = []
                async for ev in web_search_event_source(resp1, cont, "auth", 5.0):
                    collected.append(ev)

        mock_mcp.assert_not_called()
        # Nothing emitted, stream ends
        assert collected == []

    @pytest.mark.asyncio
    async def test_multiple_web_search_rounds(self):
        """Multiple consecutive searches trigger multiple continuation rounds."""
        resp1 = Mock()
        resp2 = Mock()
        resp3 = Mock()
        resp2.status_code = 200
        resp3.status_code = 200
        events1 = [KiroEvent(type="tool_use", tool_use=_make_ws_tool(query="q1"))]
        events2 = [KiroEvent(type="tool_use", tool_use=_make_ws_tool(query="q2"))]
        events3 = [KiroEvent(type="content", content="final answer")]

        responses = iter([resp2, resp3])

        async def send(payload):
            return next(responses)

        cont = WebSearchContinuation(base_payload=_make_base_payload(), send_request=send, max_iterations=5)

        mapping = {resp1: events1, resp2: events2, resp3: events3}
        with patch("kiro.mcp_tools.parse_kiro_stream", _events_for(resp1, mapping)):
            with patch("kiro.mcp_tools.call_kiro_mcp_api", new=AsyncMock(side_effect=[
                ("srvtoolu_1", _make_results()),
                ("srvtoolu_2", _make_results()),
            ])):
                collected = []
                async for ev in web_search_event_source(resp1, cont, "auth", 5.0):
                    collected.append(ev)

        ws_events = [e for e in collected if e.type == "web_search"]
        assert len(ws_events) == 2
        content_events = [e for e in collected if e.type == "content"]
        assert any("final answer" in (e.content or "") for e in content_events)

    @pytest.mark.asyncio
    async def test_continuation_response_closed_on_transition(self):
        """The previous continuation response is closed when transitioning."""
        resp1 = Mock()
        resp2 = Mock()
        resp2.status_code = 200
        resp2.aclose = AsyncMock()
        events1 = [KiroEvent(type="tool_use", tool_use=_make_ws_tool())]
        events2 = [KiroEvent(type="content", content="done")]

        async def send(payload):
            return resp2

        cont = WebSearchContinuation(base_payload=_make_base_payload(), send_request=send, max_iterations=5)

        with patch("kiro.mcp_tools.parse_kiro_stream", _events_for(resp1, {resp1: events1, resp2: events2})):
            with patch("kiro.mcp_tools.call_kiro_mcp_api", new=AsyncMock(return_value=("srvtoolu_1", _make_results()))):
                async for _ in web_search_event_source(resp1, cont, "auth", 5.0):
                    pass

        resp2.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initial_response_not_closed(self):
        """The caller-owned initial response is never closed by event_source."""
        resp1 = Mock()
        events1 = [KiroEvent(type="content", content="hi")]

        with patch("kiro.mcp_tools.parse_kiro_stream", _events_for(resp1, {resp1: events1})):
            async for _ in web_search_event_source(resp1, None, None, 5.0):
                pass

        resp1.aclose.assert_not_called()


class TestContinuationPayloadAccumulation:
    """
    Regression tests: multi-round web_search must accumulate its payload.

    Each continuation round must build on the PREVIOUS round's payload, not on
    the original ``continuation.base_payload``. Rebuilding from the original
    drops every earlier round's toolResults, so the model asks for a search,
    receives a follow-up turn that no longer contains the result it just
    requested, and searches again — burning through ``max_iterations`` and
    ending on the raw-summary fallback instead of a synthesized answer.

    Observed symptom before the fix: 5x "Intercepted web_search tool call"
    followed by "iteration limit (5) reached, emitting raw summary".
    """

    @staticmethod
    def _results_marked(marker: str):
        """Build results carrying a unique, greppable snippet marker."""
        return {
            "results": [{
                "title": f"Title {marker}",
                "url": f"https://example.com/{marker}",
                "snippet": f"DATA-{marker}",
                "publishedDate": 1700000000000,
            }],
            "totalResults": 1,
            "query": marker,
        }

    @pytest.mark.asyncio
    async def test_three_rounds_accumulate_history_and_results(self):
        """
        Across 3 search rounds, every continuation payload must retain all
        previous rounds' tool results and grow its history by 2 entries per
        round (assistant tool_use + user toolResult).
        """
        # 4 responses: 3 that request a search, 1 that answers
        responses = [Mock(status_code=200) for _ in range(4)]
        for r in responses:
            r.aclose = AsyncMock()

        event_map = {
            responses[0]: [KiroEvent(type="tool_use", tool_use=_make_ws_tool(query="q1"))],
            responses[1]: [KiroEvent(type="tool_use", tool_use=_make_ws_tool(query="q2"))],
            responses[2]: [KiroEvent(type="tool_use", tool_use=_make_ws_tool(query="q3"))],
            responses[3]: [KiroEvent(type="content", content="FINAL ANSWER")],
        }

        sent_payloads = []

        async def send(payload):
            sent_payloads.append(payload)
            return responses[len(sent_payloads)]

        async def fake_mcp(query, auth):
            return f"srvtoolu_{query}", self._results_marked(query)

        cont = WebSearchContinuation(
            base_payload=_make_base_payload(),
            send_request=send,
            max_iterations=5,
        )

        with patch("kiro.mcp_tools.parse_kiro_stream", _events_for(None, event_map)), \
             patch("kiro.mcp_tools.call_kiro_mcp_api", new=AsyncMock(side_effect=fake_mcp)):
            collected = []
            async for ev in web_search_event_source(responses[0], cont, "auth", 5.0):
                collected.append(ev)

        assert len(sent_payloads) == 3, "expected one continuation request per search"

        # History grows by 2 per completed round
        histories = [
            len(p["conversationState"].get("history", [])) for p in sent_payloads
        ]
        assert histories == [2, 4, 6], f"history did not accumulate: {histories}"

        # Every earlier round's results must still be present
        round2 = json.dumps(sent_payloads[1])
        assert "DATA-q1" in round2

        round3 = json.dumps(sent_payloads[2])
        for marker in ("DATA-q1", "DATA-q2", "DATA-q3"):
            assert marker in round3, f"{marker} lost from final continuation payload"

        # The model's answer must be the visible output, not the raw fallback
        contents = [e.content or "" for e in collected if e.type == "content"]
        assert any("FINAL ANSWER" in c for c in contents)
        assert not any("<web_search>" in c for c in contents), \
            "should not degrade to the raw-summary fallback"

    @pytest.mark.asyncio
    async def test_base_payload_never_mutated_across_rounds(self):
        """
        The caller's base_payload is reused for retries, so accumulation must
        never mutate it.
        """
        base = _make_base_payload()
        base_snapshot = json.dumps(base, sort_keys=True)

        responses = [Mock(status_code=200) for _ in range(3)]
        for r in responses:
            r.aclose = AsyncMock()

        event_map = {
            responses[0]: [KiroEvent(type="tool_use", tool_use=_make_ws_tool(query="q1"))],
            responses[1]: [KiroEvent(type="tool_use", tool_use=_make_ws_tool(query="q2"))],
            responses[2]: [KiroEvent(type="content", content="done")],
        }

        sent = []

        async def send(payload):
            sent.append(payload)
            return responses[len(sent)]

        async def fake_mcp(query, auth):
            return f"srvtoolu_{query}", self._results_marked(query)

        cont = WebSearchContinuation(base_payload=base, send_request=send, max_iterations=5)

        with patch("kiro.mcp_tools.parse_kiro_stream", _events_for(None, event_map)), \
             patch("kiro.mcp_tools.call_kiro_mcp_api", new=AsyncMock(side_effect=fake_mcp)):
            async for _ in web_search_event_source(responses[0], cont, "auth", 5.0):
                pass

        assert json.dumps(base, sort_keys=True) == base_snapshot, \
            "base_payload was mutated during continuation"

    @pytest.mark.asyncio
    async def test_tools_preserved_in_every_round(self):
        """
        The tools definition must survive every round, otherwise the model
        cannot issue a follow-up search.
        """
        responses = [Mock(status_code=200) for _ in range(3)]
        for r in responses:
            r.aclose = AsyncMock()

        event_map = {
            responses[0]: [KiroEvent(type="tool_use", tool_use=_make_ws_tool(query="q1"))],
            responses[1]: [KiroEvent(type="tool_use", tool_use=_make_ws_tool(query="q2"))],
            responses[2]: [KiroEvent(type="content", content="done")],
        }

        sent = []

        async def send(payload):
            sent.append(payload)
            return responses[len(sent)]

        async def fake_mcp(query, auth):
            return f"srvtoolu_{query}", self._results_marked(query)

        cont = WebSearchContinuation(
            base_payload=_make_base_payload(with_tools=True),
            send_request=send,
            max_iterations=5,
        )

        with patch("kiro.mcp_tools.parse_kiro_stream", _events_for(None, event_map)), \
             patch("kiro.mcp_tools.call_kiro_mcp_api", new=AsyncMock(side_effect=fake_mcp)):
            async for _ in web_search_event_source(responses[0], cont, "auth", 5.0):
                pass

        for i, payload in enumerate(sent, 1):
            ctx = (payload["conversationState"]["currentMessage"]
                   ["userInputMessage"].get("userInputMessageContext", {}))
            assert ctx.get("tools"), f"tools missing from continuation round {i}"

    @pytest.mark.asyncio
    async def test_preexisting_history_is_preserved(self):
        """
        A conversation that already has history must keep it while accumulating
        new rounds on top.
        """
        responses = [Mock(status_code=200) for _ in range(3)]
        for r in responses:
            r.aclose = AsyncMock()

        event_map = {
            responses[0]: [KiroEvent(type="tool_use", tool_use=_make_ws_tool(query="q1"))],
            responses[1]: [KiroEvent(type="tool_use", tool_use=_make_ws_tool(query="q2"))],
            responses[2]: [KiroEvent(type="content", content="done")],
        }

        sent = []

        async def send(payload):
            sent.append(payload)
            return responses[len(sent)]

        async def fake_mcp(query, auth):
            return f"srvtoolu_{query}", self._results_marked(query)

        # base already carries 2 history entries
        cont = WebSearchContinuation(
            base_payload=_make_base_payload(with_history=True),
            send_request=send,
            max_iterations=5,
        )

        with patch("kiro.mcp_tools.parse_kiro_stream", _events_for(None, event_map)), \
             patch("kiro.mcp_tools.call_kiro_mcp_api", new=AsyncMock(side_effect=fake_mcp)):
            async for _ in web_search_event_source(responses[0], cont, "auth", 5.0):
                pass

        histories = [len(p["conversationState"]["history"]) for p in sent]
        assert histories == [4, 6], f"pre-existing history not preserved: {histories}"


# ==================================================================================================
# Tests for collect_stream_to_result with web_search continuation
# ==================================================================================================

class TestCollectStreamToResultContinuation:
    """Tests for collect_stream_to_result web_search integration."""

    @pytest.mark.asyncio
    async def test_collects_web_searches_when_continuation(self):
        """collect_stream_to_result records executed web searches."""
        from kiro.streaming_core import collect_stream_to_result

        resp1 = Mock()
        resp2 = Mock()
        resp2.status_code = 200
        events1 = [KiroEvent(type="tool_use", tool_use=_make_ws_tool())]
        events2 = [KiroEvent(type="content", content="answer")]

        async def send(payload):
            return resp2

        cont = WebSearchContinuation(base_payload=_make_base_payload(), send_request=send, max_iterations=5)

        mapping = {resp1: events1, resp2: events2}
        with patch("kiro.mcp_tools.parse_kiro_stream", _events_for(resp1, mapping)):
            with patch("kiro.mcp_tools.call_kiro_mcp_api", new=AsyncMock(return_value=("srvtoolu_1", _make_results()))):
                result = await collect_stream_to_result(
                    resp1, auth_manager="auth", web_search_continuation=cont
                )

        assert len(result.web_searches) == 1
        assert result.web_searches[0]["query"] == "paris weather"
        assert result.content == "answer"

    @pytest.mark.asyncio
    async def test_no_continuation_uses_raw_parse(self):
        """Without continuation, web_searches stays empty (raw parse path)."""
        from kiro.streaming_core import collect_stream_to_result

        resp = Mock()
        events = [KiroEvent(type="content", content="plain")]

        with patch("kiro.streaming_core.parse_kiro_stream", _events_for(resp, {resp: events})):
            result = await collect_stream_to_result(resp)

        assert result.web_searches == []
        assert result.content == "plain"

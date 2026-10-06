"""A mocked Messages API for tests, so no test calls the real API."""

import json

import httpx2

from utils.anthropic_client import AnthropicClient


def message(
    text='{"present": ["Insurance"]}', stop_reason="end_turn", stop_details=None, usage=None
):
    """A Messages API response body."""
    content = [{"type": "thinking", "thinking": "", "signature": "sig"}]
    if text is not None:
        content.append({"type": "text", "text": text})
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "stop_details": stop_details,
        "usage": usage
        or {
            "input_tokens": 1000,
            "output_tokens": 50,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


def error(status, kind, text="error"):
    return status, {"type": "error", "error": {"type": kind, "message": text}}


class MockAPI:
    """Answers each request with the next queued (status, body[, headers]) and records it."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.headers = []

    def __call__(self, request):
        self.requests.append(json.loads(request.content))
        self.headers.append(request.headers)
        status, body, *headers = self.replies.pop(0)
        return httpx2.Response(status, json=body, headers=headers[0] if headers else None)

    def client(self, **kwargs):
        transport = httpx2.MockTransport(self)
        return AnthropicClient(
            api_key="test", http_client=httpx2.Client(transport=transport), **kwargs
        )

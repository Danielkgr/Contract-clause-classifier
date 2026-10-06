"""Tests for the OpenAI comparison client, with the API mocked at the HTTP layer."""

import json

import httpx2
import pytest

from utils.llm_client import FatalLLMError
from utils.openai_client import OpenAIClient
from utils.prompts import system_prompt

TYPES = ["Governing Law", "Insurance"]


def completion(
    content='{"present": ["Insurance"]}', finish_reason="stop", refusal=None, usage=(1000, 20)
):
    """A chat completion response body."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content, "refusal": refusal},
            }
        ],
        "usage": {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": sum(usage),
        },
    }


class MockAPI:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, request):
        self.requests.append(json.loads(request.content))
        status, body = self.replies.pop(0)
        return httpx2.Response(status, json=body)

    def client(self, **kwargs):
        transport = httpx2.MockTransport(self)
        return OpenAIClient(
            api_key="test", http_client=httpx2.Client(transport=transport), **kwargs
        )


def test_request_sends_the_shared_prompt_and_asks_for_json():
    api = MockAPI((200, completion()))
    api.client().classify("The Supplier shall maintain insurance.", TYPES)
    [body] = api.requests
    assert body["model"] == "gpt-4o-mini"
    assert body["messages"][0] == {"role": "system", "content": system_prompt(TYPES)}
    assert "The Supplier shall maintain insurance." in body["messages"][1]["content"]
    assert body["response_format"] == {"type": "json_object"}
    assert "temperature" not in body


def test_answer_is_read_and_cost_comes_from_usage():
    api = MockAPI((200, completion(usage=(1_000_000, 0))))
    response = api.client().classify("text", TYPES)
    assert response.outcome == "ok"
    assert response.present == {"Insurance"}
    assert response.cost_usd == pytest.approx(0.15)


def test_a_length_stop_is_a_truncated_call():
    api = MockAPI((200, completion(content='{"present": [', finish_reason="length")))
    response = api.client().classify("text", TYPES)
    assert response.present is None
    assert response.outcome == "truncated"


def test_a_refusal_is_a_failed_call():
    api = MockAPI((200, completion(content=None, refusal="I can't help with that.")))
    response = api.client().classify("text", TYPES)
    assert response.present is None
    assert response.outcome == "refusal"


def test_a_rejected_key_stops_the_run():
    api = MockAPI(
        (401, {"error": {"message": "Incorrect API key", "type": "invalid_request_error"}})
    )
    with pytest.raises(FatalLLMError):
        api.client().classify("text", TYPES)


def test_a_temperature_is_sent_when_set():
    api = MockAPI((200, completion()))
    api.client(temperature=0.0).classify("text", TYPES)
    assert api.requests[0]["temperature"] == 0.0


def test_claude_is_never_reached_through_the_openai_client():
    with pytest.raises(ValueError, match="anthropic provider"):
        OpenAIClient("claude-opus-5-5", api_key="test")

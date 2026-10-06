"""Tests for the Claude client, with the Messages API mocked at the HTTP layer."""

import pytest
from mock_api import MockAPI, error, message

from utils.anthropic_client import AnthropicClient
from utils.llm_client import FatalLLMError

TYPES = ["Governing Law", "Insurance"]


def test_request_puts_definitions_in_a_cached_system_prompt_and_asks_for_json():
    api = MockAPI((200, message()))
    api.client().classify("The Supplier shall maintain insurance.", TYPES)
    [body] = api.requests
    assert body["model"] == "claude-opus-5-5"
    [system] = body["system"]
    assert system["cache_control"] == {"type": "ephemeral"}
    assert "- Governing Law: Which state or country's law governs the contract" in system["text"]
    assert "The Supplier shall maintain insurance." in body["messages"][0]["content"]
    assert body["output_config"]["effort"] == "low"
    answer_format = body["output_config"]["format"]
    assert answer_format["type"] == "json_schema"
    assert answer_format["schema"]["properties"]["present"]["items"]["enum"] == TYPES
    # Opus 5.5 rejects sampling settings and a thinking switch, and the harness
    # never asks for a fallback model
    for key in ("temperature", "top_p", "top_k", "thinking", "fallbacks"):
        assert key not in body
    assert "anthropic-beta" not in api.headers[0]


def test_answer_is_read_and_cost_comes_from_usage():
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 100_000,
        "cache_read_input_tokens": 1_000_000,
        "cache_creation_input_tokens": 0,
    }
    api = MockAPI((200, message(usage=usage)))
    response = api.client().classify("text", TYPES)
    assert response.outcome == "ok"
    assert response.present == {"Insurance"}
    # Opus 5.5 per million tokens: $4 input, $20 output, $0.20 cache reads
    assert response.cost_usd == pytest.approx(4.00 + 2.00 + 0.20)
    assert response.usage.cache_read_tokens == 1_000_000


def test_a_refusal_is_a_failed_call_not_an_absence():
    details = {"type": "refusal", "category": "cyber", "explanation": "Declined."}
    api = MockAPI((200, message(text=None, stop_reason="refusal", stop_details=details)))
    response = api.client().classify("text", TYPES)
    assert response.present is None
    assert response.outcome == "refusal"
    assert "cyber" in response.error


def test_a_refusal_without_details_is_still_recorded():
    api = MockAPI((200, message(text=None, stop_reason="refusal")))
    response = api.client().classify("text", TYPES)
    assert response.outcome == "refusal"
    assert response.error == "refused: no details given"


def test_a_truncated_answer_is_a_failed_call():
    api = MockAPI((200, message(text='{"present": ["Insu', stop_reason="max_tokens")))
    response = api.client().classify("text", TYPES)
    assert response.present is None
    assert response.outcome == "truncated"


def test_an_answer_that_is_not_the_json_asked_for_is_a_failed_call():
    api = MockAPI((200, message(text="Insurance appears in this excerpt.")))
    response = api.client().classify("text", TYPES)
    assert response.present is None
    assert response.outcome == "unparseable"


def test_a_rejected_request_fails_the_call_but_not_the_run():
    api = MockAPI(error(400, "invalid_request_error", "bad request"))
    response = api.client().classify("text", TYPES)
    assert response.present is None
    assert response.outcome == "error"
    assert "400" in response.error


def test_a_rejected_key_stops_the_run():
    api = MockAPI(error(401, "authentication_error", "invalid x-api-key"))
    with pytest.raises(FatalLLMError, match="credentials"):
        api.client().classify("text", TYPES)


def test_an_unknown_model_id_stops_the_run():
    api = MockAPI(error(404, "not_found_error", "model: claude-opus-5-5"))
    with pytest.raises(FatalLLMError, match="not available"):
        api.client().classify("text", TYPES)


def test_haiku_gets_no_effort_setting_and_may_take_a_temperature():
    api = MockAPI((200, message()))
    api.client(model="claude-haiku-4-5", temperature=0.0).classify("text", TYPES)
    [body] = api.requests
    assert "effort" not in body["output_config"]
    assert body["temperature"] == 0.0


def test_sonnet_is_priced_at_its_own_rate():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    api = MockAPI((200, message(usage=usage)))
    response = api.client(model="claude-sonnet-5-5").classify("text", TYPES)
    assert response.cost_usd == pytest.approx(2.00 + 10.00)


def test_temperature_is_refused_for_models_that_reject_it():
    for model in ("claude-opus-5-5", "claude-sonnet-5-5"):
        with pytest.raises(ValueError, match="rejects temperature"):
            AnthropicClient(model, api_key="test", temperature=0.0)


def test_an_unlisted_model_or_effort_is_refused_before_any_call():
    with pytest.raises(ValueError, match="Claude models"):
        AnthropicClient("claude-unknown", api_key="test")
    with pytest.raises(ValueError, match="effort"):
        AnthropicClient("claude-opus-5-5", api_key="test", effort="extreme")

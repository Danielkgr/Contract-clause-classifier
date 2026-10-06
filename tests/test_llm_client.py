"""Tests for the provider switch and the provider-neutral reading of answers."""

import pytest

from config import LLMConfig
from utils.llm_client import (
    Completion,
    LLMCallError,
    LLMClient,
    Usage,
    make_client,
)

TYPES = ["Governing Law", "Insurance"]


class FixedClient(LLMClient):
    """Returns one canned completion, or raises one error."""

    provider = "fixed"

    def __init__(self, completion=None, error=None, model="claude-opus-5-5"):
        super().__init__(model, max_tokens=100)
        self.completion, self.error = completion, error

    def send(self, excerpt, clause_types):
        if self.error:
            raise self.error
        return self.completion


def test_the_default_provider_is_claude_opus():
    client = make_client(LLMConfig(api_key="test"))
    assert (client.provider, client.model) == ("anthropic", "claude-opus-5-5")


def test_claude_models_are_selected_by_name():
    client = make_client(LLMConfig(model="claude-haiku-4-5", api_key="test"))
    assert client.model == "claude-haiku-4-5"


def test_the_openai_path_is_kept_for_comparison():
    client = make_client(LLMConfig(provider="openai", api_key="test"))
    assert (client.provider, client.model) == ("openai", "gpt-4o-mini")


def test_an_unknown_provider_is_refused():
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        make_client(LLMConfig(provider="litellm"))


def test_an_unexpected_stop_reason_is_a_failed_call():
    completion = Completion("", "tool_use", Usage(10, 5), latency_ms=3.0)
    response = FixedClient(completion).classify("text", TYPES)
    assert response.present is None
    assert response.outcome == "error"


def test_a_failed_call_records_its_time_and_costs_nothing():
    response = FixedClient(error=LLMCallError("request rejected with 400")).classify("text", TYPES)
    assert response.present is None
    assert response.outcome == "error"
    assert response.latency_ms >= 0
    assert response.cost_usd == 0.0


def test_an_unpriced_model_reports_no_cost(caplog):
    completion = Completion('{"present": []}', "end_turn", Usage(10, 5), latency_ms=3.0)
    response = FixedClient(completion, model="local-model").classify("text", TYPES)
    assert response.present == frozenset()
    assert response.cost_usd is None
    assert "No listed price for local-model" in caplog.text

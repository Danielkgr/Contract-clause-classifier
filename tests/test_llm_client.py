"""Tests for LLM response handling and cost units."""

import pytest

from config import config
from utils.llm_client import LLMClient


def _client(monkeypatch, result):
    client = LLMClient(provider="openai", model="gpt-3.5-turbo")
    monkeypatch.setattr(client, "use_litellm", False)
    monkeypatch.setattr(client, "_classify_openai", lambda prompt: result)
    return client


def test_prices_are_per_million_tokens():
    # Per-thousand prices would all sit below 0.1
    for model, (inp, out) in config.llm.COSTS.items():
        assert inp >= 0.1 and out >= inp, model


def test_cost_uses_prices_per_million(monkeypatch):
    client = _client(monkeypatch, {"text": "YES", "input_tokens": 1_000_000, "output_tokens": 0})
    response = client.classify_single("contract", "Governing Law")
    assert response.text == "YES"
    assert response.error is None
    assert response.cost_usd == pytest.approx(0.50)


def test_failed_call_is_flagged_not_read_as_no(monkeypatch):
    client = _client(
        monkeypatch,
        {"text": "", "input_tokens": 0, "output_tokens": 0, "error": "timeout"},
    )
    response = client.classify_single("contract", "Governing Law")
    assert response.error == "timeout"
    with pytest.raises(RuntimeError, match="timeout"):
        client.classify_clause_exists("contract", "Governing Law")


def test_provider_exception_becomes_an_error(monkeypatch):
    client = LLMClient(provider="openai", model="gpt-3.5-turbo")
    monkeypatch.setattr(client, "use_litellm", False)
    monkeypatch.setattr(client, "openai_client", None, raising=False)
    response = client.classify_single("contract", "Governing Law")
    assert response.error
    assert response.text == ""

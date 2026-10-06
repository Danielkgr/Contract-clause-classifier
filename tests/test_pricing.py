"""Tests for the price table and the cost of a call."""

import pytest

from utils.pricing import PRICES, cost_usd, price_for


def test_claude_list_prices_per_million_tokens():
    assert (PRICES["claude-opus-5-5"].input, PRICES["claude-opus-5-5"].output) == (4.00, 20.00)
    assert (PRICES["claude-sonnet-5-5"].input, PRICES["claude-sonnet-5-5"].output) == (2.00, 10.00)
    assert (PRICES["claude-haiku-4-5"].input, PRICES["claude-haiku-4-5"].output) == (1.00, 5.00)
    assert PRICES["claude-opus-5-5"].cache_read == 0.20
    assert PRICES["claude-sonnet-5-5"].cache_read == 0.20


def test_prices_are_per_million_tokens():
    # Per-thousand prices would all sit below 0.1
    for model, price in PRICES.items():
        assert price.input >= 0.1 and price.output >= price.input, model


def test_cached_input_is_billed_at_its_own_rates():
    price = price_for("claude-opus-5-5")
    assert cost_usd(price, cache_read_tokens=1_000_000) == pytest.approx(0.20)
    assert cost_usd(price, cache_write_tokens=1_000_000) == pytest.approx(5.00)


def test_cached_input_falls_back_to_the_input_rate_without_a_listed_rate():
    assert cost_usd(price_for("gpt-4o-mini"), cache_read_tokens=1_000_000) == pytest.approx(0.15)


def test_the_batches_api_halves_the_cost():
    price = price_for("claude-opus-5-5")
    assert cost_usd(price, 1_000_000, 1_000_000, batch=True) == pytest.approx(12.00)


def test_an_unlisted_model_is_not_priced():
    assert price_for("some-local-model") is None
    assert cost_usd(None, 1_000_000, 1_000_000) is None

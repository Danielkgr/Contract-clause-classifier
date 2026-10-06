"""Tests for the cost estimator, which makes no API call."""

import pytest

from utils.chunking import char_chunks
from utils.estimate import CHARS_PER_TOKEN, estimate, format_estimate
from utils.prompts import system_prompt, user_message

TYPES = ["Governing Law", "Insurance", "Audit Rights"]


def test_calls_and_tokens_follow_the_chunking():
    texts = ["x " * 15_000, "short contract"]  # 30,000 characters, then 14
    est = estimate(
        texts, TYPES, chunk_chars=24_000, chunk_overlap=1_000, output_tokens_per_call=100
    )
    assert (est.contracts, est.chunks) == (2, 3)
    assert est.characters == 30_014
    multi, single = est.modes["multi"], est.modes["single"]
    assert multi.calls == 3
    assert single.calls == 3 * len(TYPES)
    assert multi.output_tokens == 300
    # Each call sends one chunk, with the system prompt for the types asked about
    chunks = [t[s:e] for t in texts for s, e in char_chunks(t, 24_000, 1_000)]
    excerpts = sum(len(user_message(chunk)) for chunk in chunks)
    assert multi.input_tokens == (3 * len(system_prompt(TYPES)) + excerpts) // CHARS_PER_TOKEN
    single_chars = sum(3 * len(system_prompt([ct])) + excerpts for ct in TYPES)
    assert single.input_tokens == single_chars // CHARS_PER_TOKEN


def test_single_label_mode_costs_more_than_multi_label_mode():
    est = estimate(["word " * 20_000], TYPES, chunk_chars=24_000, chunk_overlap=1_000)
    assert est.cost("claude-opus-5-5", "single") > 2 * est.cost("claude-opus-5-5", "multi")


def test_cost_uses_list_prices_and_reports_unlisted_models_as_not_priced():
    est = estimate(["a" * 4_000_000], ["Insurance"], 24_000, 1_000, output_tokens_per_call=0)
    tokens = est.modes["multi"].input_tokens
    assert est.cost("claude-opus-5-5", "multi") == pytest.approx(tokens * 4.00 / 1_000_000)
    assert est.cost("some-local-model", "multi") is None


def test_the_report_states_its_assumptions():
    est = estimate(["word " * 1_000], TYPES, 24_000, 1_000)
    text = format_estimate(est, "test.txt", models=["claude-opus-5-5", "local-model"])
    assert "An estimate only.  No API call was made." in text
    assert f"characters divided by {CHARS_PER_TOKEN}" in text
    assert "| `local-model` | not priced |" in text
    assert "| Single-label | up to " in text

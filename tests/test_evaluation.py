"""Tests for contract-level evaluation of both arms, with stub models."""

import pytest

from utils.data_loader import ContractData
from utils.evaluation import evaluate_fine_tuned, evaluate_zero_shot
from utils.llm_client import FatalLLMError, LLMResponse, Usage

FILLER = "The parties agree as follows. " * 2000  # about 60,000 characters
CHUNKING = {"chunk_chars": 10_000, "chunk_overlap": 500}


def _contract(cid, clauses, tail="", head=""):
    return ContractData(contract_id=cid, text=head + FILLER + tail, clauses=clauses)


class StubLLM:
    """Finds a clause type when the excerpt holds its marker, such as [Insurance].

    A call fails when it asks about a type in fail_for, or when the excerpt
    holds the marker [FAIL].
    """

    provider, model = "stub", "stub-model"

    def __init__(self, fail_for=()):
        self.fail_for = set(fail_for)
        self.calls = []

    def classify(self, excerpt, clause_types):
        self.calls.append(tuple(clause_types))
        if self.fail_for & set(clause_types) or "[FAIL]" in excerpt:
            return LLMResponse(None, "error", error="timeout", latency_ms=5.0, cost_usd=0.0)
        present = frozenset(ct for ct in clause_types if f"[{ct}]" in excerpt)
        return LLMResponse(present, "ok", usage=Usage(100, 1), latency_ms=10.0, cost_usd=0.001)


@pytest.mark.parametrize("mode", ["multi", "single"])
def test_zero_shot_reads_past_the_opening_of_the_contract(mode):
    contracts = [
        _contract("a", {"Governing Law": True}, tail="[Governing Law] Victoria."),
        _contract("b", {"Governing Law": False}),
    ]
    result = evaluate_zero_shot(StubLLM(), contracts, ["Governing Law"], mode=mode, **CHUNKING)
    assert result.predictions["Governing Law"] == [1, 0]
    assert result.metrics["Governing Law"].f1 == 1.0


def test_multi_label_asks_once_per_chunk_about_every_clause_type():
    client = StubLLM()
    types = ["Governing Law", "Insurance", "Exclusivity"]
    contract = _contract("a", {}, head="[Insurance] cover. ", tail="[Governing Law] Victoria.")
    result = evaluate_zero_shot(client, [contract], types, mode="multi", **CHUNKING)
    assert result.calls == len(client.calls) > 1
    assert all(asked == tuple(types) for asked in client.calls)
    # The contract's answer is the union over its chunks
    assert result.contracts[0].predictions == {"Governing Law": 1, "Insurance": 1, "Exclusivity": 0}


def test_multi_label_needs_fewer_calls_than_single_label():
    types = ["Governing Law", "Insurance", "Exclusivity"]
    contract = _contract("a", {})
    multi = evaluate_zero_shot(StubLLM(), [contract], types, mode="multi", **CHUNKING)
    single = evaluate_zero_shot(StubLLM(), [contract], types, mode="single", **CHUNKING)
    assert single.calls == multi.calls * len(types)


def test_single_label_stops_at_the_first_chunk_with_the_clause():
    client = StubLLM()
    contract = _contract("a", {"Insurance": True}, head="[Insurance] cover. ")
    evaluate_zero_shot(client, [contract], ["Insurance"], mode="single", **CHUNKING)
    assert len(client.calls) == 1


@pytest.mark.parametrize("mode", ["multi", "single"])
def test_a_failed_chunk_makes_a_clause_unknown_not_absent(mode):
    contract = _contract(
        "a", {"Insurance": True, "Exclusivity": False}, head="[FAIL] ", tail="[Insurance] cover."
    )
    result = evaluate_zero_shot(
        StubLLM(), [contract], ["Insurance", "Exclusivity"], mode=mode, **CHUNKING
    )
    # Found in a later chunk despite the failed one, so present
    assert result.contracts[0].predictions["Insurance"] == 1
    # Not found, but the failed chunk might have held it, so unknown
    assert result.contracts[0].predictions["Exclusivity"] is None
    assert result.failures == {"Insurance": 0, "Exclusivity": 1}
    assert "Exclusivity" not in result.metrics
    assert result.outcomes["error"] >= 1


@pytest.mark.parametrize("mode", ["multi", "single"])
def test_zero_shot_latency_and_cost_are_summed_per_contract(mode):
    contract = _contract("a", {"Insurance": False})
    result = evaluate_zero_shot(StubLLM(), [contract], ["Insurance"], mode=mode, **CHUNKING)
    calls = result.calls
    assert calls > 1
    assert result.stats.documents == 1
    assert result.stats.avg_latency_ms == pytest.approx(10.0 * calls)
    assert result.stats.avg_cost_usd == pytest.approx(0.001 * calls)
    assert result.stats.input_tokens == 100 * calls
    assert result.settings["mode"] == mode


def test_failed_calls_are_left_out_and_counted():
    contracts = [_contract("a", {"Insurance": True, "Exclusivity": False})]
    result = evaluate_zero_shot(
        StubLLM(fail_for={"Insurance"}),
        contracts,
        ["Insurance", "Exclusivity"],
        mode="single",
        **CHUNKING,
    )
    assert result.failures == {"Insurance": 1, "Exclusivity": 0}
    assert "Insurance" not in result.metrics
    assert result.predictions["Exclusivity"] == [0]


@pytest.mark.parametrize("mode", ["multi", "single"])
def test_concurrency_does_not_change_the_results_or_their_order(mode):
    types = ["Governing Law", "Insurance"]
    contracts = [
        _contract(f"c{i}", {}, tail="[Insurance] cover." if i % 2 else "") for i in range(6)
    ]
    one = evaluate_zero_shot(StubLLM(), contracts, types, mode=mode, **CHUNKING)
    many = evaluate_zero_shot(StubLLM(), contracts, types, mode=mode, concurrency=4, **CHUNKING)
    assert [c.contract_id for c in many.contracts] == [f"c{i}" for i in range(6)]
    assert [c.predictions for c in many.contracts] == [c.predictions for c in one.contracts]
    assert many.stats.avg_latency_ms == one.stats.avg_latency_ms


def test_a_fatal_error_stops_the_run():
    class Rejected(StubLLM):
        def classify(self, excerpt, clause_types):
            raise FatalLLMError("Anthropic rejected the credentials")

    with pytest.raises(FatalLLMError):
        evaluate_zero_shot(
            Rejected(), [_contract("a", {})], ["Insurance"], concurrency=2, **CHUNKING
        )


def test_cached_calls_and_retries_are_counted():
    class Cached(StubLLM):
        def classify(self, excerpt, clause_types):
            response = super().classify(excerpt, clause_types)
            response.cached, response.retries = True, 1
            return response

    result = evaluate_zero_shot(Cached(), [_contract("a", {})], ["Insurance"], **CHUNKING)
    assert result.cached_calls == result.calls == result.retries


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode"):
        evaluate_zero_shot(StubLLM(), [], ["Insurance"], mode="batch")


class StubClassifier:
    clause_types = ["Governing Law", "Insurance"]

    def __init__(self, probs, ms=40.0):
        self.probs = iter(probs)
        self.ms = ms

    def predict_contract(self, text):
        return next(self.probs), self.ms


def test_fine_tuned_is_unpriced_without_a_rate():
    contracts = [_contract("a", {"Governing Law": True}), _contract("b", {"Governing Law": False})]
    clf = StubClassifier(
        [{"Governing Law": 0.9, "Insurance": 0.1}, {"Governing Law": 0.2, "Insurance": 0.1}]
    )
    result = evaluate_fine_tuned(clf, contracts, ["Governing Law"], threshold=0.5)
    assert result.predictions["Governing Law"] == [1, 0]
    assert result.stats.avg_latency_ms == 40.0
    assert result.stats.total_cost_usd is None
    assert result.stats.avg_cost_usd is None


def test_fine_tuned_cost_comes_from_measured_time():
    contracts = [_contract("a", {"Insurance": False})]
    clf = StubClassifier([{"Governing Law": 0.1, "Insurance": 0.1}], ms=3_600_000)
    result = evaluate_fine_tuned(clf, contracts, ["Insurance"], cost_per_hour=2.0)
    assert result.stats.avg_cost_usd == pytest.approx(2.0)


def test_fine_tuned_rejects_clause_types_the_model_lacks():
    with pytest.raises(ValueError, match="Audit Rights"):
        evaluate_fine_tuned(StubClassifier([]), [], ["Audit Rights"])

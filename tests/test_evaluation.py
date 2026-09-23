"""Tests for contract-level evaluation of both arms, with stub models."""

import pytest

from utils.data_loader import ContractData
from utils.evaluation import evaluate_fine_tuned, evaluate_zero_shot
from utils.llm_client import LLMResponse

FILLER = "The parties agree as follows. " * 2000  # about 60,000 characters


def _contract(cid, clauses, tail=""):
    return ContractData(contract_id=cid, text=FILLER + tail, clauses=clauses)


class StubLLM:
    """Says YES when the chunk holds the marker for the clause type."""

    def __init__(self, fail_for=()):
        self.fail_for = set(fail_for)
        self.calls = []

    def classify_single(self, text, clause_type):
        self.calls.append((clause_type, len(text)))
        if clause_type in self.fail_for:
            return LLMResponse("", 0, 0, 5.0, 0.0, error="timeout")
        answer = "YES" if f"[{clause_type}]" in text else "NO"
        return LLMResponse(answer, 100, 1, 10.0, 0.001)


def test_zero_shot_reads_past_the_opening_of_the_contract():
    contracts = [
        _contract("a", {"Governing Law": True}, tail="[Governing Law] Victoria."),
        _contract("b", {"Governing Law": False}),
    ]
    result = evaluate_zero_shot(StubLLM(), contracts, ["Governing Law"],
                                chunk_chars=10_000, chunk_overlap=500)
    assert result.predictions["Governing Law"] == [1, 0]
    assert result.metrics["Governing Law"].f1 == 1.0


def test_zero_shot_stops_at_the_first_yes():
    client = StubLLM()
    contract = ContractData("a", "[Insurance] cover. " + FILLER, {"Insurance": True})
    evaluate_zero_shot(client, [contract], ["Insurance"], chunk_chars=10_000, chunk_overlap=500)
    assert len(client.calls) == 1


def test_zero_shot_latency_and_cost_are_summed_per_contract():
    contract = _contract("a", {"Insurance": False})
    result = evaluate_zero_shot(StubLLM(), [contract], ["Insurance"],
                                chunk_chars=10_000, chunk_overlap=500)
    calls = result.calls
    assert calls > 1
    assert result.stats.documents == 1
    assert result.stats.avg_latency_ms == pytest.approx(10.0 * calls)
    assert result.stats.avg_cost_usd == pytest.approx(0.001 * calls)
    assert result.stats.input_tokens == 100 * calls


def test_failed_calls_are_left_out_and_counted():
    contracts = [_contract("a", {"Insurance": True, "Exclusivity": False})]
    result = evaluate_zero_shot(StubLLM(fail_for={"Insurance"}), contracts,
                                ["Insurance", "Exclusivity"], chunk_chars=10_000, chunk_overlap=500)
    assert result.failures == {"Insurance": 1, "Exclusivity": 0}
    assert "Insurance" not in result.metrics
    assert result.predictions["Exclusivity"] == [0]


class StubClassifier:
    clause_types = ["Governing Law", "Insurance"]

    def __init__(self, probs, ms=40.0):
        self.probs = iter(probs)
        self.ms = ms

    def predict_contract(self, text):
        return next(self.probs), self.ms


def test_fine_tuned_is_unpriced_without_a_rate():
    contracts = [_contract("a", {"Governing Law": True}), _contract("b", {"Governing Law": False})]
    clf = StubClassifier([{"Governing Law": 0.9, "Insurance": 0.1},
                          {"Governing Law": 0.2, "Insurance": 0.1}])
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

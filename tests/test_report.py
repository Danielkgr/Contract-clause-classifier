"""Tests for the comparison report built from saved arm results."""

import pytest

from utils.evaluation import ArmResult, ContractResult
from utils.report import align, models_table, notes, write_report

TYPES = ["Insurance", "Governing Law"]


def arm(kind, ids, predictions=None, cost=0.01, types=TYPES, **extra):
    contracts = [
        ContractResult(
            cid,
            labels={ct: 1 for ct in types},
            predictions={ct: 1 for ct in types} if predictions is None else predictions,
            latency_ms=100.0,
            cost_usd=cost,
            calls=2 if kind == "zero_shot" else 0,
            **extra,
        )
        for cid in ids
    ]
    model = "anthropic/claude-opus-5-5" if kind == "zero_shot" else "roberta-base"
    return ArmResult(kind, model, list(types), contracts, settings={"mode": "multi"})


def test_arms_are_scored_on_the_contracts_and_clause_types_they_share():
    zero_shot = arm("zero_shot", ["a", "b", "c"])
    fine_tuned = arm("fine_tuned", ["b", "c", "d"], types=["Insurance"], cost=None)
    aligned, found = align([zero_shot, fine_tuned])
    assert [c.contract_id for c in aligned[0].contracts] == ["b", "c"]
    assert [c.contract_id for c in aligned[1].contracts] == ["b", "c"]
    assert aligned[0].clause_types == ["Insurance"]
    assert any("2 contracts" in note for note in found)
    assert any("Governing Law" in note for note in found)


def test_a_report_with_only_the_zero_shot_arm_is_written(tmp_path):
    summary = write_report([arm("zero_shot", ["a", "b"])], str(tmp_path))
    assert "Only one arm has results" in summary
    assert (tmp_path / "comparison_report.md").exists()


def test_the_models_table_marks_what_does_not_apply_or_is_not_priced():
    table = models_table([arm("zero_shot", ["a"]), arm("fine_tuned", ["a"], cost=None)])
    assert "| LLM calls | 2 | none |" in table
    assert "| Cost per contract | $0.0100 | not priced |" in table


def test_notes_explain_failures_refusals_cached_calls_and_missing_prices():
    unknown = {"Insurance": None, "Governing Law": 1}
    zero_shot = arm(
        "zero_shot", ["a"], predictions=unknown, outcomes={"ok": 1, "refusal": 1}, cached_calls=2
    )
    found = " ".join(notes([zero_shot, arm("fine_tuned", ["a"], cost=None)]))
    assert "{'Insurance': 1}" in found
    assert "1 refused" in found
    assert "2 of 2 calls were answered from the response cache" in found
    assert "is not priced" in found


def test_saved_results_load_back_unchanged(tmp_path):
    original = arm("zero_shot", ["a", "b"], predictions={"Insurance": None, "Governing Law": 0})
    path = tmp_path / "zero_shot.json"
    original.save(str(path))
    assert ArmResult.load(str(path)) == original


def test_results_in_an_unknown_layout_are_refused():
    with pytest.raises(ValueError, match="Unknown results format"):
        ArmResult.from_dict({"format": 99})

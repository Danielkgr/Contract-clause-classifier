"""End-to-end check of the windowed classifier with a tiny model.

Needs torch and transformers, and downloads a small test model from Hugging
Face, so it runs only when RUN_SMOKE=1.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(os.getenv("RUN_SMOKE") != "1", reason="set RUN_SMOKE=1 to run")

TINY_MODEL = os.getenv("SMOKE_MODEL", "hf-internal-testing/tiny-random-RobertaForSequenceClassification")


def _contract(cid, present):
    from utils.data_loader import ContractData
    filler = "The parties agree to the terms set out below. " * 400
    clause = "This Agreement is governed by the laws of the State of New York."
    text = filler + clause + " " + filler if present else filler * 2
    spans = {"Governing Law": [(len(filler), len(filler) + len(clause))] if present else [],
             "Insurance": []}
    return ContractData(cid, text, {"Governing Law": present, "Insurance": False}, spans=spans)


def test_train_save_load_and_predict_whole_contracts(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from config import config
    from utils.classifier import FineTunedClassifier
    from utils.evaluation import evaluate_fine_tuned

    monkeypatch.setattr(config.training, "num_epochs", 1)
    monkeypatch.setattr(config.training, "batch_size", 4)
    monkeypatch.setattr(config.training, "warmup_steps", 0)
    monkeypatch.setattr(config.training, "max_length", 128)
    monkeypatch.setattr(config.training, "window_stride", 32)

    clause_types = ["Governing Law", "Insurance"]
    clf = FineTunedClassifier(clause_types, model_name=TINY_MODEL)

    # Windows cover the whole contract, not a prefix
    contract = _contract("a", True)
    _, _, spans = clf.windows(contract.text)
    assert len(spans) > 10
    assert spans[-1][1] >= len(contract.text.rstrip()) - 1

    result = clf.train([_contract("t1", True), _contract("t2", False), _contract("t3", True)],
                       [_contract("v1", True)], output_dir=str(tmp_path / "model"))
    assert result.train_windows > 0 and result.val_windows > 0

    loaded = FineTunedClassifier(model_name=str(tmp_path / "model"))
    loaded.load(str(tmp_path / "model"))
    assert loaded.clause_types == clause_types

    arm = evaluate_fine_tuned(loaded, [_contract("x", True), _contract("y", False)], clause_types)
    assert arm.stats.documents == 2
    assert arm.stats.min_latency_ms > 0
    assert arm.stats.avg_cost_usd is None
    assert set(arm.metrics) == set(clause_types)

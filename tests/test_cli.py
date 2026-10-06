"""Tests for the command line: argument handling and each command end to end, with no network."""

import copy
import json
import sys
import types

import httpx2
import pytest

import compare_classifiers as cli
from config import config
from mock_api import MockAPI, message
from utils.data_loader import split_of
from utils.evaluation import ArmResult, ContractResult

TYPES = ["Insurance", "Governing Law"]
KEY = "sk-ant-api03-" + "x" * 80 + "WXYZ"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Each test gets its own copy of the settings, empty output and cache folders, and no key."""
    for name in ("llm", "training", "data", "paths"):
        monkeypatch.setattr(config, name, copy.deepcopy(getattr(config, name)))
    config.paths.outputs_dir = str(tmp_path / "outputs")
    config.llm.cache_dir = str(tmp_path / "cache")
    config.llm.api_key = None
    config.data.cuad_path = None
    for var in ("LLM_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def cuad(tmp_path):
    """A small CUAD file with three test contracts, one of which has an insurance clause."""
    titles = [t for t in (f"contract-{i}" for i in range(500)) if split_of(t) == "test"][:3]
    data = []
    for n, title in enumerate(titles):
        text = "The parties agree as follows. " + ("Each party shall insure. " if n == 0 else "")
        qas = []
        for category in TYPES:
            answer = "Each party shall insure." if category == "Insurance" and n == 0 else None
            answers = [{"text": answer, "answer_start": text.index(answer)}] if answer else []
            qas.append(
                {"id": f"{title}__{category}", "answers": answers, "is_impossible": not answers}
            )
        data.append({"title": title, "paragraphs": [{"context": text, "qas": qas}]})
    path = tmp_path / "CUAD_v1.json"
    path.write_text(json.dumps({"version": "aok_v1.0", "data": data}))
    return str(path), titles


def parse(*argv):
    return cli.build_parser().parse_args(list(argv))


def test_each_command_parses_with_its_defaults():
    assert parse("zero-shot").provider is None
    assert parse("zero-shot").no_cache is False
    assert parse("estimate").output_tokens == 200
    assert parse("compare").arm_files == []
    args = parse("fine-tune")
    assert not args.skip_training and not args.skip_evaluation


def test_a_command_is_required(capsys):
    with pytest.raises(SystemExit):
        parse()


def test_clause_types_are_checked_and_spelled_as_cuad_spells_them(capsys):
    assert parse("show-config", "--clause-types", "governing law", "INSURANCE").clause_types == [
        "Governing Law",
        "Insurance",
    ]
    with pytest.raises(SystemExit):
        parse("show-config", "--clause-types", "Arbitration")
    assert "not a CUAD category" in capsys.readouterr().err


def test_choices_and_counts_are_validated(capsys):
    for argv in (
        ("zero-shot", "--mode", "batch"),
        ("zero-shot", "--provider", "litellm"),
        ("zero-shot", "--effort", "extreme"),
        ("zero-shot", "--max-contracts", "0"),
        ("fine-tune", "--skip-training", "--skip-evaluation"),
    ):
        with pytest.raises(SystemExit):
            parse(*argv)


def test_flags_override_the_settings():
    cli.apply_llm(parse("zero-shot", "--model", "claude-haiku-4-5", "--mode", "single"))
    assert (config.llm.resolved_model, config.llm.mode) == ("claude-haiku-4-5", "single")
    cli.apply_training(parse("fine-tune", "--epochs", "1", "--learning-rate", "3e-5"))
    assert (config.training.num_epochs, config.training.learning_rate) == (1, 3e-5)


def test_show_config_masks_the_key(monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    assert cli.main(["show-config"]) == 0
    out = capsys.readouterr().out
    assert "...WXYZ from ANTHROPIC_API_KEY" in out
    assert KEY[:-4] not in out
    assert "claude-opus-5-5" in out


def test_estimate_needs_no_key(tmp_path, capsys):
    contract = tmp_path / "contract.txt"
    contract.write_text("The Supplier shall maintain insurance. " * 2000)
    assert cli.main(["estimate", "--text-file", str(contract)]) == 0
    out = capsys.readouterr().out
    assert "No API call was made" in out
    assert "| Source | contract.txt |" in out


def test_estimate_counts_the_cuad_test_split(cuad, capsys):
    path, titles = cuad
    assert cli.main(["estimate", "--cuad", path, "--clause-types", *TYPES]) == 0
    out = capsys.readouterr().out
    assert "| Source | CUAD test split |" in out
    assert f"| Contracts | {len(titles)} |" in out


def test_zero_shot_without_a_key_stops_before_any_call(cuad, caplog):
    assert cli.main(["zero-shot", "--cuad", cuad[0]]) == 2
    assert "No API key for anthropic" in caplog.text


def test_zero_shot_then_compare_end_to_end(cuad, monkeypatch, tmp_path, capsys):
    path, titles = cuad
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    api = MockAPI(*[(200, message('{"present": ["Insurance"]}'))] * len(titles))
    real = cli.make_client

    def mocked(settings, use_cache=True):
        transport = httpx2.MockTransport(api)
        return real(settings, http_client=httpx2.Client(transport=transport), use_cache=use_cache)

    monkeypatch.setattr(cli, "make_client", mocked)
    argv = ["zero-shot", "--cuad", path, "--clause-types", *TYPES]
    assert cli.main(argv) == 0
    saved = tmp_path / "outputs" / "zero_shot-claude-opus-5-5-multi.json"
    arm = ArmResult.load(str(saved))
    assert [c.contract_id for c in arm.contracts] == titles
    assert arm.predictions["Insurance"] == [1, 1, 1]
    assert arm.settings["provider"] == "anthropic"
    assert len(api.requests) == len(titles)

    # A re-run reads every answer from the cache and makes no request
    assert cli.main(argv) == 0
    assert len(api.requests) == len(titles)
    assert ArmResult.load(str(saved)).cached_calls == len(titles)

    fine_tuned = ArmResult(
        "fine_tuned",
        "roberta-base",
        TYPES,
        [
            ContractResult(
                t, {"Insurance": int(i == 0), "Governing Law": 0}, {ct: 0 for ct in TYPES}, 40.0
            )
            for i, t in enumerate(titles)
        ],
    )
    fine_tuned.save(str(tmp_path / "outputs" / "fine_tuned-roberta-base.json"))
    capsys.readouterr()
    assert cli.main(["compare"]) == 0
    out = capsys.readouterr().out
    assert "Zero-shot claude-opus-5-5, multi-label" in out and "Fine-tuned roberta-base" in out
    for name in (
        "comparison_metrics.csv",
        "summary.md",
        "comparison_report.md",
        "comparison_plot.png",
    ):
        assert (tmp_path / "outputs" / name).exists(), name


def test_compare_without_results_says_what_to_run(caplog):
    assert cli.main(["compare"]) == 2
    assert "Run zero-shot or fine-tune first" in caplog.text


def test_a_settings_error_stops_with_a_message(cuad, monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    argv = ["zero-shot", "--cuad", cuad[0], "--provider", "openai", "--model", "claude-opus-5-5"]
    assert cli.main(argv) == 2
    assert "Stopped: Claude models are reached through the anthropic provider" in caplog.text


class StubClassifier:
    """Stands in for the torch classifier: finds a clause type when its words appear."""

    def __init__(self, clause_types=None, model_name=None):
        self.clause_types = list(clause_types or TYPES)
        self.base_model_name = "stub-model"

    def train(self, train, validation, output_dir):
        return types.SimpleNamespace(training_time_seconds=1.0, train_windows=4, val_windows=2)

    def load(self, path):
        pass

    def predict_contract(self, text):
        return {
            ct: 0.9 if "insure" in text and ct == "Insurance" else 0.1 for ct in self.clause_types
        }, 5.0


def test_fine_tune_trains_scores_and_saves(cuad, monkeypatch, tmp_path):
    module = types.ModuleType("utils.classifier")
    module.FineTunedClassifier = StubClassifier
    monkeypatch.setitem(sys.modules, "utils.classifier", module)
    argv = ["fine-tune", "--cuad", cuad[0], "--clause-types", *TYPES, "--epochs", "1"]
    assert cli.main(argv) == 0
    arm = ArmResult.load(str(tmp_path / "outputs" / "fine_tuned-stub-model.json"))
    assert arm.predictions["Insurance"] == [1, 0, 0]
    assert arm.settings["epochs"] == 1
    assert arm.settings["train_windows"] == 4


def test_fine_tune_skip_training_needs_a_saved_model(monkeypatch, tmp_path, caplog):
    module = types.ModuleType("utils.classifier")
    module.FineTunedClassifier = StubClassifier
    monkeypatch.setitem(sys.modules, "utils.classifier", module)
    argv = ["fine-tune", "--skip-training", "--model-dir", str(tmp_path / "none")]
    assert cli.main(argv) == 2
    assert "No saved model" in caplog.text

"""Tests for CUAD parsing, splitting, and clause-type validation."""

import json
import os

import pytest

from config import CUAD_CATEGORIES, DEFAULT_CLAUSE_TYPES, DataConfig
from utils.data_loader import SPLITS, parse_cuad_json, split_of


def _qa(title, category, answered, start=0, text="clause text"):
    return {
        "id": f"{title}__{category}",
        "question": f'Highlight the parts (if any) of this contract related to "{category}"',
        "answers": [{"text": text, "answer_start": start}] if answered else [],
        "is_impossible": not answered,
    }


def _write_cuad(path, documents):
    """Write a small file in the CUAD SQuAD 2.0 layout."""
    data = []
    for title, paragraphs in documents:
        data.append({
            "title": title,
            "paragraphs": [
                {"context": context, "qas": [_qa(title, *answer) for answer in answers]}
                for context, answers in paragraphs
            ],
        })
    path.write_text(json.dumps({"version": "aok_v1.0", "data": data}))
    return str(path)


def _title_in(split):
    """First generated title that hashes into the given split."""
    return next(t for t in (f"contract-{i}" for i in range(1000)) if split_of(t) == split)


def test_presence_comes_from_answer_spans(tmp_path):
    title = _title_in("test")
    path = _write_cuad(tmp_path / "cuad.json", [
        (title, [("Full text.", [("Governing Law", True), ("Audit Rights", False)])]),
    ])
    [contract] = parse_cuad_json(path, "test", clause_types=["Governing Law", "Audit Rights"])
    assert contract.contract_id == title
    assert contract.text == "Full text."
    assert contract.clauses == {"Governing Law": True, "Audit Rights": False}


def test_clause_types_match_categories_ignoring_case(tmp_path):
    title = _title_in("train")
    path = _write_cuad(tmp_path / "cuad.json", [
        (title, [("Text.", [("Cap On Liability", True)])]),
    ])
    [contract] = parse_cuad_json(path, "train", clause_types=["Cap on Liability"])
    assert contract.clauses == {"Cap on Liability": True}


def test_unknown_clause_type_raises_instead_of_labelling_absent(tmp_path):
    title = _title_in("train")
    path = _write_cuad(tmp_path / "cuad.json", [
        (title, [("Text.", [("Governing Law", True)])]),
    ])
    with pytest.raises(ValueError, match="Arbitration"):
        parse_cuad_json(path, "train", clause_types=["Arbitration"])


def test_a_clause_in_any_paragraph_counts_as_present(tmp_path):
    title = _title_in("validation")
    path = _write_cuad(tmp_path / "cuad.json", [
        (title, [
            ("First part.", [("Insurance", False)]),
            ("Second part.", [("Insurance", True)]),
        ]),
    ])
    [contract] = parse_cuad_json(path, "validation", clause_types=["Insurance"])
    assert contract.clauses == {"Insurance": True}
    assert contract.text == "First part.\nSecond part."


def test_splits_are_disjoint_complete_and_stable(tmp_path):
    titles = [f"contract-{i}" for i in range(300)]
    path = _write_cuad(tmp_path / "cuad.json", [
        (t, [("Text.", [("Governing Law", True)])]) for t in titles
    ])
    by_split = {
        s: [c.contract_id for c in parse_cuad_json(path, s, clause_types=["Governing Law"])]
        for s in SPLITS
    }
    seen = [t for ids in by_split.values() for t in ids]
    assert sorted(seen) == sorted(titles)
    assert len(seen) == len(set(seen))
    assert len(by_split["train"]) > len(by_split["validation"])
    assert by_split["test"] == sorted(by_split["test"])


def test_max_samples_limits_contracts(tmp_path):
    titles = [f"contract-{i}" for i in range(100)]
    path = _write_cuad(tmp_path / "cuad.json", [
        (t, [("Text.", [("Governing Law", True)])]) for t in titles
    ])
    assert len(parse_cuad_json(path, "train", max_samples=5, clause_types=["Governing Law"])) == 5


def test_default_clause_types_are_real_cuad_categories():
    assert len(CUAD_CATEGORIES) == len(set(CUAD_CATEGORIES)) == 41
    assert set(DEFAULT_CLAUSE_TYPES) <= set(CUAD_CATEGORIES)
    assert len(DEFAULT_CLAUSE_TYPES) == 12


def test_data_config_adds_and_removes_clause_types():
    data = DataConfig()
    data.add_clause_type("Insurance")
    assert data.clause_types.count("Insurance") == 1
    data.add_clause_type("Source Code Escrow")
    assert data.clause_types[-1] == "Source Code Escrow"
    data.remove_clause_type("Source Code Escrow")
    assert "Source Code Escrow" not in data.clause_types


def test_spans_give_character_offsets_into_the_joined_text(tmp_path):
    title = _title_in("test")
    first = "Recitals.  Nothing to see."
    second = "Clause 9.  This Agreement is governed by the laws of Victoria."
    law = "governed by the laws of Victoria"
    path = _write_cuad(tmp_path / "cuad.json", [
        (title, [
            (first, [("Governing Law", False)]),
            (second, [("Governing Law", True, second.index(law), law)]),
        ]),
    ])
    [contract] = parse_cuad_json(path, "test", clause_types=["Governing Law"])
    [(start, end)] = contract.spans["Governing Law"]
    assert contract.text[start:end] == law


REAL_CUAD = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "CUAD_v1.json")


@pytest.mark.skipif(not os.path.exists(REAL_CUAD), reason="data/CUAD_v1.json not downloaded")
def test_real_cuad_splits_and_spans():
    sizes = {s: len(parse_cuad_json(REAL_CUAD, s, clause_types=DEFAULT_CLAUSE_TYPES)) for s in SPLITS}
    assert sizes == {"train": 401, "validation": 53, "test": 56}

    with open(REAL_CUAD) as fh:
        answers = {
            (d["title"], q["id"].rsplit("__", 1)[-1]): {a["text"] for a in q["answers"]}
            for d in json.load(fh)["data"] for p in d["paragraphs"] for q in p["qas"]
        }
    for contract in parse_cuad_json(REAL_CUAD, "test", clause_types=DEFAULT_CLAUSE_TYPES):
        for ct in DEFAULT_CLAUSE_TYPES:
            assert contract.clauses[ct] == bool(contract.spans[ct])
            for start, end in contract.spans[ct]:
                assert contract.text[start:end] in answers[(contract.contract_id, ct)]

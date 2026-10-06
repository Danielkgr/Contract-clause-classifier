"""Tests for the shared zero-shot prompt and the reading of its answers."""

import pytest

from config import DEFAULT_CLAUSE_TYPES
from utils.prompts import (
    CLAUSE_DEFINITIONS,
    answer_schema,
    parse_answer,
    system_prompt,
    user_message,
)

TYPES = ["Governing Law", "Insurance", "Audit Rights"]


def test_every_default_clause_type_has_a_definition():
    assert set(CLAUSE_DEFINITIONS) == set(DEFAULT_CLAUSE_TYPES)


def test_system_prompt_defines_each_clause_type_asked_about():
    prompt = system_prompt(TYPES)
    for clause_type in TYPES:
        assert f"- {clause_type}: {CLAUSE_DEFINITIONS[clause_type]}" in prompt
    assert "Exclusivity" not in prompt


def test_a_clause_type_without_a_definition_is_listed_by_name():
    assert "\n- Source Code Escrow\n" in system_prompt(["Insurance", "Source Code Escrow"])


def test_the_system_prompt_does_not_depend_on_the_excerpt():
    assert system_prompt(TYPES) == system_prompt(list(TYPES))
    assert "excerpt text" in user_message("excerpt text")


def test_schema_limits_answers_to_the_clause_types_asked_about():
    schema = answer_schema(TYPES)
    assert schema["properties"]["present"]["items"]["enum"] == TYPES
    assert schema["required"] == ["present"]
    assert schema["additionalProperties"] is False


def test_answers_are_matched_ignoring_case_and_repeats():
    answer = '{"present": ["insurance ", "Insurance", "GOVERNING LAW"]}'
    assert parse_answer(answer, TYPES) == {"Insurance", "Governing Law"}
    assert parse_answer('{"present": []}', TYPES) == frozenset()


@pytest.mark.parametrize(
    "answer, message",
    [
        ("Insurance", "not JSON"),
        ('{"found": ["Insurance"]}', '"present"'),
        ('{"present": ["Exclusivity"]}', "not asked about"),
    ],
)
def test_malformed_answers_raise(answer, message):
    with pytest.raises(ValueError, match=message):
        parse_answer(answer, TYPES)

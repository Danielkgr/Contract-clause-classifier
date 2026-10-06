"""
The zero-shot prompt, shared by every provider and both prompt modes.

The system prompt defines the clause types being asked about and asks for a
JSON answer listing those that appear in one excerpt of a contract.
Multi-label mode asks about every configured clause type in one call per
excerpt.  Single-label mode asks about one clause type per call.  The two
modes use the same template, so they differ only in how many clause types
each call covers.
"""

import json
from collections.abc import Sequence

# Recorded with every result and part of the response cache key.  Bump it
# whenever the wording or the answer format changes.
PROMPT_VERSION = "2"

MODES = ("multi", "single")

# Short definitions of the default clause types, as in the README table
CLAUSE_DEFINITIONS = {
    "Governing Law": "Which state or country's law governs the contract",
    "Anti-Assignment": "Consent or notice needed before the contract can be assigned",
    "Cap On Liability": "A cap on a party's liability for breach",
    "Uncapped Liability": "Liability left uncapped, for all breaches or a particular kind",
    "Audit Rights": "A right to audit the other party's books, records, or premises",
    "Termination For Convenience": "A right to terminate without cause",
    "Change Of Control": "Rights triggered by a change of control, merger, or asset sale",
    "Exclusivity": "An exclusive dealing commitment with the other party",
    "Non-Compete": "A restriction on competing with the other party",
    "Insurance": "A requirement to maintain insurance",
    "License Grant": "A licence granted by one party to the other",
    "Warranty Duration": "How long a warranty lasts",
}

_DEFINITIONS_BY_KEY = {name.casefold(): text for name, text in CLAUSE_DEFINITIONS.items()}


def definition(clause_type: str) -> str | None:
    """The definition of a clause type, ignoring case, or None if it has none."""
    return _DEFINITIONS_BY_KEY.get(clause_type.casefold())


def system_prompt(clause_types: Sequence[str]) -> str:
    """Instructions and definitions for the clause types asked about.

    Nothing in it varies between excerpts, so it can be cached.  A clause type
    without a definition is listed by name only.
    """
    lines = []
    for clause_type in clause_types:
        text = definition(clause_type)
        lines.append(f"- {clause_type}: {text}" if text else f"- {clause_type}")
    return (
        "You review commercial contracts.  Each message contains one excerpt from a longer "
        "contract.  Decide which of the clause types listed below appear in that excerpt.\n\n"
        "A clause type appears when the excerpt contains contract language of that type, "
        "whatever heading or wording it uses.  Judge only the text of the excerpt.  The rest of "
        "the contract is reviewed separately.\n\n"
        "Clause types:\n" + "\n".join(lines) + "\n\n"
        'Answer with a JSON object with one key, "present", whose value lists every clause type '
        "above that appears in the excerpt, spelled exactly as listed.  Use an empty list when "
        "none of them appears."
    )


def user_message(excerpt: str) -> str:
    """The excerpt, delimited so it cannot be mistaken for instructions."""
    return f"<excerpt>\n{excerpt}\n</excerpt>"


def answer_schema(clause_types: Sequence[str]) -> dict:
    """JSON schema for the answer: the clause types present, from those asked about."""
    return {
        "type": "object",
        "properties": {
            "present": {
                "type": "array",
                "items": {"type": "string", "enum": list(clause_types)},
            }
        },
        "required": ["present"],
        "additionalProperties": False,
    }


def parse_answer(text: str, clause_types: Sequence[str]) -> frozenset:
    """The clause types an answer lists as present.

    Names are matched to the clause types asked about, ignoring case and
    surrounding space, and a name listed twice counts once.

    Raises:
        ValueError: when the answer is not the expected JSON, or names a
            clause type that was not asked about
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"answer is not JSON: {text[:80]!r}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("present"), list):
        raise ValueError(f'answer has no "present" list: {text[:80]!r}')
    by_key = {clause_type.casefold(): clause_type for clause_type in clause_types}
    found, unknown = set(), []
    for item in data["present"]:
        key = item.strip().casefold() if isinstance(item, str) else None
        if key in by_key:
            found.add(by_key[key])
        else:
            unknown.append(item)
    if unknown:
        raise ValueError(f"answer names clause types that were not asked about: {unknown}")
    return frozenset(found)

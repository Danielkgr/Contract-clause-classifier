"""
Estimate the calls, tokens, and cost of a zero-shot run before making it.

Nothing here calls an API, so no key is needed.  Calls and characters are
counted exactly from the same chunking the run uses.  Tokens are estimated
as characters divided by CHARS_PER_TOKEN, and output tokens per call are an
assumption, so every token and cost figure is an estimate.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from utils.chunking import char_chunks
from utils.pricing import PRICES, cost_usd
from utils.prompts import system_prompt, user_message

CHARS_PER_TOKEN = 4
# A short JSON answer plus some thinking.  Check it against the usage a
# small run reports before relying on the output cost.
DEFAULT_OUTPUT_TOKENS = 200
DEFAULT_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5")


@dataclass
class ModeEstimate:
    """Calls and tokens for one prompt mode."""

    mode: str
    calls: int
    input_tokens: int
    output_tokens: int


@dataclass
class Estimate:
    """What a run over some contracts would send."""

    contracts: int
    characters: int
    chunks: int
    clause_types: list[str]
    chunk_chars: int
    chunk_overlap: int
    output_tokens_per_call: int
    modes: dict[str, ModeEstimate]

    def cost(self, model: str, mode: str) -> float | None:
        """Estimated cost in USD, or None when the model has no listed price."""
        m = self.modes[mode]
        return cost_usd(PRICES.get(model), m.input_tokens, m.output_tokens)


def estimate(
    texts: Sequence[str],
    clause_types: Sequence[str],
    chunk_chars: int,
    chunk_overlap: int,
    output_tokens_per_call: int = DEFAULT_OUTPUT_TOKENS,
) -> Estimate:
    """Count what a zero-shot run over texts would send, in both prompt modes.

    Single-label calls are an upper bound: the run skips a contract's
    remaining chunks for a clause type once one chunk has it.
    """
    chunks = [text[s:e] for text in texts for s, e in char_chunks(text, chunk_chars, chunk_overlap)]
    excerpt_chars = sum(len(user_message(chunk)) for chunk in chunks)

    multi_calls = len(chunks)
    multi_chars = multi_calls * len(system_prompt(clause_types)) + excerpt_chars
    single_calls = len(chunks) * len(clause_types)
    single_chars = sum(
        len(chunks) * len(system_prompt([ct])) + excerpt_chars for ct in clause_types
    )

    def mode(name, calls, chars):
        return ModeEstimate(name, calls, chars // CHARS_PER_TOKEN, calls * output_tokens_per_call)

    return Estimate(
        contracts=len(texts),
        characters=sum(len(text) for text in texts),
        chunks=len(chunks),
        clause_types=list(clause_types),
        chunk_chars=chunk_chars,
        chunk_overlap=chunk_overlap,
        output_tokens_per_call=output_tokens_per_call,
        modes={
            "multi": mode("multi", multi_calls, multi_chars),
            "single": mode("single", single_calls, single_chars),
        },
    )


def _millions(tokens: int) -> str:
    return f"{tokens / 1_000_000:.2f}M"


def _usd(value: float | None) -> str:
    return "not priced" if value is None else f"${value:,.2f}"


def format_estimate(est: Estimate, source: str, models: Sequence[str] = DEFAULT_MODELS) -> str:
    """The estimate as Markdown, with its assumptions spelled out."""
    multi, single = est.modes["multi"], est.modes["single"]
    lines = [
        "# Zero-shot cost estimate",
        "",
        "An estimate only.  No API call was made.",
        "",
        "| Input | Value |",
        "|---|---:|",
        f"| Source | {source} |",
        f"| Contracts | {est.contracts:,} |",
        f"| Characters | {est.characters:,} |",
        f"| Chunks of up to {est.chunk_chars:,} characters, overlapping by {est.chunk_overlap:,} "
        f"| {est.chunks:,} |",
        f"| Clause types | {len(est.clause_types)} |",
        "",
        "| Prompt mode | Calls | Input tokens | Output tokens |",
        "|---|---:|---:|---:|",
        f"| Multi-label | {multi.calls:,} | {_millions(multi.input_tokens)} "
        f"| {_millions(multi.output_tokens)} |",
        f"| Single-label | up to {single.calls:,} | up to {_millions(single.input_tokens)} "
        f"| up to {_millions(single.output_tokens)} |",
        "",
        "| Model | Multi-label | Single-label |",
        "|---|---:|---:|",
    ]
    for model in models:
        lines.append(
            f"| `{model}` | {_usd(est.cost(model, 'multi'))} "
            f"| up to {_usd(est.cost(model, 'single'))} |"
        )
    lines += [
        "",
        "Assumptions:",
        "",
        f"- Tokens are characters divided by {CHARS_PER_TOKEN}.  Real counts depend on the "
        "model's tokenizer.",
        f"- Each call returns {est.output_tokens_per_call} output tokens, thinking included.",
        "- No prompt-cache discount, and list prices without the Batches API discount.",
        "- Single-label figures are upper bounds, because a run stops asking about a clause type "
        "once one chunk of the contract has it.",
    ]
    return "\n".join(lines) + "\n"

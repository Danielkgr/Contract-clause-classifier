"""
Contract-level evaluation for both arms.

Each arm answers one question per contract and clause type, whether the
clause appears anywhere in the contract, so both are scored against the same
CUAD labels.  Latency and cost are measured per contract.
"""

import dataclasses
import json
import os
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from config import config
from utils.chunking import char_chunks
from utils.metrics import (
    ClassificationMetrics,
    InferenceStats,
    calculate_metrics,
    summarise_documents,
)
from utils.prompts import MODES, PROMPT_VERSION


@dataclass
class ContractResult:
    """One contract's labels, predictions, and measured cost."""

    contract_id: str
    labels: dict[str, int]
    # 1 present, 0 absent, or None when a failed call left the answer unknown
    predictions: dict[str, int | None]
    latency_ms: float = 0.0
    cost_usd: float | None = None  # None when not priced
    calls: int = 0
    cached_calls: int = 0  # answered from the response cache, with no API call
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)  # call outcome -> count


# Version of the saved results layout, checked when results are loaded
RESULTS_FORMAT = 1


@dataclass
class ArmResult:
    """What one arm produced on the test contracts."""

    arm: str  # "zero_shot" or "fine_tuned"
    model: str
    clause_types: list[str]
    contracts: list[ContractResult]
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def predictions(self) -> dict[str, list[int]]:
        """Decisions per clause type, leaving out those that failed calls left unknown."""
        return {
            ct: [c.predictions[ct] for c in self.contracts if c.predictions.get(ct) is not None]
            for ct in self.clause_types
        }

    @property
    def labels(self) -> dict[str, list[int]]:
        """Labels per clause type, aligned with predictions."""
        return {
            ct: [c.labels[ct] for c in self.contracts if c.predictions.get(ct) is not None]
            for ct in self.clause_types
        }

    @property
    def failures(self) -> dict[str, int]:
        """Contracts per clause type left out because failed calls left the answer unknown."""
        return {
            ct: sum(1 for c in self.contracts if c.predictions.get(ct) is None)
            for ct in self.clause_types
        }

    @property
    def metrics(self) -> dict[str, ClassificationMetrics]:
        labels, predictions = self.labels, self.predictions
        return {
            ct: calculate_metrics(labels[ct], predictions[ct])
            for ct in self.clause_types
            if predictions[ct]
        }

    @property
    def stats(self) -> InferenceStats:
        costs = [c.cost_usd for c in self.contracts]
        return summarise_documents(
            [c.latency_ms for c in self.contracts],
            None if any(cost is None for cost in costs) else costs,
            sum(
                c.input_tokens + c.cache_read_tokens + c.cache_write_tokens for c in self.contracts
            ),
            sum(c.output_tokens for c in self.contracts),
        )

    @property
    def calls(self) -> int:
        return sum(c.calls for c in self.contracts)

    @property
    def cached_calls(self) -> int:
        return sum(c.cached_calls for c in self.contracts)

    @property
    def retries(self) -> int:
        return sum(c.retries for c in self.contracts)

    @property
    def outcomes(self) -> dict[str, int]:
        """How every call ended, counted across the contracts."""
        total: Counter = Counter()
        for contract in self.contracts:
            total.update(contract.outcomes)
        return dict(total)

    @property
    def label(self) -> str:
        """A readable name, such as "Zero-shot claude-opus-5-5, multi-label"."""
        if self.arm == "zero_shot":
            mode = self.settings.get("mode")
            name = f"Zero-shot {self.model.split('/', 1)[-1]}"
            return f"{name}, {mode}-label" if mode else name
        return f"Fine-tuned {self.model}"

    def restrict(self, contract_ids: Sequence[str], clause_types: Sequence[str]) -> "ArmResult":
        """The same arm, keeping only these contracts, in this order, and clause types.

        Latency and cost stay as measured for each contract's whole run.
        """
        by_id = {c.contract_id: c for c in self.contracts}
        contracts = [
            dataclasses.replace(
                by_id[cid],
                labels={ct: by_id[cid].labels[ct] for ct in clause_types},
                predictions={ct: by_id[cid].predictions[ct] for ct in clause_types},
            )
            for cid in contract_ids
            if cid in by_id
        ]
        return dataclasses.replace(self, clause_types=list(clause_types), contracts=contracts)

    def to_dict(self) -> dict:
        return {
            "format": RESULTS_FORMAT,
            "arm": self.arm,
            "model": self.model,
            "clause_types": self.clause_types,
            "settings": self.settings,
            "contracts": [dataclasses.asdict(c) for c in self.contracts],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ArmResult":
        if data.get("format") != RESULTS_FORMAT:
            raise ValueError(
                f"Unknown results format {data.get('format')!r}, expected {RESULTS_FORMAT}"
            )
        return cls(
            arm=data["arm"],
            model=data["model"],
            clause_types=list(data["clause_types"]),
            contracts=[ContractResult(**c) for c in data["contracts"]],
            settings=dict(data.get("settings", {})),
        )

    def save(self, path: str) -> None:
        """Write the results as JSON, creating the directory if needed."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)

    @classmethod
    def load(cls, path: str) -> "ArmResult":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def _labels(contract, clause_types: Sequence[str]) -> dict[str, int]:
    return {ct: int(contract.clauses.get(ct, False)) for ct in clause_types}


def _union(responses, clause_types: Sequence[str]) -> dict[str, int | None]:
    """Contract-level answers from the answers about its chunks.

    A clause type is present when any chunk has it.  Otherwise a failed chunk
    makes it unknown, because that chunk might have held it.
    """
    found = set().union(*(r.present for r in responses if r.present is not None))
    failed = any(r.present is None for r in responses)
    return {ct: 1 if ct in found else (None if failed else 0) for ct in clause_types}


def _contract_result(contract, clause_types, predictions, responses) -> ContractResult:
    costs = [r.cost_usd for r in responses]
    return ContractResult(
        contract_id=contract.contract_id,
        labels=_labels(contract, clause_types),
        predictions=predictions,
        latency_ms=sum(r.latency_ms for r in responses),
        cost_usd=None if any(cost is None for cost in costs) else sum(costs),
        calls=len(responses),
        cached_calls=sum(1 for r in responses if r.cached),
        retries=sum(r.retries for r in responses),
        input_tokens=sum(r.usage.input_tokens for r in responses),
        output_tokens=sum(r.usage.output_tokens for r in responses),
        cache_read_tokens=sum(r.usage.cache_read_tokens for r in responses),
        cache_write_tokens=sum(r.usage.cache_write_tokens for r in responses),
        outcomes=dict(Counter(r.outcome for r in responses)),
    )


def _ask_multi(client, chunks, clause_types):
    """Every clause type in one call per chunk."""
    responses = [client.classify(chunk, clause_types) for chunk in chunks]
    return _union(responses, clause_types), responses


def _ask_single(client, chunks, clause_types):
    """One clause type per call, moving to the next type at the first chunk that has it."""
    predictions, responses = {}, []
    for ct in clause_types:
        asked = []
        for chunk in chunks:
            asked.append(client.classify(chunk, [ct]))
            if asked[-1].present is not None and ct in asked[-1].present:
                break
        predictions[ct] = _union(asked, [ct])[ct]
        responses.extend(asked)
    return predictions, responses


def evaluate_zero_shot(
    client,
    contracts,
    clause_types: Sequence[str],
    mode: str = "multi",
    chunk_chars: int | None = None,
    chunk_overlap: int | None = None,
    concurrency: int = 1,
    on_contract: Callable[[int, int], None] | None = None,
) -> ArmResult:
    """Ask the LLM about every chunk of each contract.

    Up to concurrency contracts are asked about at the same time, each one
    call at a time.  Results come back in contract order whatever the
    concurrency.  A FatalLLMError stops the run and cancels what is queued.

    In multi-label mode each chunk is one call about every clause type.  In
    single-label mode each call is about one clause type, and the remaining
    chunks are skipped for that type once one chunk has it.  Either way a
    clause is present in a contract when any chunk has it.  When no chunk has
    it and a call failed, the answer is unknown, so that contract and clause
    type are left out of the metrics and counted in failures rather than
    scored as absent.  Latency and cost per contract are the sums over its
    calls, so they do not depend on the concurrency.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    chunk_chars = chunk_chars or config.llm.chunk_chars
    chunk_overlap = config.llm.chunk_overlap if chunk_overlap is None else chunk_overlap
    ask = _ask_multi if mode == "multi" else _ask_single

    def run(contract):
        spans = char_chunks(contract.text, chunk_chars, chunk_overlap)
        chunks = [contract.text[start:end] for start, end in spans]
        predictions, responses = ask(client, chunks, clause_types)
        return _contract_result(contract, clause_types, predictions, responses)

    results: list[ContractResult | None] = [None] * len(contracts)
    pool = ThreadPoolExecutor(max_workers=max(1, concurrency))
    try:
        futures = {pool.submit(run, contract): i for i, contract in enumerate(contracts)}
        for done, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            if on_contract:
                on_contract(done, len(contracts))
    finally:
        # On an error, drop the queued contracts instead of asking about them
        pool.shutdown(wait=True, cancel_futures=True)

    return ArmResult(
        arm="zero_shot",
        model=f"{getattr(client, 'provider', '')}/{getattr(client, 'model', '')}".strip("/"),
        clause_types=list(clause_types),
        contracts=results,
        settings={
            "mode": mode,
            "prompt_version": PROMPT_VERSION,
            "chunk_chars": chunk_chars,
            "chunk_overlap": chunk_overlap,
            "concurrency": concurrency,
        },
    )


def evaluate_fine_tuned(
    classifier,
    contracts,
    clause_types: Sequence[str],
    threshold: float | None = None,
    cost_per_hour: float | None = None,
    on_contract: Callable[[int, int], None] | None = None,
) -> ArmResult:
    """Score each contract with the windowed classifier.

    Latency per contract is the measured time to tokenise and score all of
    its windows.  Cost is that time multiplied by cost_per_hour, and is left
    unpriced (None) when no rate is given.
    """
    threshold = config.training.threshold if threshold is None else threshold
    missing = [ct for ct in clause_types if ct not in classifier.clause_types]
    if missing:
        raise ValueError(
            f"The fine-tuned model has no output for {missing}.  "
            f"It was trained on {classifier.clause_types}."
        )

    results = []
    for n, contract in enumerate(contracts, 1):
        probs, ms = classifier.predict_contract(contract.text)
        results.append(
            ContractResult(
                contract_id=contract.contract_id,
                labels=_labels(contract, clause_types),
                predictions={ct: int(probs[ct] >= threshold) for ct in clause_types},
                latency_ms=ms,
                cost_usd=None if cost_per_hour is None else ms / 3_600_000 * cost_per_hour,
            )
        )
        if on_contract:
            on_contract(n, len(contracts))

    return ArmResult(
        arm="fine_tuned",
        model=getattr(classifier, "base_model_name", None) or getattr(classifier, "model_name", ""),
        clause_types=list(clause_types),
        contracts=results,
        settings={"threshold": threshold, "cost_per_hour": cost_per_hour},
    )

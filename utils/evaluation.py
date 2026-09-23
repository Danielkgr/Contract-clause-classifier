"""
Contract-level evaluation for both arms.

Each arm answers one question per contract and clause type, whether the
clause appears anywhere in the contract, so both are scored against the same
CUAD labels.  Latency and cost are measured per contract.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from config import config
from utils.chunking import char_chunks
from utils.metrics import (
    ClassificationMetrics, InferenceStats, calculate_metrics, summarise_documents,
)


@dataclass
class ArmResult:
    """What one arm produced on the test contracts."""
    metrics: Dict[str, ClassificationMetrics]
    stats: InferenceStats
    predictions: Dict[str, List[int]]
    labels: Dict[str, List[int]]
    # Contract and clause-type decisions left out because an LLM call failed
    failures: Dict[str, int] = field(default_factory=dict)
    calls: int = 0
    model: str = ""


def _score(labels: Dict[str, List[int]], predictions: Dict[str, List[int]]
           ) -> Dict[str, ClassificationMetrics]:
    return {ct: calculate_metrics(labels[ct], predictions[ct])
            for ct in predictions if predictions[ct]}


def evaluate_zero_shot(
    client,
    contracts,
    clause_types: Sequence[str],
    chunk_chars: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    on_contract: Optional[Callable[[int, int], None]] = None,
) -> ArmResult:
    """Ask the LLM about each clause type, one contract chunk at a time.

    A clause is present when any chunk gets YES, so later chunks are skipped
    after the first YES.  If a call fails before a YES, the decision for that
    contract and clause type is left out and counted in failures, rather than
    scored as NO.  Latency and cost per contract are the sums over its calls.
    """
    chunk_chars = chunk_chars or config.llm.chunk_chars
    chunk_overlap = config.llm.chunk_overlap if chunk_overlap is None else chunk_overlap

    predictions = {ct: [] for ct in clause_types}
    labels = {ct: [] for ct in clause_types}
    failures = {ct: 0 for ct in clause_types}
    latencies, costs = [], []
    calls = input_tokens = output_tokens = 0

    for n, contract in enumerate(contracts, 1):
        chunks = char_chunks(contract.text, chunk_chars, chunk_overlap)
        doc_ms = doc_cost = 0.0
        for ct in clause_types:
            decision, failed = 0, False
            for start, end in chunks:
                response = client.classify_single(contract.text[start:end], ct)
                calls += 1
                doc_ms += response.latency_ms
                doc_cost += response.cost_usd
                input_tokens += response.input_tokens
                output_tokens += response.output_tokens
                if response.error:
                    failed = True
                    break
                if response.text.strip().upper().startswith("YES"):
                    decision = 1
                    break
            if failed:
                failures[ct] += 1
                continue
            predictions[ct].append(decision)
            labels[ct].append(int(contract.clauses.get(ct, False)))
        latencies.append(doc_ms)
        costs.append(doc_cost)
        if on_contract:
            on_contract(n, len(contracts))

    return ArmResult(
        metrics=_score(labels, predictions),
        stats=summarise_documents(latencies, costs, input_tokens, output_tokens),
        predictions=predictions,
        labels=labels,
        failures=failures,
        calls=calls,
        model=f"{getattr(client, 'provider', '')}/{getattr(client, 'model', '')}".strip("/"),
    )


def evaluate_fine_tuned(
    classifier,
    contracts,
    clause_types: Sequence[str],
    threshold: Optional[float] = None,
    cost_per_hour: Optional[float] = None,
    on_contract: Optional[Callable[[int, int], None]] = None,
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

    predictions = {ct: [] for ct in clause_types}
    labels = {ct: [] for ct in clause_types}
    latencies = []

    for n, contract in enumerate(contracts, 1):
        probs, ms = classifier.predict_contract(contract.text)
        latencies.append(ms)
        for ct in clause_types:
            predictions[ct].append(int(probs[ct] >= threshold))
            labels[ct].append(int(contract.clauses.get(ct, False)))
        if on_contract:
            on_contract(n, len(contracts))

    costs = None
    if cost_per_hour is not None:
        costs = [ms / 3_600_000 * cost_per_hour for ms in latencies]

    return ArmResult(
        metrics=_score(labels, predictions),
        stats=summarise_documents(latencies, costs),
        predictions=predictions,
        labels=labels,
        model=getattr(classifier, "base_model_name", None) or getattr(classifier, "model_name", ""),
    )

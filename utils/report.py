"""
The comparison report, written from saved arm results.

Every arm is scored on the contracts and clause types that all the arms
share, so the figures compare like with like.  Nothing here calls a model.
"""

import os
from collections.abc import Sequence

import pandas as pd

from utils.evaluation import ArmResult
from utils.metrics import aggregate_metrics

METRICS = ("precision", "recall", "f1", "accuracy")
COLOURS = ("#2ecc71", "#3498db", "#e67e22", "#9b59b6", "#e74c3c", "#7f8c8d")
FAILED_OUTCOMES = {
    "refusal": "refused",
    "truncated": "cut off at max tokens",
    "unparseable": "not the JSON asked for",
    "error": "failed with an error",
}


def align(arms: Sequence[ArmResult]) -> tuple[list[ArmResult], list[str]]:
    """Keep only the contracts and clause types every arm covers.

    Returns the restricted arms and notes on anything left out.
    """
    if len(arms) < 2:
        return list(arms), []
    shared = set.intersection(*({c.contract_id for c in arm.contracts} for arm in arms))
    ids = [c.contract_id for c in arms[0].contracts if c.contract_id in shared]
    types = [ct for ct in arms[0].clause_types if all(ct in arm.clause_types for arm in arms)]
    notes = []
    if any(len(arm.contracts) != len(ids) for arm in arms):
        notes.append(f"Every arm is scored on the {len(ids)} contracts that all of them cover.")
    dropped = sorted({ct for arm in arms for ct in arm.clause_types} - set(types))
    if dropped:
        notes.append(f"Clause types that not every arm covers are left out: {', '.join(dropped)}.")
    return [arm.restrict(ids, types) for arm in arms], notes


def _usd(value: float | None) -> str:
    return "not priced" if value is None else f"${value:,.4f}"


def _ms(value: float) -> str:
    return f"{value:,.0f} ms"


def _llm(arm: ArmResult, value) -> str:
    """A value that applies only to an arm that calls an LLM."""
    return str(value) if arm.arm == "zero_shot" else "none"


def models_table(arms: Sequence[ArmResult]) -> str:
    """Model, calls, latency, and cost per arm, as a Markdown table."""
    rows = [
        ("Model", lambda a: a.model),
        ("Contracts", lambda a: str(a.stats.documents)),
        ("LLM calls", lambda a: _llm(a, f"{a.calls:,}")),
        ("Calls answered from the cache", lambda a: _llm(a, f"{a.cached_calls:,}")),
        ("Failed calls", lambda a: _llm(a, f"{a.calls - a.outcomes.get('ok', 0):,}")),
        ("Average latency per contract", lambda a: _ms(a.stats.avg_latency_ms)),
        (
            "Fastest and slowest contract",
            lambda a: f"{_ms(a.stats.min_latency_ms)} and {_ms(a.stats.max_latency_ms)}",
        ),
        ("Cost per contract", lambda a: _usd(a.stats.avg_cost_usd)),
        ("Total cost", lambda a: _usd(a.stats.total_cost_usd)),
    ]
    md = "|  | " + " | ".join(a.label for a in arms) + " |\n"
    md += "|---|" + "---|" * len(arms) + "\n"
    for name, cell in rows:
        md += f"| {name} | " + " | ".join(cell(a) for a in arms) + " |\n"
    return md


def aggregate_table(arms: Sequence[ArmResult]) -> str:
    """Mean precision, recall, F1, and accuracy across clause types."""
    md = "|  | Precision | Recall | F1 | Accuracy |\n|---|---|---|---|---|\n"
    for arm in arms:
        agg = aggregate_metrics(arm.metrics)
        if not agg:
            md += f"| {arm.label} | no result | no result | no result | no result |\n"
            continue
        md += (
            f"| {arm.label} | {agg['avg_precision']:.4f} | {agg['avg_recall']:.4f} | "
            f"{agg['avg_f1']:.4f} | {agg['avg_accuracy']:.4f} |\n"
        )
    return md


def per_clause_table(arms: Sequence[ArmResult]) -> str:
    """Every metric and confusion count for each clause type and arm."""
    md = (
        "| Clause type | Method | Precision | Recall | F1 | Accuracy | TP | FP | TN | FN |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    for ct in arms[0].clause_types if arms else []:
        for arm in arms:
            m = arm.metrics.get(ct)
            if m:
                md += (
                    f"| {ct} | {arm.label} | {m.precision:.4f} | {m.recall:.4f} | {m.f1:.4f} | "
                    f"{m.accuracy:.4f} | {m.true_positives} | {m.false_positives} | "
                    f"{m.true_negatives} | {m.false_negatives} |\n"
                )
    return md


def notes(arms: Sequence[ArmResult], extra: Sequence[str] = ()) -> list[str]:
    """What a reader needs to know to trust the figures."""
    found = list(extra)
    if len(arms) == 1:
        found.append("Only one arm has results, so there is nothing to compare it with yet.")
    for arm in arms:
        if not arm.metrics:
            found.append(f"{arm.label} has no result, because every decision it made failed.")
        failed = {ct: n for ct, n in arm.failures.items() if n}
        if failed:
            found.append(
                f"{arm.label}: decisions left out because a call that could have found the "
                f"clause failed: {failed}."
            )
        outcomes = arm.outcomes
        bad = [f"{outcomes[k]} {text}" for k, text in FAILED_OUTCOMES.items() if outcomes.get(k)]
        if bad:
            found.append(f"{arm.label}: of {arm.calls:,} calls, {', '.join(bad)}.")
        if arm.cached_calls:
            found.append(
                f"{arm.label}: {arm.cached_calls:,} of {arm.calls:,} calls were answered from the "
                "response cache.  Their latency and cost are those measured when they were made."
            )
        if arm.arm == "fine_tuned" and arm.stats.avg_cost_usd is None:
            found.append(
                f"{arm.label} is not priced.  Its latency is measured compute time, and "
                "--cost-per-hour or FT_COST_PER_HOUR turns that time into a cost."
            )
    return found


def metrics_rows(arms: Sequence[ArmResult]) -> list[dict]:
    """One row per clause type, arm, and metric, for the CSV."""
    rows = []
    for ct in arms[0].clause_types if arms else []:
        for arm in arms:
            m = arm.metrics.get(ct)
            if m:
                for metric in METRICS:
                    rows.append(
                        {
                            "clause_type": ct,
                            "method": arm.label,
                            "arm": arm.arm,
                            "model": arm.model,
                            "metric": metric,
                            "value": getattr(m, metric),
                        }
                    )
    return rows


def write_plot(arms: Sequence[ArmResult], path: str) -> bool:
    """A two-by-two bar chart of the four metrics.  False when there is nothing to plot."""
    clause_types = [ct for ct in arms[0].clause_types if all(ct in a.metrics for a in arms)]
    if not clause_types:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(clause_types))
    width = 0.8 / len(arms)
    for ax, metric in zip(axes.ravel(), METRICS, strict=True):
        for i, arm in enumerate(arms):
            values = [getattr(arm.metrics[ct], metric) for ct in clause_types]
            offset = (i - (len(arms) - 1) / 2) * width
            colour = COLOURS[i % len(COLOURS)]
            ax.bar(x + offset, values, width, label=arm.label, color=colour, alpha=0.8)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} by clause type")
        ax.set_xticks(x)
        ax.set_xticklabels(clause_types, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _settings(arm: ArmResult) -> str:
    return ", ".join(f"{key} {value}" for key, value in arm.settings.items() if value is not None)


def write_report(arms: Sequence[ArmResult], out_dir: str) -> str:
    """Write the CSV, summary, full report, and plot to out_dir, and return the summary."""
    arms, alignment = align(arms)
    os.makedirs(out_dir, exist_ok=True)
    found = notes(arms, alignment)
    notes_md = "".join(f"- {note}\n" for note in found)
    contracts = len(arms[0].contracts) if arms else 0
    clause_types = arms[0].clause_types if arms else []

    pd.DataFrame(metrics_rows(arms)).to_csv(
        os.path.join(out_dir, "comparison_metrics.csv"), index=False
    )
    summary = (
        "# Contract Clause Classifier Comparison\n\n"
        f"{contracts} test contracts, {len(clause_types)} clause types.\n\n"
        f"## Models\n\n{models_table(arms)}\n"
        f"## Aggregate metrics\n\n{aggregate_table(arms)}\n"
        + (f"## Notes\n\n{notes_md}" if found else "")
    )
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(summary)

    report = (
        "# Contract Clause Classification Report\n\n"
        "Each method decides, for each test contract and clause type, whether the clause appears "
        "anywhere in the contract.  Labels come from CUAD answer spans.\n\n"
        f"## Data\n\n- Test contracts: {contracts}\n- Clause types: {', '.join(clause_types)}\n\n"
        f"## Models and measured cost\n\n{models_table(arms)}\n"
        f"## Aggregate metrics\n\n{aggregate_table(arms)}\n"
        f"## Per-clause comparison\n\n{per_clause_table(arms)}\n"
        "## Configuration\n\n"
        + "".join(f"- {arm.label}: {_settings(arm)}\n" for arm in arms)
        + (f"\n## Notes\n\n{notes_md}" if found else "")
    )
    with open(os.path.join(out_dir, "comparison_report.md"), "w", encoding="utf-8") as fh:
        fh.write(report)

    write_plot(arms, os.path.join(out_dir, "comparison_plot.png"))
    return summary

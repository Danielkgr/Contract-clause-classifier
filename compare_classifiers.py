"""
Contract Clause Classifier — Interactive CLI

Run ``python compare_classifiers.py`` and choose options from a styled
menu. Every setting (LLM provider / model / API key, training params,
clause types, output path) is configurable inside the CLI — no manual
`.env` edits required.

Legacy flags still work: ``--quick-test``, ``--max-samples N``,
``--output DIR``, ``--interactive``.
"""

import os
import sys
import logging
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ rich ---
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.rule import Rule
from rich.logging import RichHandler
from rich.table import Table
from rich.tree import Tree

from config import config, DEFAULT_CLAUSE_TYPES, is_cuad_category
from utils.llm_client import LLMClient
from utils.data_loader import load_cuad_dataset, get_clause_distribution
from utils.classifier import FineTunedClassifier, TrainingResult
from utils.evaluation import ArmResult, evaluate_zero_shot, evaluate_fine_tuned
from utils.metrics import aggregate_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, console=Console(stderr=True))],
)
logger = logging.getLogger(__name__)

console = Console()

# ----------------------------------------------------------------------- #
# Data classes                                                              #
# ----------------------------------------------------------------------- #

# A quick test asks the LLM about this many test contracts
QUICK_TEST_CONTRACTS = 5


@dataclass
class ComparisonResult:
    """Complete comparison result."""
    documents: int
    clause_types: List[str]
    zero_shot: ArmResult
    fine_tuned: Optional[ArmResult]  # None when no trained model was available
    training_result: Optional[TrainingResult]


# ----------------------------------------------------------------------- #
# CLI helpers                                                               #
# ----------------------------------------------------------------------- #

def _header():
    console.print(Rule("[bold cyan]Contract Clause Classifier[/]", style="dim"))
    console.print(
        Panel(
            "[green]Compare[/] zero-shot LLM vs fine-tuned transformer on\n"
            "the [blue]CUAD[/] contract dataset.",
            border_style="cyan",
            padding=(0, 1),
        )
    )


def _main_menu() -> int:
    """Return user's main-menu choice (int)."""
    options = [
        (f"Quick test ({QUICK_TEST_CONTRACTS} contracts, zero-shot)", "quick"),
        ("Full comparison (train + evaluate)", "full"),
        ("Train only", "train"),
        ("Evaluate only (use existing model)", "evaluate"),
        ("Configure settings", "configure"),
        ("View configuration", "view_config"),
        ("Export / Import settings", "export_import"),
        ("Exit", "exit"),
    ]

    console.print()
    console.print(Rule("[bold]Main Menu[/]", style="bold cyan"))
    for i, (label, _) in enumerate(options, 1):
        console.print(f"  [bold yellow]{i}[/]. {label}")
    console.print()

    choice = IntPrompt.ask(
        "Choose an option",
        default=1,
        choices=[str(i) for i in range(1, len(options) + 1)],
    )
    return choice


# ----------------------------------------------------------------------- #
# Configuration sub-menus                                                  #
# ----------------------------------------------------------------------- #

def _configure_llm_settings():
    """Full LLM configuration: provider, model, API key, temperature, max tokens, base URL."""
    console.print()
    console.print(Rule("[bold]LLM Configuration[/]", style="bold cyan"))

    # Provider
    provider = Prompt.ask(
        "  [cyan]1[/]. Provider",
        default=config.llm.provider,
        choices=["openai", "anthropic", "google", "custom"],
    )
    config.llm.provider = provider

    # Model
    model = Prompt.ask("  [cyan]2[/]. Model name", default=config.llm.model)
    config.llm.model = model

    # API Key
    api_key = Prompt.ask("  [cyan]3[/]. API key", default=config.llm.api_key or "(not set)")
    config.llm.api_key = api_key if api_key != "(not set)" else None

    # Base URL
    base_url = Prompt.ask("  [cyan]4[/]. Base URL (optional)", default=config.llm.base_url or "(none)")
    config.llm.base_url = base_url if base_url != "(none)" else None

    # Temperature
    temp_str = Prompt.ask("  [cyan]5[/]. Temperature", default=str(config.llm.temperature))
    try:
        config.llm.temperature = float(temp_str)
    except ValueError:
        console.print("  [yellow]Invalid temperature; keeping previous value.[/]")

    # Max tokens
    max_t_str = Prompt.ask("  [cyan]6[/]. Max tokens", default=str(config.llm.max_tokens))
    try:
        config.llm.max_tokens = int(max_t_str)
    except ValueError:
        console.print("  [yellow]Invalid max tokens; keeping previous value.[/]")

    # Price lookup for the chosen model
    if config.llm.model in config.llm.COSTS:
        console.print("  [green]✓ Listed price found for this model[/]")
    else:
        console.print("  [yellow]No listed price for this model, so costs use the default rate.[/]")


def _configure_training_settings():
    """Training configuration: model name, epochs, batch size, lr, max length, weight decay, warmup, eval/save steps."""
    console.print()
    console.print(Rule("[bold]Training Configuration[/]", style="bold cyan"))

    model_name = Prompt.ask(
        "  [cyan]1[/]. Model name",
        default=config.training.model_name,
    )
    config.training.model_name = model_name

    epochs_str = Prompt.ask("  [cyan]2[/]. Epochs", default=str(config.training.num_epochs))
    try:
        config.training.num_epochs = int(epochs_str)
    except ValueError:
        console.print("  [yellow]Invalid epochs; keeping previous value.[/]")

    bs_str = Prompt.ask("  [cyan]3[/]. Batch size", default=str(config.training.batch_size))
    try:
        config.training.batch_size = int(bs_str)
    except ValueError:
        console.print("  [yellow]Invalid batch size; keeping previous value.[/]")

    lr_str = Prompt.ask(
        "  [cyan]4[/]. Learning rate",
        default=str(config.training.learning_rate),
    )
    try:
        config.training.learning_rate = float(lr_str)
    except ValueError:
        console.print("  [yellow]Invalid learning rate; keeping previous value.[/]")

    ml_str = Prompt.ask(
        "  [cyan]5[/]. Max token length",
        default=str(config.training.max_length),
    )
    try:
        config.training.max_length = int(ml_str)
    except ValueError:
        console.print("  [yellow]Invalid max length; keeping previous value.[/]")

    wd_str = Prompt.ask(
        "  [cyan]6[/]. Weight decay",
        default=str(config.training.weight_decay),
    )
    try:
        config.training.weight_decay = float(wd_str)
    except ValueError:
        console.print("  [yellow]Invalid weight decay; keeping previous value.[/]")

    ws_str = Prompt.ask(
        "  [cyan]7[/]. Warmup steps",
        default=str(config.training.warmup_steps),
    )
    try:
        config.training.warmup_steps = int(ws_str)
    except ValueError:
        console.print("  [yellow]Invalid warmup steps; keeping previous value.[/]")

    es_str = Prompt.ask(
        "  [cyan]8[/]. Eval steps",
        default=str(config.training.eval_steps),
    )
    try:
        config.training.eval_steps = int(es_str)
    except ValueError:
        console.print("  [yellow]Invalid eval steps; keeping previous value.[/]")

    ss_str = Prompt.ask(
        "  [cyan]9[/]. Save steps",
        default=str(config.training.save_steps),
    )
    try:
        config.training.save_steps = int(ss_str)
    except ValueError:
        console.print("  [yellow]Invalid save steps; keeping previous value.[/]")


def _configure_clause_types_menu():
    """Interactive clause-type selection with add / remove support."""
    all_types = list(config.data.clause_types)

    while True:
        console.print()
        console.print(Rule("[bold]Clause Types[/]", style="bold cyan"))
        console.print(f"  [dim]Active ({len(all_types)}):[/]")
        for i, ct in enumerate(all_types, 1):
            console.print(f"    [cyan]{i}[/]. [green]✓[/] {ct}")

        # Show default clause types that aren't active yet
        extra = [ct for ct in DEFAULT_CLAUSE_TYPES if ct not in all_types]
        if extra:
            console.print()
            console.print("  [dim]Available defaults:[/]", style="dim")
            for ct in extra:
                console.print(f"    • {ct}")

        console.print()
        console.print("  [bold yellow]1[/]. Toggle active clauses")
        console.print("  [bold yellow]2[/]. Add custom clause type")
        console.print("  [bold yellow]3[/]. Remove selected clause type")
        console.print("  [bold yellow]4[/]. Select all default clause types")
        console.print("  [bold yellow]0[/]. Back to configuration menu")

        choice = Prompt.ask(
            "Choose action",
            choices=["0", "1", "2", "3", "4"],
            default="0",
        )

        if choice == "0":
            # Update the config list and return
            config.data.clause_types = all_types
            break

        elif choice == "1":
            console.print()
            for i, ct in enumerate(all_types, 1):
                ok = Confirm.ask(f"  [cyan]{i}[/]. {ct}", default=True)
                if not ok:
                    all_types.remove(ct)

        elif choice == "2":
            new_ct = Prompt.ask("  Enter new clause type name")
            if new_ct and new_ct.strip():
                if not is_cuad_category(new_ct.strip()):
                    console.print(
                        f"  [yellow]{new_ct.strip()} is not a CUAD category, so loading CUAD "
                        f"will stop with an error until it is removed.[/]"
                    )
                config.data.add_clause_type(new_ct.strip())
                all_types = list(config.data.clause_types)

        elif choice == "3":
            if not all_types:
                console.print("  [yellow]No clauses to remove.[/]")
                continue
            ct = Prompt.ask(
                "  Enter clause type name to remove",
                choices=all_types,
            )
            config.data.remove_clause_type(ct)
            all_types = list(config.data.clause_types)

        elif choice == "4":
            for ct in DEFAULT_CLAUSE_TYPES:
                config.data.add_clause_type(ct)
            all_types = list(config.data.clause_types)


def _configure_output_dir():
    new_dir = Prompt.ask("  Output directory path", default=config.paths.outputs_dir)
    if new_dir != config.paths.outputs_dir:
        config.paths.outputs_dir = new_dir
        os.makedirs(new_dir, exist_ok=True)
        console.print(f"  [green]✓ Output dir updated to:[/]\n    [dim]{new_dir}[/]")


def _configure_menu():
    """Top-level configuration submenu covering every setting."""
    while True:
        console.print()
        console.print(Rule("[bold]Configuration Menu[/]", style="bold cyan"))
        console.print("  [bold yellow]1[/]. LLM Settings (provider, model, API key, temp, tokens, base URL)")
        console.print("  [bold yellow]2[/]. Training Settings (model, epochs, lr, batch size, max length, …)")
        console.print("  [bold yellow]3[/]. Clause Types (select / add / remove)")
        console.print("  [bold yellow]4[/]. Output Directory")
        console.print("  [bold yellow]0[/]. Back to main menu")

        choice = Prompt.ask(
            "Choose setting",
            choices=["0", "1", "2", "3", "4"],
            default="0",
        )

        if choice == "0":
            break
        elif choice == "1":
            _configure_llm_settings()
        elif choice == "2":
            _configure_training_settings()
        elif choice == "3":
            _configure_clause_types_menu()
        elif choice == "4":
            _configure_output_dir()


def _export_import_menu():
    """Export / Import settings as .env or JSON."""
    console.print()
    console.print(Rule("[bold]Settings[/]", style="bold cyan"))
    console.print("  [bold yellow]1[/]. Export to .env")
    console.print("  [bold yellow]2[/]. Show current environment (read-only)")
    console.print("  [bold yellow]0[/]. Back")

    choice = Prompt.ask("Choose action", choices=["0", "1", "2"], default="0")
    if choice == "1":
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local")
        with open(path, "w") as f:
            f.write(f"LLM_PROVIDER={config.llm.provider}\n")
            f.write(f"LLM_MODEL={config.llm.model}\n")
            f.write(f"LLM_API_KEY={config.llm.api_key or ''}\n")
            f.write(f"LLM_BASE_URL={config.llm.base_url or ''}\n")
            f.write(f"LLM_TEMPERATURE={config.llm.temperature}\n")
            f.write(f"LLM_MAX_TOKENS={config.llm.max_tokens}\n")
            f.write(f"TRAIN_MODEL={config.training.model_name}\n")
            f.write(f"BATCH_SIZE={config.training.batch_size}\n")
            f.write(f"LR={config.training.learning_rate}\n")
            f.write(f"NUM_EPOCHS={config.training.num_epochs}\n")
            f.write(f"MAX_LENGTH={config.training.max_length}\n")
            f.write(f"WEIGHT_DECAY={config.training.weight_decay}\n")
            f.write(f"WARMUP_STEPS={config.training.warmup_steps}\n")
            f.write(f"EVAL_STEPS={config.training.eval_steps}\n")
            f.write(f"SAVE_STEPS={config.training.save_steps}\n")
        console.print(Panel(
            f"[green]✓ Settings exported to:[/]\n  [dim]{path}[/]",
            border_style="green",
        ))

    elif choice == "2":
        table = Table(title="Current Environment (would be written to .env)", show_header=True)
        table.add_column("Variable")
        table.add_column("Value", style="green")
        for row in [
            ("LLM_PROVIDER", config.llm.provider),
            ("LLM_MODEL", config.llm.model),
            ("LLM_API_KEY", config.llm.api_key or "(not set)"),
            ("LLM_BASE_URL", config.llm.base_url or ""),
            ("LLM_TEMPERATURE", str(config.llm.temperature)),
            ("LLM_MAX_TOKENS", str(config.llm.max_tokens)),
            ("TRAIN_MODEL", config.training.model_name),
            ("BATCH_SIZE", str(config.training.batch_size)),
            ("LR", str(config.training.learning_rate)),
            ("NUM_EPOCHS", str(config.training.num_epochs)),
            ("MAX_LENGTH", str(config.training.max_length)),
            ("WEIGHT_DECAY", str(config.training.weight_decay)),
            ("WARMUP_STEPS", str(config.training.warmup_steps)),
            ("EVAL_STEPS", str(config.training.eval_steps)),
            ("SAVE_STEPS", str(config.training.save_steps)),
        ]:
            table.add_row(*row)
        console.print(table)


# ----------------------------------------------------------------------- #
# View config (read-only summary)                                         #
# ----------------------------------------------------------------------- #

def _view_config():
    table = Table(title="Current Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Category")
    table.add_column("Setting")
    table.add_column("Value", style="green")

    rows = [
        ("LLM", "Provider", config.llm.provider),
        ("LLM", "Model", config.llm.model),
        ("LLM", "API Key", (config.llm.api_key or "(not set)")[:32] + ("…" if len(config.llm.api_key or "") > 32 else "")),
        ("LLM", "Base URL", config.llm.base_url or "(none)"),
        ("LLM", "Temperature", str(config.llm.temperature)),
        ("LLM", "Max Tokens", str(config.llm.max_tokens)),
        ("Training", "Model", config.training.model_name),
        ("Training", "Epochs", str(config.training.num_epochs)),
        ("Training", "Batch Size", str(config.training.batch_size)),
        ("Training", "Learning Rate", str(config.training.learning_rate)),
        ("Training", "Max Length", str(config.training.max_length)),
        ("Training", "Weight Decay", str(config.training.weight_decay)),
        ("Training", "Warmup Steps", str(config.training.warmup_steps)),
        ("Training", "Eval Steps", str(config.training.eval_steps)),
        ("Training", "Save Steps", str(config.training.save_steps)),
        ("Data", "Clause Types", f"{len(config.data.clause_types)} active"),
        ("Paths", "Output Dir", config.paths.outputs_dir),
    ]

    for row in rows:
        table.add_row(*row)

    console.print(table)


# ----------------------------------------------------------------------- #
# Results display helpers                                                   #
# ----------------------------------------------------------------------- #

def _arms(result: "ComparisonResult"):
    """(key, label, ArmResult) for each arm that produced a result."""
    return [
        (key, label, arm)
        for key, label, arm in (
            ("zero_shot", "Zero-Shot LLM", result.zero_shot),
            ("fine_tuned", "Fine-Tuned", result.fine_tuned),
        )
        if arm is not None
    ]


def _usd(value: Optional[float]) -> str:
    return "not priced" if value is None else f"${value:.6f}"


def _show_results_table(result: "ComparisonResult"):
    """Render a Rich comparison table."""
    table = Table(title="Classifier Comparison", show_header=True, header_style="bold cyan")
    table.add_column("Clause Type", style="dim")
    table.add_column("Method")
    for name in ("Precision", "Recall", "F1", "Accuracy"):
        table.add_column(name, justify="right")

    for ct in result.clause_types:
        for _, label, arm in _arms(result):
            m = arm.metrics.get(ct)
            if m:
                table.add_row(ct, label, f"{m.precision:.4f}", f"{m.recall:.4f}",
                              f"{m.f1:.4f}", f"{m.accuracy:.4f}")
    console.print(table)


def _show_cost_latency(result: "ComparisonResult"):
    """Measured latency and cost per contract for each arm."""
    table = Table(title=f"Per Contract ({result.documents} contracts)", show_header=True,
                  header_style="bold cyan")
    table.add_column("Method")
    for name in ("Avg latency", "Min", "Max", "Cost / contract", "Calls"):
        table.add_column(name, justify="right")
    for key, label, arm in _arms(result):
        s = arm.stats
        table.add_row(label, f"{s.avg_latency_ms:.0f} ms", f"{s.min_latency_ms:.0f} ms",
                      f"{s.max_latency_ms:.0f} ms", _usd(s.avg_cost_usd),
                      str(arm.calls) if key == "zero_shot" else "")
    console.print(table)
    if result.fine_tuned and result.fine_tuned.stats.avg_cost_usd is None:
        console.print("  [dim]Set FT_COST_PER_HOUR to price the fine-tuned arm's measured compute time.[/]")


def _show_clause_dist(dist):
    table = Table(title="Clause Distribution", show_header=True)
    table.add_column("Clause Type", style="dim")
    table.add_column("Present", style="green")
    table.add_column("Absent", style="red")
    for row in dist.itertuples():
        table.add_row(row.Index, str(row.present), str(row.absent))
    console.print(table)


# ----------------------------------------------------------------------- #
# Core comparison logic                                                     #
# ----------------------------------------------------------------------- #

def _progress(label: str):
    def report(n: int, total: int):
        console.print(f"  [dim]{label}: contract {n} of {total}[/]")
    return report


def _train_fine_tuned(clause_types: List[str], max_train: Optional[int] = None):
    """Train on the train split, evaluating on the validation split."""
    console.print("[cyan]Loading the train and validation splits…[/]")
    train_contracts = load_cuad_dataset(split="train", max_samples=max_train)
    val_contracts = load_cuad_dataset(split="validation")
    fine_tuned = FineTunedClassifier(clause_types)
    console.print(f"[cyan]Training on {len(train_contracts)} contracts…[/]")
    training_result = fine_tuned.train(train_contracts, val_contracts)
    console.print(Panel(
        f"[green]✓ Training complete[/]\n"
        f"  Time: {training_result.training_time_seconds:.1f}s\n"
        f"  Windows: {training_result.train_windows} train, {training_result.val_windows} validation\n"
        f"  Metrics: {training_result.metrics}",
        border_style="green",
    ))
    return fine_tuned, training_result


def _load_saved_model(clause_types: List[str]) -> Optional[FineTunedClassifier]:
    """Load models/fine_tuned if it exists and covers the clause types."""
    path = os.path.join(config.paths.models_dir, "fine_tuned")
    if not os.path.exists(os.path.join(path, "config.json")):
        console.print("[yellow]No saved model in models/fine_tuned, so the fine-tuned arm is "
                      "skipped.  Train one with option 2 or 3.[/]")
        return None
    fine_tuned = FineTunedClassifier(clause_types, model_name=path)
    try:
        fine_tuned.load(path)
    except ValueError as e:
        console.print(f"[yellow]{e}  The fine-tuned arm is skipped.[/]")
        return None
    missing = [ct for ct in clause_types if ct not in fine_tuned.clause_types]
    if missing:
        console.print(f"[yellow]The saved model has no output for {missing}, so the fine-tuned "
                      f"arm is skipped.  Retrain with the current clause types.[/]")
        return None
    return fine_tuned


def _run_comparison(
    max_samples: Optional[int] = None,
    clause_types: Optional[List[str]] = None,
    mode: str = "full",
) -> Optional["ComparisonResult"]:
    """Evaluate both arms on the test split.

    mode is "quick" (a few contracts, saved model only), "full" (train, then
    evaluate), or "evaluate" (saved model only).
    """
    clause_types = list(clause_types or config.data.clause_types)
    limit = QUICK_TEST_CONTRACTS if mode == "quick" else max_samples

    console.print()
    console.print("[cyan]Loading the test split…[/]")
    contracts = load_cuad_dataset(split="test", max_samples=limit)
    if not contracts:
        console.print("[red]No test contracts were loaded.[/]")
        return None
    console.print(f"  [bold green]✓ Loaded {len(contracts)} contracts[/]")
    _show_clause_dist(get_clause_distribution(contracts))

    training_result = None
    if mode == "full":
        fine_tuned, training_result = _train_fine_tuned(clause_types)
    else:
        fine_tuned = _load_saved_model(clause_types)

    console.print()
    console.print("[cyan]Evaluating the zero-shot LLM…[/]")
    zero_shot = evaluate_zero_shot(
        LLMClient(), contracts, clause_types, on_contract=_progress("Zero-shot"),
    )
    for ct, n in zero_shot.failures.items():
        if n:
            console.print(f"[yellow]  {n} of {len(contracts)} contracts had a failed call for "
                          f"{ct} and were left out of its metrics.[/]")
    for ct in clause_types:
        if ct not in zero_shot.metrics:
            console.print(f"[red]  Every call failed for {ct}, so it has no zero-shot result.[/]")

    fine_tuned_result = None
    if fine_tuned is not None:
        console.print("[cyan]Evaluating the fine-tuned model…[/]")
        fine_tuned_result = evaluate_fine_tuned(
            fine_tuned, contracts, clause_types,
            cost_per_hour=config.training.cost_per_hour,
            on_contract=_progress("Fine-tuned"),
        )

    return ComparisonResult(
        documents=len(contracts),
        clause_types=clause_types,
        zero_shot=zero_shot,
        fine_tuned=fine_tuned_result,
        training_result=training_result,
    )


# ----------------------------------------------------------------------- #
# Result save / plot helpers                                               #
# ----------------------------------------------------------------------- #

def _save_csv(result: "ComparisonResult"):
    rows = []
    for ct in result.clause_types:
        for key, _, arm in _arms(result):
            m = arm.metrics.get(ct)
            if m:
                for metric in ("precision", "recall", "f1", "accuracy"):
                    rows.append({"clause_type": ct, "method": key, "metric": metric,
                                 "value": getattr(m, metric)})
    pd.DataFrame(rows).to_csv(os.path.join(config.paths.outputs_dir, "comparison_metrics.csv"), index=False)


def _models_table(result: "ComparisonResult") -> str:
    arms = _arms(result)
    md = "|  | " + " | ".join(label for _, label, _ in arms) + " |\n"
    md += "|---|" + "---|" * len(arms) + "\n"
    rows = [
        ("Model", lambda k, a: a.model),
        ("Contracts", lambda k, a: str(a.stats.documents)),
        ("Avg latency / contract", lambda k, a: f"{a.stats.avg_latency_ms:.0f} ms"),
        ("Min / max latency", lambda k, a: f"{a.stats.min_latency_ms:.0f} / {a.stats.max_latency_ms:.0f} ms"),
        ("Cost / contract", lambda k, a: _usd(a.stats.avg_cost_usd)),
        ("LLM calls", lambda k, a: str(a.calls) if k == "zero_shot" else "none"),
    ]
    for name, cell in rows:
        md += f"| {name} | " + " | ".join(cell(k, a) for k, _, a in arms) + " |\n"
    return md


def _aggregate_table(result: "ComparisonResult") -> str:
    md = "|  | Precision | Recall | F1 | Accuracy |\n|---|---|---|---|---|\n"
    for _, label, arm in _arms(result):
        agg = aggregate_metrics(arm.metrics)
        if not agg:
            md += f"| {label} | no result | no result | no result | no result |\n"
            continue
        md += (f"| {label} | {agg['avg_precision']:.4f} | {agg['avg_recall']:.4f} | "
               f"{agg['avg_f1']:.4f} | {agg['avg_accuracy']:.4f} |\n")
    return md


def _notes(result: "ComparisonResult") -> str:
    notes = []
    if result.fine_tuned is None:
        notes.append("The fine-tuned arm did not run, so this report covers the zero-shot LLM only.")
    if not result.zero_shot.metrics:
        notes.append("Every zero-shot LLM call failed, so that arm has no result.  Check the "
                     "provider, model, and API key.")
    elif result.fine_tuned.stats.avg_cost_usd is None:
        notes.append("The fine-tuned arm is not priced.  Its latency is measured compute time, "
                     "and FT_COST_PER_HOUR turns that time into a cost.")
    failed = {ct: n for ct, n in result.zero_shot.failures.items() if n}
    if failed:
        notes.append(f"Zero-shot decisions left out after failed calls: {failed}.")
    return "\n".join(f"- {n}" for n in notes) + ("\n" if notes else "")


def _save_summary(result: "ComparisonResult"):
    md = (f"# Contract Clause Classifier Comparison\n\n"
          f"{result.documents} test contracts, {len(result.clause_types)} clause types.\n\n"
          f"## Models\n\n{_models_table(result)}\n"
          f"## Aggregate metrics\n\n{_aggregate_table(result)}\n{_notes(result)}")
    with open(os.path.join(config.paths.outputs_dir, "summary.md"), "w") as f:
        f.write(md)


def _save_report(result: "ComparisonResult"):
    md = (f"# Contract Clause Classification Report\n\n"
          f"Both methods decide, for each test contract and clause type, whether the clause "
          f"appears anywhere in the contract.  Labels come from CUAD answer spans.\n\n"
          f"## Data\n\n"
          f"- Test contracts: {result.documents}\n"
          f"- Clause types: {', '.join(result.clause_types)}\n\n"
          f"## Models and measured cost\n\n{_models_table(result)}\n"
          f"## Aggregate metrics\n\n{_aggregate_table(result)}\n"
          f"## Per-clause comparison\n\n"
          f"| Clause Type | Method | Precision | Recall | F1 | Accuracy | TP | FP | TN | FN |\n"
          f"|---|---|---|---|---|---|---|---|---|---|\n")
    for ct in result.clause_types:
        for _, label, arm in _arms(result):
            m = arm.metrics.get(ct)
            if m:
                md += (f"| {ct} | {label} | {m.precision:.4f} | {m.recall:.4f} | {m.f1:.4f} | "
                       f"{m.accuracy:.4f} | {m.true_positives} | {m.false_positives} | "
                       f"{m.true_negatives} | {m.false_negatives} |\n")
    md += (f"\n## Configuration\n\n"
           f"- Zero-shot: {config.llm.provider}/{config.llm.model}, chunks of "
           f"{config.llm.chunk_chars} characters overlapping by {config.llm.chunk_overlap}\n"
           f"- Fine-tuned: {config.training.model_name}, windows of {config.training.max_length} "
           f"tokens with a stride of {config.training.window_stride}, threshold "
           f"{config.training.threshold}, {config.training.num_epochs} epochs, "
           f"learning rate {config.training.learning_rate}, batch size {config.training.batch_size}\n")
    if result.training_result is not None:
        tr = result.training_result
        md += (f"- Training: {tr.training_time_seconds:.1f} s on {tr.train_windows} windows, "
               f"with {tr.val_windows} validation windows\n")
    notes = _notes(result)
    if notes:
        md += f"\n## Notes\n\n{notes}"
    with open(os.path.join(config.paths.outputs_dir, "comparison_report.md"), "w") as f:
        f.write(md)


def _plot_results(result: "ComparisonResult"):
    arms = _arms(result)
    clause_types = [ct for ct in result.clause_types if all(ct in a.metrics for _, _, a in arms)]
    if not clause_types:
        return
    metrics_list = ["precision", "recall", "f1", "accuracy"]
    colours = {"zero_shot": "#2ecc71", "fine_tuned": "#3498db"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    x = np.arange(len(clause_types))
    width = 0.8 / len(arms)

    for idx, metric in enumerate(metrics_list):
        ax = axes[idx]
        for i, (key, label, arm) in enumerate(arms):
            values = [getattr(arm.metrics[ct], metric) for ct in clause_types]
            ax.bar(x + (i - (len(arms) - 1) / 2) * width, values, width,
                   label=label, color=colours[key], alpha=0.8)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(clause_types, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(config.paths.outputs_dir, "comparison_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------- #
# Interactive run flow                                                      #
# ----------------------------------------------------------------------- #

def _interactive_run(mode: str, max_samples: Optional[int] = None, ask: bool = True):
    """Run a comparison and save its outputs.

    mode is "quick", "full", or "evaluate".  With ask=False (command-line
    flags) nothing is prompted and max_samples is used as given.
    """
    clause_types = list(config.data.clause_types)
    if ask:
        if not Confirm.ask("  Keep [bold]current clause types[/]?", default=True):
            _configure_clause_types_menu()
            clause_types = list(config.data.clause_types)
        if mode != "quick":
            ms_val = Prompt.ask(
                "  Maximum test contracts (leave blank for the whole test split)",
                default="",
            )
            max_samples = int(ms_val) if ms_val.isdigit() and int(ms_val) > 0 else None

    result = _run_comparison(max_samples=max_samples, clause_types=clause_types, mode=mode)
    if result is None:
        return

    console.print()
    _show_results_table(result)
    _show_cost_latency(result)

    summary_table = Table(title="Aggregate Summary", show_header=True)
    summary_table.add_column("Metric")
    for _, label, _ in _arms(result):
        summary_table.add_column(label)
    aggregates = [aggregate_metrics(arm.metrics) for _, _, arm in _arms(result)]
    for key, label in [("avg_precision", "Precision"), ("avg_recall", "Recall"),
                       ("avg_f1", "F1 Score"), ("avg_accuracy", "Accuracy")]:
        summary_table.add_row(label, *[f"{agg[key]:.4f}" if agg else "no result" for agg in aggregates])
    console.print(summary_table)

    os.makedirs(config.paths.outputs_dir, exist_ok=True)
    _save_csv(result)
    _save_summary(result)
    _save_report(result)
    _plot_results(result)

    tree = Tree("[bold]Output files[/]")
    for fname in sorted(os.listdir(config.paths.outputs_dir)):
        tree.add(f"[green]{fname}[/]")

    console.print()
    console.print(Panel(
        f"[green]✓ Results saved to:[/]\n  [dim]{config.paths.outputs_dir}[/]",
        border_style="green",
    ))
    console.print(tree)


# ----------------------------------------------------------------------- #
# Entry point                                                               #
# ----------------------------------------------------------------------- #

def main():
    """Interactive CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare contract clause classifiers (interactive by default)",
    )
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Train, then evaluate on at most N test contracts")
    parser.add_argument("--quick-test", action="store_true",
                        help=f"Zero-shot on {QUICK_TEST_CONTRACTS} test contracts, plus any saved model")
    parser.add_argument("--output", type=str, default=None, help="Custom output directory for results")

    args = parser.parse_args()

    if args.output:
        config.paths.outputs_dir = args.output
        os.makedirs(args.output, exist_ok=True)

    use_menu = not (args.quick_test or args.max_samples is not None)

    if use_menu:
        _header()

    while use_menu:
        choice = _main_menu()

        if choice == 1:
            console.print(f"[cyan]Running quick test ({QUICK_TEST_CONTRACTS} contracts)…[/]")
            _interactive_run("quick")

        elif choice == 2:
            if not Confirm.ask("Start full comparison (downloads data, trains model)?"):
                continue
            _interactive_run("full")

        elif choice == 3:
            ms_val = Prompt.ask("  Maximum training contracts (leave blank for the whole train split)",
                                default="")
            max_train = int(ms_val) if ms_val.isdigit() and int(ms_val) > 0 else None
            _train_fine_tuned(list(config.data.clause_types), max_train)

        elif choice == 4:
            _interactive_run("evaluate")

        elif choice == 5:
            _configure_menu()

        elif choice == 6:
            _view_config()

        elif choice == 7:
            _export_import_menu()

        elif choice == 8:
            console.print("[bold yellow]Goodbye![/]")
            break

    # Non-interactive mode (legacy flags)
    if not use_menu:
        if args.quick_test:
            console.print("[cyan]Running quick test (--quick-test flag)…[/]")
            _interactive_run("quick", ask=False)
        else:
            console.print(f"[cyan]Running full comparison (max-samples={args.max_samples})…[/]")
            _interactive_run("full", max_samples=args.max_samples, ask=False)


if __name__ == "__main__":
    main()

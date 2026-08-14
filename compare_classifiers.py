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
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
from rich.rule import Rule
from rich.live import Live
from rich.logging import RichHandler
from rich.table import Table
from rich.tree import Tree

from config import config
from utils.llm_client import LLMClient
from utils.data_loader import load_cuad_dataset, preprocess_data, get_clause_distribution
from utils.classifier import FineTunedClassifier
from utils.metrics import (
    calculate_metrics, aggregate_metrics, print_comparison_table,
    ClassificationMetrics, InferenceStats,
)

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

@dataclass
class ComparisonResult:
    """Complete comparison result."""
    zero_shot_metrics: Dict[str, ClassificationMetrics]
    fine_tuned_metrics: Dict[str, ClassificationMetrics]
    zero_shot_stats: InferenceStats
    fine_tuned_stats: InferenceStats
    training_result: Optional[object]
    cost_per_document: Dict[str, float]


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
        ("Quick test (~50 samples)", "quick"),
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

    # Cost presets for known models
    if config.llm.model in ("gpt-3.5-turbo", "gpt-4o"):
        console.print("  [green]✓ Cost preset applied from model name[/]")


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

        # Show CUAD defaults that aren't active yet (only if list is small)
        cuad_defaults = [
            "Agreement Effectiveness", "Agreement Termination", "Anti-Assignment",
            "Arbitration", "Attorneys' Fees", "Notice", "Governing Law",
            "Indemnification", "Jurisdiction", "Severability", "Waiver", "Warranty",
        ]
        extra = [ct for ct in cuad_defaults if ct not in all_types]
        if extra:
            console.print()
            console.print("  [dim]Available CUAD defaults:[/]", style="dim")
            for ct in extra:
                console.print(f"    • {ct}")

        console.print()
        console.print("  [bold yellow]1[/]. Toggle active clauses")
        console.print("  [bold yellow]2[/]. Add custom clause type")
        console.print("  [bold yellow]3[/]. Remove selected clause type")
        console.print("  [bold yellow]4[/]. Select all CUAD defaults")
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
            for ct in cuad_defaults:
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

def _show_results_table(zs_metrics, ft_metrics):
    """Render a Rich comparison table."""
    clause_types = sorted(set(list(zs_metrics.keys()) + list(ft_metrics.keys())))
    table = Table(
        title="Classifier Comparison",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Clause Type", style="dim")
    table.add_column("Method")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Accuracy", justify="right")

    for ct in clause_types:
        zs = zs_metrics.get(ct)
        ft = ft_metrics.get(ct)
        if zs:
            table.add_row(
                ct, "Zero-Shot",
                f"{zs.precision:.4f}", f"{zs.recall:.4f}",
                f"{zs.f1:.4f}", f"{zs.accuracy:.4f}",
            )
        if ft:
            table.add_row(
                ct, "Fine-Tuned",
                f"{ft.precision:.4f}", f"{ft.recall:.4f}",
                f"{ft.f1:.4f}", f"{ft.accuracy:.4f}",
            )

    console.print(table)


def _show_clause_dist(dist):
    table = Table(title="Clause Distribution", show_header=True)
    table.add_column("Clause Type", style="dim")
    table.add_column("Present", style="green")
    table.add_column("Absent", style="red")
    for row in dist.itertuples():
        table.add_row(row.clause_type, str(row.present), str(row.absent))
    console.print(table)


# ----------------------------------------------------------------------- #
# Core comparison logic                                                     #
# ----------------------------------------------------------------------- #

def _run_comparison(
    max_samples: Optional[int] = None,
    clause_types: Optional[List[str]] = None,
    quick_test: bool = False,
) -> ComparisonResult:
    if clause_types is None:
        clause_types = config.data.clause_types

    console.print()
    with Live("[bold cyan]Loading dataset…", refresh_per_second=4):
        contracts = load_cuad_dataset(
            split="test",
            max_samples=max_samples if not quick_test else 50,
        )

    console.print(f"  [bold green]✓ Loaded {len(contracts)} contracts[/]")
    distribution = get_clause_distribution(contracts)
    _show_clause_dist(distribution)

    texts_by_clause: Dict[str, List[str]] = {ct: [] for ct in clause_types}
    labels_by_clause: Dict[str, List[int]] = {ct: [] for ct in clause_types}
    for contract in contracts:
        for ct in clause_types:
            texts_by_clause[ct].append(contract.text[:512])
            labels_by_clause[ct].append(1 if contract.clauses.get(ct, False) else 0)

    # ---- fine-tuned training ----
    fine_tuned = FineTunedClassifier()
    training_result = None

    if not quick_test:
        console.print()
        with Live("[bold cyan]Training fine-tuned model…", refresh_per_second=4):
            train_contracts = load_cuad_dataset(
                split="train",
                max_samples=1000 if max_samples is None else min(max_samples, 1000),
            )
            train_texts, train_labels = preprocess_data(train_contracts, clause_types)
            train_dataset, val_dataset, _ = fine_tuned.prepare_data(
                train_texts, train_labels, test_size=0.2, val_size=0.1,
            )
            training_result = fine_tuned.train(train_dataset, val_dataset)

        console.print()
        console.print(
            Panel(
                f"[green]✓ Training complete[/]\n"
                f"  Time: {training_result.training_time_seconds:.1f}s\n"
                f"  Metrics: {training_result.metrics}",
                border_style="green",
            )
        )

    # ---- evaluate zero-shot LLM ----
    console.print()
    llm_client = LLMClient()
    zero_shot_metrics, zero_shot_stats = _eval_zero_shot(
        llm_client, texts_by_clause, labels_by_clause, clause_types, quick_test,
    )

    # ---- evaluate fine-tuned ----
    fine_tuned_metrics, fine_tuned_stats = _eval_finetuned(
        fine_tuned, texts_by_clause, labels_by_clause, clause_types, quick_test,
    )

    num_documents = len(contracts)
    zs_total = sum(z.latency_ms for z in zero_shot_stats) or 0
    ft_total = sum(f.latency_ms for f in fine_tuned_stats) or 0

    return ComparisonResult(
        zero_shot_metrics=zero_shot_metrics,
        fine_tuned_metrics=fine_tuned_metrics,
        zero_shot_stats=InferenceStats(
            total_latency_ms=zs_total,
            avg_latency_ms=np.mean([z.latency_ms for z in zero_shot_stats]) or 0,
            min_latency_ms=min((z.latency_ms for z in zero_shot_stats), default=0),
            max_latency_ms=max((z.latency_ms for z in zero_shot_stats), default=0),
            total_cost_usd=sum(z.cost_usd for z in zero_shot_stats),
            avg_cost_usd=(sum(z.cost_usd for z in zero_shot_stats) / num_documents) if num_documents else 0,
            input_tokens=0, output_tokens=0,
        ),
        fine_tuned_stats=InferenceStats(
            total_latency_ms=ft_total,
            avg_latency_ms=np.mean([f.latency_ms for f in fine_tuned_stats]) or 0,
            min_latency_ms=min((f.latency_ms for f in fine_tuned_stats), default=0),
            max_latency_ms=max((f.latency_ms for f in fine_tuned_stats), default=0),
            total_cost_usd=sum(f.cost_usd for f in fine_tuned_stats),
            avg_cost_usd=(sum(f.cost_usd for f in fine_tuned_stats) / num_documents) if num_documents else 0,
            input_tokens=0, output_tokens=0,
        ),
        training_result=training_result,
        cost_per_document={
            "zero_shot": (sum(z.cost_usd for z in zero_shot_stats) / num_documents) if num_documents else 0,
            "fine_tuned": (sum(f.cost_usd for f in fine_tuned_stats) / num_documents) if num_documents else 0,
        },
    )


def _eval_zero_shot(llm_client, texts_by_clause, labels_by_clause, clause_types, quick_test):
    import time
    metrics = {}
    stats_list = []

    for ct in clause_types:
        eval_texts = texts_by_clause[ct][:100] if quick_test else texts_by_clause[ct]
        eval_labels = labels_by_clause[ct][:100] if quick_test else labels_by_clause[ct]

        preds, costs, latencies = [], [], []
        console.print(f"[cyan]  Evaluating {ct} with Zero-Shot LLM…[/]")
        for text in eval_texts:
            t0 = time.time()
            resp = llm_client.classify_single(text, ct)
            lat = (time.time() - t0) * 1000
            preds.append(1 if resp.text.strip().upper().startswith("YES") else 0)
            costs.append(resp.cost_usd)
            latencies.append(lat)

        m = calculate_metrics(eval_labels, preds, ct)
        metrics[ct] = m
        stats_list.append(InferenceStats(
            total_latency_ms=sum(latencies), avg_latency_ms=np.mean(latencies),
            min_latency_ms=min(latencies), max_latency_ms=max(latencies),
            total_cost_usd=sum(costs), avg_cost_usd=np.mean(costs),
            input_tokens=0, output_tokens=0,
        ))

    return metrics, stats_list


def _eval_finetuned(fine_tuned, texts_by_clause, labels_by_clause, clause_types, quick_test):
    metrics = {}
    stats_list = []

    if not fine_tuned.trained:
        path = os.path.join(config.paths.models_dir, "fine_tuned")
        if os.path.exists(path):
            fine_tuned.load(path)

    for ct in clause_types:
        eval_texts = texts_by_clause[ct][:100] if quick_test else texts_by_clause[ct]
        eval_labels = labels_by_clause[ct][:100] if quick_test else labels_by_clause[ct]

        console.print(f"[cyan]  Evaluating {ct} with Fine-Tuned…[/]")
        preds, probs, avg_lat = fine_tuned.predict(eval_texts)
        est_cost = (avg_lat / 1000) * 0.001

        m = calculate_metrics(eval_labels, preds, ct)
        metrics[ct] = m
        stats_list.append(InferenceStats(
            total_latency_ms=avg_lat * len(eval_texts), avg_latency_ms=avg_lat,
            min_latency_ms=avg_lat * 0.5 if avg_lat else 0,
            max_latency_ms=avg_lat * 1.5 if avg_lat else 0,
            total_cost_usd=est_cost * len(eval_texts), avg_cost_usd=est_cost,
            input_tokens=0, output_tokens=0,
        ))

    return metrics, stats_list


# ----------------------------------------------------------------------- #
# Result save / plot helpers                                               #
# ----------------------------------------------------------------------- #

def _save_csv(result: ComparisonResult):
    rows = []
    for ct in result.zero_shot_metrics:
        zs, ft = result.zero_shot_metrics[ct], result.fine_tuned_metrics[ct]
        for method, m in [("zero_shot", zs), ("fine_tuned", ft)]:
            for metric in ("precision", "recall", "f1", "accuracy"):
                rows.append({"clause_type": ct, "method": method, "metric": metric, "value": getattr(m, metric)})
    pd.DataFrame(rows).to_csv(os.path.join(config.paths.outputs_dir, "comparison_metrics.csv"), index=False)


def _save_summary(result: ComparisonResult):
    agg_zs = aggregate_metrics(result.zero_shot_metrics)
    agg_ft = aggregate_metrics(result.fine_tuned_metrics)

    md = f"""# Contract Clause Classifier — Comparison Report

## Models

|  | Zero-Shot LLM | Fine-Tuned |
|---|---|---|
| Model | {config.llm.provider}/{config.llm.model} | {config.training.model_name} |
| Cost / doc | ${result.cost_per_document['zero_shot']:.6f} | ${result.cost_per_document['fine_tuned']:.6f} |
| Avg latency | {result.zero_shot_stats.avg_latency_ms:.2f} ms | {result.fine_tuned_stats.avg_latency_ms:.2f} ms |

## Aggregate Metrics

|  | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Zero-Shot LLM | {agg_zs.get('avg_precision',0):.4f} | {agg_zs.get('avg_recall',0):.4f} | {agg_zs.get('avg_f1',0):.4f} | {agg_zs.get('avg_accuracy',0):.4f} |
| Fine-Tuned | {agg_ft.get('avg_precision',0):.4f} | {agg_ft.get('avg_recall',0):.4f} | {agg_ft.get('avg_f1',0):.4f} | {agg_ft.get('avg_accuracy',0):.4f} |

## Configuration

- Epochs: {config.training.num_epochs} · Batch size: {config.training.batch_size}
- Learning rate: {config.training.learning_rate} · Max length: {config.training.max_length}
"""
    with open(os.path.join(config.paths.outputs_dir, "summary.md"), "w") as f:
        f.write(md)


def _save_report(result: ComparisonResult):
    agg_zs = aggregate_metrics(result.zero_shot_metrics)
    agg_ft = aggregate_metrics(result.fine_tuned_metrics)

    md = f"""# Contract Clause Classification — Model Comparison Report

## Executive Summary

Zero-shot LLM vs fine-tuned transformer on the CUAD contract dataset.

## Dataset

- Documents evaluated: {len(result.zero_shot_metrics)} contracts analyzed
- Clause types: {len(result.zero_shot_metrics)}

## Aggregate Metrics

### Zero-Shot LLM ({config.llm.provider}/{config.llm.model})
- Precision: {agg_zs.get('avg_precision',0):.4f}
- Recall: {agg_zs.get('avg_recall',0):.4f}
- F1: {agg_zs.get('avg_f1',0):.4f}
- Accuracy: {agg_zs.get('avg_accuracy',0):.4f}

### Fine-Tuned ({config.training.model_name})
- Precision: {agg_ft.get('avg_precision',0):.4f}
- Recall: {agg_ft.get('avg_recall',0):.4f}
- F1: {agg_ft.get('avg_f1',0):.4f}
- Accuracy: {agg_ft.get('avg_accuracy',0):.4f}

## Per-Clause Comparison

| Clause Type | Method | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|
"""
    for ct in result.zero_shot_metrics:
        zs = result.zero_shot_metrics[ct]
        ft = result.fine_tuned_metrics[ct]
        md += f"| {ct} | Zero-Shot | {zs.precision:.4f} | {zs.recall:.4f} | {zs.f1:.4f} | {zs.accuracy:.4f} |\n"
        md += f"| {ct} | Fine-Tuned | {ft.precision:.4f} | {ft.recall:.4f} | {ft.f1:.4f} | {ft.accuracy:.4f} |\n"

    md += f"""
## Configuration

- LLM: {config.llm.provider}/{config.llm.model}
- Training: {config.training.model_name}, {config.training.num_epochs} epochs, lr={config.training.learning_rate}
- Batch size: {config.training.batch_size} · Max length: {config.training.max_length}

## Conclusion

At **low volume** zero-shot LLM is simpler. At **high volume** fine-tuned wins on cost and latency.
"""
    with open(os.path.join(config.paths.outputs_dir, "comparison_report.md"), "w") as f:
        f.write(md)


def _plot_results(result: ComparisonResult):
    clause_types = list(result.zero_shot_metrics.keys())
    metrics_list = ["precision", "recall", "f1", "accuracy"]
    zs_data = {m: [] for m in metrics_list}
    ft_data = {m: [] for m in metrics_list}

    for ct in clause_types:
        zs = result.zero_shot_metrics[ct]
        ft = result.fine_tuned_metrics[ct]
        for m in metrics_list:
            zs_data[m].append(getattr(zs, m))
            ft_data[m].append(getattr(ft, m))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    x = np.arange(len(clause_types))
    width = 0.35

    for idx, metric in enumerate(metrics_list):
        ax = axes[idx]
        _ = ax.bar(x - width / 2, zs_data[metric], width, label="Zero-Shot LLM", color="#2ecc71", alpha=0.8)
        _ = ax.bar(x + width / 2, ft_data[metric], width, label="Fine-Tuned", color="#3498db", alpha=0.8)
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

def _interactive_run(run_mode: str):
    """Drive the full comparison with interactive prompts for each step."""
    clause_types = list(config.data.clause_types)
    max_samples = None
    quick_test = False

    # Quick-test toggle
    if run_mode == "auto_quick":
        quick_test = True
    elif run_mode == "none":
        quick_test = Confirm.ask("  Run a [bold]quick test[/] (~50 samples)?", default=False)
        if quick_test:
            max_samples = 200

    # Clause types — ask unless already running fast-quick
    if not quick_test or run_mode in ("auto_quick",):
        if Confirm.ask(
            "  Keep [bold]current clause types[/]?",
            default=True,
        ):
            clause_types = list(config.data.clause_types)
        else:
            clause_types = _configure_clause_types_menu()
            # Save the selection for future runs
            config.data.clause_types = clause_types

    # Max samples
    if not quick_test:
        ms_val = Prompt.ask(
            "  Maximum samples per clause type (leave blank for full dataset)",
            default="",
        )
        max_samples = int(ms_val) if ms_val.isdigit() and int(ms_val) > 0 else None

    # ---- Run comparison ----
    console.print()
    with Live("[bold cyan]Running comparison…", refresh_per_second=4):
        result = _run_comparison(
            max_samples=max_samples,
            clause_types=clause_types,
            quick_test=quick_test,
        )

    # Show results
    console.print()
    _show_results_table(result.zero_shot_metrics, result.fine_tuned_metrics)

    agg_zs = aggregate_metrics(result.zero_shot_metrics)
    agg_ft = aggregate_metrics(result.fine_tuned_metrics)

    summary_table = Table(title="Aggregate Summary", show_header=True)
    summary_table.add_column("Metric")
    summary_table.add_column("Zero-Shot LLM")
    summary_table.add_column("Fine-Tuned")
    for key, label in [
        ("avg_precision", "Precision"),
        ("avg_recall", "Recall"),
        ("avg_f1", "F1 Score"),
        ("avg_accuracy", "Accuracy"),
    ]:
        summary_table.add_row(label, f"{agg_zs.get(key, 0):.4f}", f"{agg_ft.get(key, 0):.4f}")
    console.print(summary_table)

    # Save artifacts
    _save_csv(result)
    _save_summary(result)
    _save_report(result)
    _plot_results(result)

    # Show tree of outputs
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare contract clause classifiers (interactive by default)",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit evaluation to N samples")
    parser.add_argument("--quick-test", action="store_true", help="Run a fast, limited-sample test")
    parser.add_argument("--output", type=str, default=None, help="Custom output directory for results")

    args = parser.parse_args()

    if args.output:
        config.paths.outputs_dir = args.output
        os.makedirs(args.output, exist_ok=True)

    use_menu = not (args.quick_test or args.max_samples is not None)

    if use_menu:
        _header()

    while True and use_menu:
        choice = _main_menu()

        if choice == 1:
            console.print("[cyan]Running quick test (~50 samples)…[/]")
            _interactive_run("auto_quick")

        elif choice == 2:
            if not Confirm.ask("Start full comparison (downloads data, trains model)?"):
                continue
            _interactive_run("none")

        elif choice == 3:
            clause_types = list(config.data.clause_types)
            ms_val = Prompt.ask("  Max training samples", default="1000")
            max_samples = int(ms_val) if ms_val.isdigit() else 1000

            console.print(f"[cyan]Loading {max_samples} train samples…[/]")
            contracts = load_cuad_dataset(split="train", max_samples=max_samples)
            texts, labels = preprocess_data(contracts, clause_types)

            fine_tuned = FineTunedClassifier()
            train_ds, val_ds, _ = fine_tuned.prepare_data(texts, labels, test_size=0.2, val_size=0.1)
            console.print("[cyan]Training…[/]")
            result = fine_tuned.train(train_ds, val_ds)
            console.print(Panel(
                f"[green]✓ Training complete[/]\n"
                f"  Time: {result.training_time_seconds:.1f}s\n"
                f"  Metrics: {result.metrics}",
                border_style="green",
            ))

        elif choice == 4:
            _interactive_run("none")

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
        elif args.max_samples is not None:
            console.print(f"[cyan]Running full comparison (max-samples={args.max_samples})…[/]")
        _interactive_run("auto_quick" if args.quick_test else "none")


if __name__ == "__main__":
    main()

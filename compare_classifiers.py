"""
Command line for the contract clause classifier comparison.

Each arm runs on its own and saves its results, so the zero-shot arm can run
on a laptop with an API key and the fine-tuned arm on a machine with a GPU.
The compare command then writes the report from the saved results.

    python compare_classifiers.py estimate
    python compare_classifiers.py zero-shot --max-contracts 5
    python compare_classifiers.py fine-tune
    python compare_classifiers.py compare
    python compare_classifiers.py show-config
"""

import argparse
import datetime
import glob
import logging
import os
import re
import sys

from config import CUAD_CATEGORIES, EFFORTS, config, mask_secret
from utils.data_loader import load_cuad_dataset
from utils.estimate import DEFAULT_MODELS as ESTIMATE_MODELS
from utils.estimate import DEFAULT_OUTPUT_TOKENS, estimate, format_estimate
from utils.evaluation import ArmResult, evaluate_fine_tuned, evaluate_zero_shot
from utils.llm_client import PROVIDERS, FatalLLMError, make_client
from utils.prompts import MODES, PROMPT_VERSION
from utils.report import models_table, notes, per_clause_table, write_report

logger = logging.getLogger("compare_classifiers")

# Where each provider's SDK looks for a key when LLM_API_KEY is unset
KEY_VARIABLES = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "openai": ("OPENAI_API_KEY",),
}

# Training settings that fine-tune takes as flags: flag, TrainingConfig field, type, help
TRAINING_FLAGS = [
    ("--base-model", "model_name", str, "Hugging Face model to fine-tune"),
    ("--epochs", "num_epochs", int, "training epochs"),
    ("--batch-size", "batch_size", int, "training and inference batch size"),
    ("--learning-rate", "learning_rate", float, "learning rate"),
    ("--max-length", "max_length", int, "window length in tokens"),
    ("--weight-decay", "weight_decay", float, "AdamW weight decay"),
    ("--warmup-steps", "warmup_steps", int, "learning-rate warmup steps"),
    ("--eval-steps", "eval_steps", int, "evaluation frequency during training"),
    ("--save-steps", "save_steps", int, "checkpoint frequency"),
    ("--window-stride", "window_stride", int, "tokens shared by consecutive windows"),
    ("--negative-ratio", "negative_window_ratio", float, "clause-free windows per clause window"),
    ("--threshold", "threshold", float, "score at which a window counts as having a clause"),
    ("--cost-per-hour", "cost_per_hour", float, "USD per hour of the inference machine"),
]


def clause_type(name: str) -> str:
    """argparse type for a CUAD category, matched ignoring case and spelled as CUAD spells it."""
    by_key = {category.casefold(): category for category in CUAD_CATEGORIES}
    try:
        return by_key[name.strip().casefold()]
    except KeyError:
        raise argparse.ArgumentTypeError(
            f"{name!r} is not a CUAD category.  Valid categories: {', '.join(CUAD_CATEGORIES)}"
        ) from None


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--clause-types",
        nargs="+",
        type=clause_type,
        metavar="TYPE",
        help="CUAD categories to classify, by default the 12 in config.py",
    )
    common.add_argument("--output", metavar="DIR", help="directory for results, default outputs/")
    common.add_argument("--cuad", metavar="PATH", help="a CUAD_v1.json to read instead of data/")
    common.add_argument("-v", "--verbose", action="store_true", help="log debug detail")

    chunking = argparse.ArgumentParser(add_help=False)
    chunking.add_argument(
        "--chunk-chars", type=positive_int, help="characters of contract per call"
    )
    chunking.add_argument("--chunk-overlap", type=int, help="characters shared by adjacent chunks")

    parser = argparse.ArgumentParser(
        prog="compare_classifiers.py",
        description="Compare a zero-shot LLM with a fine-tuned transformer on CUAD contracts.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    est = sub.add_parser(
        "estimate",
        parents=[common, chunking],
        help="estimate calls, tokens, and cost of the zero-shot arm, with no API call",
    )
    est.add_argument("--text-file", nargs="+", metavar="FILE", help="contract text files to use")
    est.add_argument("--max-contracts", type=positive_int, help="test contracts to count")
    est.add_argument(
        "--output-tokens",
        type=int,
        default=DEFAULT_OUTPUT_TOKENS,
        help=f"output tokens assumed per call, default {DEFAULT_OUTPUT_TOKENS}",
    )
    est.add_argument("--models", nargs="+", default=list(ESTIMATE_MODELS), help="models to price")

    zs = sub.add_parser(
        "zero-shot",
        parents=[common, chunking],
        help="run the zero-shot LLM arm on the test split and save its results",
    )
    zs.add_argument("--provider", choices=PROVIDERS, help="default anthropic")
    zs.add_argument("--model", help="default claude-opus-5-5 for anthropic")
    zs.add_argument("--mode", choices=MODES, help="prompt mode, default multi")
    zs.add_argument("--effort", choices=EFFORTS, help="thinking depth for Opus and Sonnet")
    zs.add_argument("--max-tokens", type=positive_int, help="cap on output tokens per call")
    zs.add_argument("--max-contracts", type=positive_int, help="test contracts to ask about")
    zs.add_argument("--concurrency", type=positive_int, help="contracts asked about at once")
    zs.add_argument("--no-cache", action="store_true", help="neither read nor write the cache")

    ft = sub.add_parser(
        "fine-tune",
        parents=[common],
        help="train the fine-tuned arm, score the test split, and save its results",
    )
    which = ft.add_mutually_exclusive_group()
    which.add_argument("--skip-training", action="store_true", help="score the saved model")
    which.add_argument("--skip-evaluation", action="store_true", help="train and save only")
    ft.add_argument("--max-train-contracts", type=positive_int, help="train contracts to use")
    ft.add_argument("--max-contracts", type=positive_int, help="test contracts to score")
    ft.add_argument("--model-dir", help="where the model is saved, default models/fine_tuned")
    for flag, field_name, kind, text in TRAINING_FLAGS:
        ft.add_argument(flag, dest=field_name, type=kind, help=text)

    cmp = sub.add_parser(
        "compare", parents=[common], help="write the comparison report from saved arm results"
    )
    cmp.add_argument(
        "arm_files",
        nargs="*",
        metavar="ARM_FILE",
        help="saved arm results, by default every one in the output directory",
    )

    sub.add_parser(
        "show-config", parents=[common], help="show the active settings, with the key masked"
    )
    return parser


def apply_common(args) -> None:
    if args.clause_types:
        config.data.clause_types = list(dict.fromkeys(args.clause_types))
    if args.output:
        config.paths.outputs_dir = os.path.abspath(args.output)
    if args.cuad:
        config.data.cuad_path = args.cuad


def apply_llm(args) -> None:
    names = ("provider", "model", "mode", "effort", "max_tokens", "concurrency")
    for name in (*names, "chunk_chars", "chunk_overlap"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(config.llm, name, value)


def apply_training(args) -> None:
    for _, field_name, _, _ in TRAINING_FLAGS:
        value = getattr(args, field_name, None)
        if value is not None:
            setattr(config.training, field_name, value)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _progress(label: str):
    def report(done: int, total: int) -> None:
        logger.info("%s: %d of %d contracts", label, done, total)

    return report


def _save(arm: ArmResult, name: str) -> str:
    path = os.path.join(config.paths.outputs_dir, name)
    arm.save(path)
    print(f"## {arm.label}\n\n{models_table([arm])}\n{per_clause_table([arm])}")
    for note in notes([arm]):
        if "nothing to compare" not in note:
            print(f"- {note}")
    print(f"\nSaved {path}")
    return path


def api_key_source(provider: str) -> tuple[str | None, str | None]:
    """The key in effect for provider and the variable it came from, or (None, None)."""
    if config.llm.api_key:
        return config.llm.api_key, "LLM_API_KEY"
    for name in KEY_VARIABLES.get(provider, ()):
        if os.environ.get(name):
            return os.environ[name], name
    return None, None


def cmd_estimate(args) -> int:
    apply_llm(args)
    if args.text_file:
        texts = []
        for name in args.text_file:
            with open(name, encoding="utf-8") as fh:
                texts.append(fh.read())
        source = ", ".join(os.path.basename(name) for name in args.text_file)
    else:
        contracts = load_cuad_dataset("test", max_samples=args.max_contracts)
        texts = [c.text for c in contracts]
        source = "CUAD test split"
    est = estimate(
        texts,
        config.data.clause_types,
        config.llm.chunk_chars,
        config.llm.chunk_overlap,
        args.output_tokens,
    )
    print(format_estimate(est, source, args.models))
    return 0


def cmd_zero_shot(args) -> int:
    apply_llm(args)
    settings = config.llm
    if api_key_source(settings.provider)[0] is None:
        names = " or ".join(("LLM_API_KEY", *KEY_VARIABLES.get(settings.provider, ())))
        logger.error(
            "No API key for %s.  Set %s, in the shell or in .env.", settings.provider, names
        )
        return 2
    client = make_client(settings, use_cache=not args.no_cache)
    contracts = load_cuad_dataset("test", max_samples=args.max_contracts)
    logger.info(
        "Asking %s about %d test contracts with the %s-label prompt, %d at a time",
        settings.resolved_model,
        len(contracts),
        settings.mode,
        settings.concurrency,
    )
    arm = evaluate_zero_shot(
        client,
        contracts,
        config.data.clause_types,
        mode=settings.mode,
        chunk_chars=settings.chunk_chars,
        chunk_overlap=settings.chunk_overlap,
        concurrency=settings.concurrency,
        on_contract=_progress("Zero-shot"),
    )
    arm.settings.update(
        provider=settings.provider,
        effort=settings.effort,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        created_at=_now(),
    )
    _save(arm, f"zero_shot-{_safe(settings.resolved_model)}-{settings.mode}.json")
    return 0


def cmd_fine_tune(args) -> int:
    apply_training(args)
    from utils.classifier import FineTunedClassifier  # needs torch and transformers

    model_dir = args.model_dir or os.path.join(config.paths.models_dir, "fine_tuned")
    training = None
    if args.skip_training:
        if not os.path.exists(os.path.join(model_dir, "config.json")):
            logger.error(
                "No saved model in %s.  Train one first, without --skip-training.", model_dir
            )
            return 2
        classifier = FineTunedClassifier(model_name=model_dir)
        classifier.load(model_dir)
    else:
        train = load_cuad_dataset("train", max_samples=args.max_train_contracts)
        validation = load_cuad_dataset("validation")
        classifier = FineTunedClassifier(config.data.clause_types)
        logger.info("Training %s on %d contracts", config.training.model_name, len(train))
        training = classifier.train(train, validation, output_dir=model_dir)
        logger.info(
            "Trained in %.0f s on %d windows, with %d validation windows.  Saved to %s",
            training.training_time_seconds,
            training.train_windows,
            training.val_windows,
            model_dir,
        )
    if args.skip_evaluation:
        return 0

    contracts = load_cuad_dataset("test", max_samples=args.max_contracts)
    arm = evaluate_fine_tuned(
        classifier,
        contracts,
        config.data.clause_types,
        cost_per_hour=config.training.cost_per_hour,
        on_contract=_progress("Fine-tuned"),
    )
    tc = config.training
    arm.settings.update(
        max_length=tc.max_length,
        window_stride=tc.window_stride,
        epochs=tc.num_epochs,
        learning_rate=tc.learning_rate,
        batch_size=tc.batch_size,
        created_at=_now(),
    )
    if training is not None:
        arm.settings.update(
            training_seconds=round(training.training_time_seconds, 1),
            train_windows=training.train_windows,
            validation_windows=training.val_windows,
        )
    _save(arm, f"fine_tuned-{_safe(arm.model)}.json")
    return 0


def cmd_compare(args) -> int:
    out = config.paths.outputs_dir
    files = args.arm_files or sorted(
        glob.glob(os.path.join(out, "zero_shot-*.json"))
        + glob.glob(os.path.join(out, "fine_tuned-*.json"))
    )
    if not files:
        logger.error("No arm results in %s.  Run zero-shot or fine-tune first.", out)
        return 2
    arms = [ArmResult.load(path) for path in files]
    print(write_report(arms, out))
    print(f"Compared {', '.join(os.path.basename(f) for f in files)}.  Report written to {out}")
    return 0


def cmd_show_config(args) -> int:
    llm, tc = config.llm, config.training
    key, source = api_key_source(llm.provider)
    rows = [
        ("LLM provider", llm.provider),
        ("LLM model", llm.resolved_model),
        ("API key", f"{mask_secret(key)} from {source}" if key else mask_secret(None)),
        ("Prompt mode", f"{llm.mode}-label, prompt version {PROMPT_VERSION}"),
        ("Effort", llm.effort),
        ("Max tokens per call", llm.max_tokens),
        ("Temperature", "unset" if llm.temperature is None else llm.temperature),
        ("Chunks", f"{llm.chunk_chars:,} characters, overlapping by {llm.chunk_overlap:,}"),
        ("Concurrency", llm.concurrency),
        ("Tries per call", llm.max_attempts),
        ("Response cache", llm.cache_dir),
        ("Clause types", f"{len(config.data.clause_types)}: {', '.join(config.data.clause_types)}"),
        ("CUAD file", config.data.cuad_path or "data/CUAD_v1.json, or a download"),
        ("Fine-tuned model", tc.model_name),
        (
            "Epochs, batch size, learning rate",
            f"{tc.num_epochs}, {tc.batch_size}, {tc.learning_rate}",
        ),
        ("Window length and stride", f"{tc.max_length} and {tc.window_stride} tokens"),
        ("Threshold", tc.threshold),
        ("Inference cost per hour", "not priced" if tc.cost_per_hour is None else tc.cost_per_hour),
        ("Output directory", config.paths.outputs_dir),
    ]
    print("| Setting | Value |\n|---|---|")
    for name, value in rows:
        print(f"| {name} | {value} |")
    return 0


COMMANDS = {
    "estimate": cmd_estimate,
    "zero-shot": cmd_zero_shot,
    "fine-tune": cmd_fine_tune,
    "compare": cmd_compare,
    "show-config": cmd_show_config,
}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    apply_common(args)
    try:
        return COMMANDS[args.command](args)
    except (FatalLLMError, FileNotFoundError, ValueError) as exc:
        logger.debug("Details", exc_info=True)
        logger.error("Stopped: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())

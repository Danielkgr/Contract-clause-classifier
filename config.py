"""
Configuration for Contract Clause Classifier.

Settings come from environment variables.  A .env file next to this module
is loaded once, when the configuration is built at startup, and a variable
already set in the environment takes precedence over the same one in .env.
"""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

T = TypeVar("T")


def _setting(env: Mapping[str, str], name: str, default: T, kind: Callable[[str], T] = str) -> T:
    """Read one setting, treating an empty value as unset.

    Raises:
        ValueError: naming the variable, when its value is not a valid kind
    """
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return kind(raw)
    except ValueError:
        expected = {int: "a whole number", float: "a number"}.get(kind, "valid")
        raise ValueError(f"{name} must be {expected}, got {raw!r}") from None


# The model used when LLM_MODEL is not set
DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", "openai": "gpt-4o-mini"}


@dataclass
class LLMConfig:
    """Zero-shot LLM settings."""

    provider: str = "anthropic"  # "anthropic" or "openai"
    model: str | None = None  # None means the provider's default model
    api_key: str | None = field(default=None, repr=False)  # never printed
    base_url: str | None = None  # an OpenAI-compatible server, openai provider only
    temperature: float | None = None  # sent only when set
    # A cap, not a charge: output is billed as used.  It leaves room for the
    # thinking that claude-opus-5-5 always does before its short JSON answer.
    max_tokens: int = 2048
    effort: str = "low"  # thinking depth for Claude models that take it
    mode: str = "multi"  # "multi" asks about every clause type per call, "single" one
    timeout: float = 120.0  # seconds to wait for one response
    max_attempts: int = 6  # tries per call when the API fails in a way that may pass
    concurrency: int = 4  # contracts asked about at the same time
    # Completed responses are kept here, so a re-run is free and a crash loses nothing
    cache_dir: str = str(BASE_DIR / ".cache" / "llm")
    # Contracts are sent in chunks that fit the model's context window.  The
    # overlap keeps a clause that straddles a boundary whole in one chunk.
    chunk_chars: int = 24000
    chunk_overlap: int = 1000

    @property
    def resolved_model(self) -> str:
        """The model to call: LLM_MODEL, or the provider's default."""
        return self.model or DEFAULT_MODELS.get(self.provider, "")

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "LLMConfig":
        """Build the settings from environment variables, falling back to the defaults."""
        return cls(
            provider=_setting(env, "LLM_PROVIDER", cls.provider),
            model=_setting(env, "LLM_MODEL", None),
            api_key=_setting(env, "LLM_API_KEY", None),
            base_url=_setting(env, "LLM_BASE_URL", None),
            temperature=_setting(env, "LLM_TEMPERATURE", None, float),
            max_tokens=_setting(env, "LLM_MAX_TOKENS", cls.max_tokens, int),
            effort=_setting(env, "LLM_EFFORT", cls.effort),
            mode=_setting(env, "LLM_MODE", cls.mode),
            timeout=_setting(env, "LLM_TIMEOUT", cls.timeout, float),
            max_attempts=_setting(env, "LLM_MAX_ATTEMPTS", cls.max_attempts, int),
            concurrency=_setting(env, "LLM_CONCURRENCY", cls.concurrency, int),
            cache_dir=_setting(env, "LLM_CACHE_DIR", cls.cache_dir),
            chunk_chars=_setting(env, "LLM_CHUNK_CHARS", cls.chunk_chars, int),
            chunk_overlap=_setting(env, "LLM_CHUNK_OVERLAP", cls.chunk_overlap, int),
        )


@dataclass
class TrainingConfig:
    """Training configuration for fine-tuned model."""

    model_name: str = "roberta-base"
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    max_length: int = 512
    weight_decay: float = 0.01
    warmup_steps: int = 500
    eval_steps: int = 500
    save_steps: int = 1000
    # Contracts are split into max_length-token windows overlapping by this many tokens
    window_stride: int = 128
    # All-negative training windows kept per window with a clause in it
    negative_window_ratio: float = 1.0
    # A clause is predicted present when any window scores at least this
    threshold: float = 0.5
    # USD per hour for the machine running inference.  Unset means the
    # fine-tuned arm reports measured compute time and no cost.
    cost_per_hour: float | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "TrainingConfig":
        """Build the settings from environment variables, falling back to the defaults."""
        return cls(
            model_name=_setting(env, "TRAIN_MODEL", cls.model_name),
            batch_size=_setting(env, "BATCH_SIZE", cls.batch_size, int),
            learning_rate=_setting(env, "LR", cls.learning_rate, float),
            num_epochs=_setting(env, "NUM_EPOCHS", cls.num_epochs, int),
            max_length=_setting(env, "MAX_LENGTH", cls.max_length, int),
            weight_decay=_setting(env, "WEIGHT_DECAY", cls.weight_decay, float),
            warmup_steps=_setting(env, "WARMUP_STEPS", cls.warmup_steps, int),
            eval_steps=_setting(env, "EVAL_STEPS", cls.eval_steps, int),
            save_steps=_setting(env, "SAVE_STEPS", cls.save_steps, int),
            window_stride=_setting(env, "WINDOW_STRIDE", cls.window_stride, int),
            negative_window_ratio=_setting(
                env, "NEGATIVE_WINDOW_RATIO", cls.negative_window_ratio, float
            ),
            threshold=_setting(env, "FT_THRESHOLD", cls.threshold, float),
            cost_per_hour=_setting(env, "FT_COST_PER_HOUR", None, float),
        )


# The 41 CUAD v1 categories, spelled as they appear in CUAD_v1.json
CUAD_CATEGORIES = [
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "Competitive Restriction Exception",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
]

# Default clause types, a mix of common and rarer CUAD categories
DEFAULT_CLAUSE_TYPES = [
    "Governing Law",
    "Anti-Assignment",
    "Cap On Liability",
    "Uncapped Liability",
    "Audit Rights",
    "Termination For Convenience",
    "Change Of Control",
    "Exclusivity",
    "Non-Compete",
    "Insurance",
    "License Grant",
    "Warranty Duration",
]


def is_cuad_category(name: str) -> bool:
    """True if name is a CUAD category, ignoring case."""
    return name.casefold() in {c.casefold() for c in CUAD_CATEGORIES}


def mask_secret(value: str | None) -> str:
    """Show a secret such as an API key without revealing it.

    At most the last four characters are shown, and none of a value shorter
    than 16 characters, so the result is safe to print or log.
    """
    if not value:
        return "(not set)"
    if len(value) < 16:
        return "****"
    return "..." + value[-4:]


@dataclass
class DataConfig:
    """Data configuration."""

    # CUAD v1 in SQuAD 2.0 format, from https://huggingface.co/datasets/theatticusproject/cuad
    dataset_repo: str = "theatticusproject/cuad"
    dataset_file: str = "CUAD_v1/CUAD_v1.json"
    max_samples: int | None = None  # Set None for full dataset
    clause_types: list = None

    def __post_init__(self):
        if self.clause_types is None:
            self.clause_types = list(DEFAULT_CLAUSE_TYPES)

    def add_clause_type(self, name: str) -> None:
        """Add a clause type if it is not already active."""
        if name not in self.clause_types:
            self.clause_types.append(name)

    def remove_clause_type(self, name: str) -> None:
        """Remove a clause type if it is active."""
        if name in self.clause_types:
            self.clause_types.remove(name)


@dataclass
class Paths:
    """Path configuration."""

    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    data_dir: str = os.path.join(base_dir, "data")
    outputs_dir: str = os.path.join(base_dir, "outputs")
    models_dir: str = os.path.join(base_dir, "models")
    notebook_dir: str = base_dir

    def __post_init__(self):
        for dir_path in [self.data_dir, self.outputs_dir, self.models_dir]:
            os.makedirs(dir_path, exist_ok=True)


@dataclass
class Config:
    """Every setting, grouped by area."""

    llm: LLMConfig
    training: TrainingConfig
    data: DataConfig
    paths: Paths


def load_config(env_file: os.PathLike | None = ENV_FILE) -> Config:
    """Build the configuration from .env and the environment.

    The .env file is optional.  A variable already set in the environment,
    such as a shell export or a CI secret, wins over the same one in .env.
    """
    if env_file is not None:
        load_dotenv(env_file, override=False)
    return Config(
        llm=LLMConfig.from_env(os.environ),
        training=TrainingConfig.from_env(os.environ),
        data=DataConfig(),
        paths=Paths(),
    )


# Built once at startup, which is when .env is read
config = load_config()

"""
Configuration file for Contract Clause Classifier.
Manages environment variables and default settings.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = os.getenv("LLM_PROVIDER", "openai")
    model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    api_key: Optional[str] = os.getenv("LLM_API_KEY")
    base_url: Optional[str] = os.getenv("LLM_BASE_URL")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "500"))
    
    # List price per million tokens (input, output) in USD.
    # Source: https://openai.com/api/pricing  Check before relying on costs.
    COSTS = {
        "gpt-3.5-turbo": (0.50, 1.50),
        "gpt-3.5-turbo-16k": (3.00, 4.00),
        "gpt-4": (30.00, 60.00),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
    }
    # Used, with a warning, for any model not listed in COSTS
    DEFAULT_COST_PER_MILLION = (1.00, 2.00)

    def get_cost_per_million(self, model: Optional[str] = None) -> tuple:
        """Get cost per million tokens (input, output) for a model."""
        return self.COSTS.get(model or self.model, self.DEFAULT_COST_PER_MILLION)


@dataclass
class TrainingConfig:
    """Training configuration for fine-tuned model."""
    model_name: str = os.getenv("TRAIN_MODEL", "roberta-base")
    batch_size: int = int(os.getenv("BATCH_SIZE", "8"))
    learning_rate: float = float(os.getenv("LR", "2e-5"))
    num_epochs: int = int(os.getenv("NUM_EPOCHS", "3"))
    max_length: int = int(os.getenv("MAX_LENGTH", "512"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "0.01"))
    warmup_steps: int = int(os.getenv("WARMUP_STEPS", "500"))
    eval_steps: int = int(os.getenv("EVAL_STEPS", "500"))
    save_steps: int = int(os.getenv("SAVE_STEPS", "1000"))


# The 41 CUAD v1 categories, spelled as they appear in CUAD_v1.json
CUAD_CATEGORIES = [
    "Document Name", "Parties", "Agreement Date", "Effective Date",
    "Expiration Date", "Renewal Term", "Notice Period To Terminate Renewal",
    "Governing Law", "Most Favored Nation", "Non-Compete", "Exclusivity",
    "No-Solicit Of Customers", "Competitive Restriction Exception",
    "No-Solicit Of Employees", "Non-Disparagement", "Termination For Convenience",
    "Rofr/Rofo/Rofn", "Change Of Control", "Anti-Assignment",
    "Revenue/Profit Sharing", "Price Restrictions", "Minimum Commitment",
    "Volume Restriction", "Ip Ownership Assignment", "Joint Ip Ownership",
    "License Grant", "Non-Transferable License", "Affiliate License-Licensor",
    "Affiliate License-Licensee", "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License", "Source Code Escrow",
    "Post-Termination Services", "Audit Rights", "Uncapped Liability",
    "Cap On Liability", "Liquidated Damages", "Warranty Duration", "Insurance",
    "Covenant Not To Sue", "Third Party Beneficiary",
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


@dataclass
class DataConfig:
    """Data configuration."""
    # CUAD v1 in SQuAD 2.0 format, from https://huggingface.co/datasets/theatticusproject/cuad
    dataset_repo: str = "theatticusproject/cuad"
    dataset_file: str = "CUAD_v1/CUAD_v1.json"
    max_samples: Optional[int] = None  # Set None for full dataset
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


# Global configuration instance
config = type('Config', (), {})()
config.llm = LLMConfig()
config.training = TrainingConfig()
config.data = DataConfig()
config.paths = Paths()

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
    
    # Cost per million tokens (input, output) in USD
    # Source: https://openai.com/pricing
    COSTS = {
        "gpt-3.5-turbo": (0.0005, 0.0015),
        "gpt-3.5-turbo-16k": (0.003, 0.004),
        "gpt-4": (0.03, 0.06),
        "gpt-4-turbo": (0.01, 0.03),
        "gpt-4o": (0.005, 0.015),
    }
    
    def get_cost_per_million(self) -> tuple:
        """Get cost per million tokens (input, output)."""
        return self.COSTS.get(self.model, (0.001, 0.002))


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


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_name: str = "cuad"  # HuggingFace dataset
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    max_samples: Optional[int] = None  # Set None for full dataset
    clause_types: list = None
    
    def __post_init__(self):
        if self.clause_types is None:
            # CUAD clause types
            self.clause_types = [
                "Agreement Effectiveness",
                "Agreement Termination",
                "Anti-Assignment",
                "Arbitration",
                "Attorneys' Fees",
                "Notice",
                "Governing Law",
                "Indemnification",
                "Jurisdiction",
                "Severability",
                "Waiver",
                "Warranty",
            ]


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

"""
Fine-tuned transformer for contract clause classification.

One multi-label model scores every active clause type at once.  Each contract
is split into overlapping windows of max_length tokens.  A training window is
labelled with the clause types whose CUAD answer spans it overlaps, and a
contract is predicted to contain a clause type when any of its windows scores
at or above the threshold.
"""

import os
import random
import time
import logging
from typing import Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)

from config import config
from utils.chunking import window_labels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Results from model training."""
    model_path: str
    metrics: Dict[str, float]
    training_time_seconds: float
    epochs: int
    train_windows: int
    val_windows: int


class WindowDataset(Dataset):
    """Tokenised contract windows with one 0/1 label per clause type."""

    def __init__(self, input_ids: List[List[int]], attention_mask: List[List[int]],
                 labels: List[List[int]]):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": [float(x) for x in self.labels[idx]],
        }


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FineTunedClassifier:
    """Multi-label transformer that reads whole contracts window by window."""

    def __init__(
        self,
        clause_types: Optional[Sequence[str]] = None,
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize classifier.

        Args:
            clause_types: Clause types the model scores, one output each
            model_name: Transformer model name (e.g., 'roberta-base')
            device: PyTorch device (CUDA, MPS, or CPU)
        """
        self.clause_types = list(clause_types or config.data.clause_types)
        self.model_name = model_name or config.training.model_name
        self.device = device or _pick_device()
        self.base_model_name = self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None
        self.trainer = None

        logger.info(f"Initialized classifier with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")

    def windows(self, text: str) -> Tuple[List[List[int]], List[List[int]], List[Tuple[int, int]]]:
        """Split text into overlapping token windows covering all of it.

        The whole text is tokenised once and sliced here, rather than through
        return_overflowing_tokens, which in tokenizers 0.23 returns only one
        overflow window and so silently drops the rest of a long contract.

        Returns:
            Tuple of (input_ids, attention_mask, char_spans), one entry per
            window, where char_spans gives each window's (start, end) in text
        """
        enc = self.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True, verbose=False,
        )
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        prefix, suffix = self._special_tokens()
        # Never exceed what the model accepts (roberta-base takes 512 tokens)
        max_length = min(config.training.max_length, self.tokenizer.model_max_length)
        body = max_length - len(prefix) - len(suffix)
        step = body - config.training.window_stride
        if step <= 0:
            raise ValueError("window_stride must be less than max_length minus special tokens")

        input_ids, attention_mask, char_spans = [], [], []
        start = 0
        while True:
            piece = ids[start:start + body]
            window = prefix + piece + suffix
            input_ids.append(window)
            attention_mask.append([1] * len(window))
            piece_offsets = offsets[start:start + body]
            char_spans.append(
                (piece_offsets[0][0], piece_offsets[-1][1]) if piece_offsets else (0, 0)
            )
            if start + body >= len(ids):
                return input_ids, attention_mask, char_spans
            start += step

    def _special_tokens(self) -> Tuple[List[int], List[int]]:
        """The ids the tokenizer puts before and after a single sequence."""
        inner = self.tokenizer("contract", add_special_tokens=False)["input_ids"]
        full = self.tokenizer("contract")["input_ids"]
        for i in range(len(full) - len(inner) + 1):
            if full[i:i + len(inner)] == inner:
                return full[:i], full[i + len(inner):]
        raise ValueError("Could not locate the special tokens this tokenizer adds")

    def build_dataset(self, contracts, negative_ratio: Optional[float] = None,
                      seed: int = 42) -> WindowDataset:
        """Build labelled windows from contracts.

        Every window containing at least one clause is kept.  All-negative
        windows vastly outnumber them, so negative_ratio of them are sampled
        for each positive window.
        """
        if negative_ratio is None:
            negative_ratio = config.training.negative_window_ratio

        positives, negatives = [], []
        for contract in contracts:
            missing = [ct for ct in self.clause_types
                       if contract.clauses.get(ct) and not contract.spans.get(ct)]
            if missing:
                raise ValueError(
                    f"{contract.contract_id} marks {missing} present but has no answer "
                    f"spans.  Fine-tuning needs the spans that the CUAD loader provides."
                )
            ids, mask, char_spans = self.windows(contract.text)
            labels = window_labels(char_spans, contract.spans, self.clause_types)
            for row in zip(ids, mask, labels):
                (positives if any(row[2]) else negatives).append(row)

        if not positives:
            raise ValueError("No window contains any active clause type, so there is nothing to learn")

        rng = random.Random(seed)
        keep = min(len(negatives), int(round(len(positives) * negative_ratio)))
        rows = positives + rng.sample(negatives, keep)
        rng.shuffle(rows)
        logger.info(f"Built {len(rows)} windows: {len(positives)} with a clause, "
                    f"{keep} of {len(negatives)} without")
        return WindowDataset([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])

    def train(self, train_contracts, val_contracts=None,
              output_dir: Optional[str] = None) -> TrainingResult:
        """Fine-tune on the windows of train_contracts.

        Args:
            train_contracts: ContractData with answer spans
            val_contracts: Optional ContractData for evaluation during training
            output_dir: Output directory for model checkpoints

        Returns:
            TrainingResult with metrics and timing
        """
        if output_dir is None:
            output_dir = os.path.join(config.paths.models_dir, "fine_tuned")
        os.makedirs(output_dir, exist_ok=True)

        train_dataset = self.build_dataset(train_contracts)
        val_dataset = self.build_dataset(val_contracts) if val_contracts else None

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.training.num_epochs,
            per_device_train_batch_size=config.training.batch_size,
            per_device_eval_batch_size=config.training.batch_size,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            warmup_steps=config.training.warmup_steps,
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=config.training.eval_steps,
            save_strategy="steps",
            save_steps=config.training.save_steps,
            save_total_limit=2,
            logging_steps=50,
            load_best_model_at_end=bool(val_dataset),
            metric_for_best_model="eval_loss" if val_dataset else None,
            report_to=[],  # Disable wandb/tensorboard
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.clause_types),
            problem_type="multi_label_classification",
            id2label=dict(enumerate(self.clause_types)),
            label2id={ct: i for i, ct in enumerate(self.clause_types)},
            ignore_mismatched_sizes=True,  # replace any existing classification head
        )
        self.model.config.base_model_name = self.base_model_name

        threshold = config.training.threshold

        def compute_metrics(eval_pred):
            from sklearn.metrics import precision_recall_fscore_support
            logits, labels = eval_pred
            preds = (1 / (1 + np.exp(-logits)) >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels.astype(int), preds, average="micro", zero_division=0
            )
            return {"window_precision": precision, "window_recall": recall, "window_f1": f1}

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=compute_metrics if val_dataset else None,
        )

        start_time = time.time()
        train_result = self.trainer.train()
        training_time = time.time() - start_time

        self.model = self.trainer.model
        self.device = self.model.device
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        metrics = {**train_result.metrics, "training_time_seconds": training_time}
        if val_dataset:
            metrics.update(self.trainer.evaluate())

        logger.info(f"Training completed in {training_time:.2f} seconds")

        return TrainingResult(
            model_path=output_dir,
            metrics=metrics,
            training_time_seconds=training_time,
            epochs=config.training.num_epochs,
            train_windows=len(train_dataset),
            val_windows=len(val_dataset) if val_dataset else 0,
        )

    def load(self, model_path: str):
        """Load a trained model and take its clause types from its labels.

        Args:
            model_path: Path to saved model
        """
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        if model.config.problem_type != "multi_label_classification":
            raise ValueError(
                f"{model_path} is not a multi-label clause model.  Models trained "
                f"before the windowed classifier need retraining."
            )
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.clause_types = [model.config.id2label[i] for i in range(model.config.num_labels)]
        self.base_model_name = getattr(model.config, "base_model_name", model_path)

        logger.info(f"Loaded model from {model_path}")

    def predict_contract(self, text: str, batch_size: Optional[int] = None
                         ) -> Tuple[Dict[str, float], float]:
        """Score a whole contract.

        Returns:
            Tuple of (clause type -> highest window probability, latency in ms).
            The latency covers tokenising and scoring every window.
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained. Call load() or train() first.")
        batch_size = batch_size or config.training.batch_size

        self.model.eval()
        start = time.perf_counter()
        ids, mask, _ = self.windows(text)
        best = np.zeros(len(self.clause_types))
        with torch.no_grad():
            for i in range(0, len(ids), batch_size):
                batch = self.tokenizer.pad(
                    {"input_ids": ids[i:i + batch_size], "attention_mask": mask[i:i + batch_size]},
                    return_tensors="pt",
                ).to(self.device)
                probs = torch.sigmoid(self.model(**batch).logits).cpu().numpy()
                best = np.maximum(best, probs.max(axis=0))
        latency_ms = (time.perf_counter() - start) * 1000

        return dict(zip(self.clause_types, best.tolist())), latency_ms

    def save(self, output_path: str):
        """Save model to path.

        Args:
            output_path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")

        os.makedirs(output_path, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        logger.info(f"Saved model to {output_path}")

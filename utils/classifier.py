"""
Fine-tuned classifier module using HuggingFace Transformers.
Supports BERT, RoBERTa, and other transformer models.
"""

import os
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Results from model training."""
    model_path: str
    metrics: Dict[str, float]
    training_time_seconds: float
    epochs: int


class ClauseDataset(Dataset):
    """PyTorch Dataset for clause classification."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class FineTunedClassifier:
    """Fine-tuned transformer model for clause classification."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        num_labels: int = 2,
        device: Optional[torch.device] = None
    ):
        """Initialize classifier.
        
        Args:
            model_name: Transformer model name (e.g., 'roberta-base')
            num_labels: Number of output labels (2 for binary classification)
            device: PyTorch device (CPU or GPU)
        """
        self.model_name = model_name or config.training.model_name
        self.num_labels = num_labels
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None
        self.trainer = None
        
        logger.info(f"Initialized classifier with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(
        self,
        texts: List[str],
        labels: List[int],
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Split data into train/val/test datasets.
        
        Args:
            texts: List of text samples
            labels: List of binary labels
            test_size: Test set proportion
            val_size: Validation set proportion
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Split into train+val and test
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Split train+val into train and val
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels,
            test_size=val_size, random_state=42, stratify=train_val_labels
        )
        
        # Create datasets
        train_dataset = ClauseDataset(
            train_texts, train_labels, self.tokenizer, config.training.max_length
        )
        val_dataset = ClauseDataset(
            val_texts, val_labels, self.tokenizer, config.training.max_length
        )
        test_dataset = ClauseDataset(
            test_texts, test_labels, self.tokenizer, config.training.max_length
        )
        
        logger.info(f"Created datasets: train={len(train_dataset)}, "
                   f"val={len(val_dataset)}, test={len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        output_dir: Optional[str] = None
    ) -> TrainingResult:
        """Fine-tune the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            output_dir: Output directory for model checkpoints
            
        Returns:
            TrainingResult with metrics and timing
        """
        if output_dir is None:
            output_dir = os.path.join(config.paths.models_dir, "fine_tuned")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.training.num_epochs,
            per_device_train_batch_size=config.training.batch_size,
            per_device_eval_batch_size=config.training.batch_size,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            warmup_steps=config.training.warmup_steps,
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=config.training.eval_steps,
            save_steps=config.training.save_steps,
            save_total_limit=2,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=100,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            report_to=[],  # Disable wandb/tensorboard
        )
        
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        ).to(self.device)
        
        # Data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="max_length",
            max_length=config.training.max_length
        )
        
        # Metrics function
        def compute_metrics(eval_pred):
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average="binary", zero_division=0
            )
            
            return {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy_score(labels, predictions),
            }
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset if val_dataset else None,
            data_collator=data_collator,
            compute_metrics=compute_metrics if val_dataset else None,
        )
        
        # Train
        start_time = time.time()
        train_result = self.trainer.train()
        training_time = time.time() - start_time
        
        # Save model
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Get metrics
        metrics = {
            **train_result.metrics,
            "training_time_seconds": training_time,
        }
        
        # Evaluate on validation set
        if val_dataset:
            eval_results = self.trainer.evaluate()
            metrics.update({f"eval_{k}": v for k, v in eval_results.items()})
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return TrainingResult(
            model_path=output_dir,
            metrics=metrics,
            training_time_seconds=training_time,
            epochs=config.training.num_epochs
        )
    
    def load(self, model_path: str):
        """Load a trained model.
        
        Args:
            model_path: Path to saved model
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded model from {model_path}")
    
    def predict(
        self,
        texts: List[str],
        batch_size: int = 8
    ) -> Tuple[List[int], List[float], float]:
        """Predict labels for input texts.
        
        Args:
            texts: List of text samples
            batch_size: Batch size for inference
            
        Returns:
            Tuple of (predictions, probabilities, avg_latency_ms)
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained. Call load() or train() first.")
        
        self.model.eval()
        predictions = []
        probabilities = []
        latencies = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encodings = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=config.training.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass
                start_time = time.time()
                outputs = self.model(**encodings)
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)
                
                # Get predictions
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                batch_preds = torch.argmax(probs, dim=-1).cpu().numpy()
                batch_probs = probs[:, 1].cpu().numpy()  # Probability of positive class
                
                predictions.extend(batch_preds.tolist())
                probabilities.extend(batch_probs.tolist())
        
        avg_latency = np.mean(latencies) if latencies else 0
        
        return predictions, probabilities, avg_latency
    
    def predict_single(self, text: str) -> Tuple[int, float]:
        """Predict label for a single text.
        
        Args:
            text: Single text sample
            
        Returns:
            Tuple of (prediction, probability)
        """
        preds, probs, _ = self.predict([text])
        return preds[0], probs[0]
    
    def save(self, output_path: str):
        """Save model to path.
        
        Args:
            output_path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")
        
        os.makedirs(output_path, exist_ok=True)
        self.trainer.save_model(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        logger.info(f"Saved model to {output_path}")

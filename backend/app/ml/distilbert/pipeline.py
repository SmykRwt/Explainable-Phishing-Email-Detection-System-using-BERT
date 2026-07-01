import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger("phishing_platform")

class SpamDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item


class DistilBERTPipeline:
    def __init__(self, model_name: str = "distilbert-base-uncased", save_dir: str = "./bert_model"):
        self.model_name = model_name
        self.save_dir = save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def initialize_new(self):
        """Loads default model and tokenizer from Hugging Face."""
        logger.info(f"Initializing new DistilBERT model ({self.model_name})...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.model.to(self.device)

    def load(self, model_dir: str = None):
        """Loads a fine-tuned model and tokenizer from disk."""
        target_dir = model_dir or self.save_dir
        if not os.path.exists(target_dir):
            raise FileNotFoundError(f"Model checkpoint directory '{target_dir}' does not exist.")
        logger.info(f"Loading DistilBERT checkpoint from {target_dir}...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(target_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(target_dir)
        self.model.to(self.device)
        self.model.eval()

    def train(self, 
              train_texts: List[str], 
              train_labels: List[int],
              val_texts: List[str], 
              val_labels: List[int],
              epochs: int = 3,
              batch_size: int = 16,
              lr: float = 2e-5) -> Dict[str, Any]:
        """Runs the fine-tuning process on DistilBERT."""
        if self.model is None or self.tokenizer is None:
            self.initialize_new()

        logger.info("Tokenizing train and validation sets...")
        train_encodings = self.tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
        val_encodings = self.tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

        train_ds = SpamDataset(train_encodings, train_labels)
        val_ds = SpamDataset(val_encodings, val_labels)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # Set up class imbalance weights
        n_ham = sum(1 for l in train_labels if l == 0)
        n_spam = sum(1 for l in train_labels if l == 1)
        # Weight for class 1 = ham count / spam count
        weights = torch.tensor([1.0, n_ham / max(n_spam, 1)], dtype=torch.float).to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

        optimizer = AdamW(self.model.parameters(), lr=lr)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        logger.info(f"Starting DistilBERT fine-tuning on {self.device} for {epochs} epochs...")
        best_val_f1 = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=ids, attention_mask=mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            
            # Run Validation
            val_metrics = self.evaluate(val_loader, loss_fn)
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | Val F1: {val_metrics['f1']*100:.2f}%")
            
            # Save if best
            if val_metrics["f1"] >= best_val_f1:
                best_val_f1 = val_metrics["f1"]
                logger.info(f"Saving new best model to {self.save_dir} with F1 {best_val_f1*100:.2f}%")
                os.makedirs(self.save_dir, exist_ok=True)
                self.model.save_pretrained(self.save_dir)
                self.tokenizer.save_pretrained(self.save_dir)

        # Reload best weights
        self.load(self.save_dir)
        return {"best_val_f1": best_val_f1}

    def evaluate(self, val_loader: DataLoader, loss_fn: torch.nn.Module) -> Dict[str, float]:
        """Evaluates on validation loader."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=ids, attention_mask=mask)
                loss = loss_fn(outputs.logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        return {
            "val_loss": avg_loss,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Returns spam probabilities for list of texts."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer is not loaded. Train or load model first.")
        
        self.model.eval()
        encodings = self.tokenizer(list(texts), truncation=True, padding=True, max_length=128, return_tensors="pt")
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        with torch.no_grad():
            logits = self.model(**encodings).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

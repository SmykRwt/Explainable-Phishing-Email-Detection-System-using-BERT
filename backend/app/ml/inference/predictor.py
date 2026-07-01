import os
import torch
import pickle
import numpy as np
import logging
from typing import Optional
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from backend.app.ml.classical.pipeline import ClassicalModelWrapper
from backend.app.core.config import settings

logger = logging.getLogger("phishing_platform")

class Predictor:
    def __init__(self, 
                 bert_model_dir: str = "./bert_model",
                 classical_model_path: str = "./best_model.pkl"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.bert_model = None
        self.classical_model = None

        # 1. Load DistilBERT
        try:
            if os.path.exists(bert_model_dir):
                self.tokenizer = DistilBertTokenizerFast.from_pretrained(bert_model_dir)
                self.bert_model = DistilBertForSequenceClassification.from_pretrained(bert_model_dir)
                logger.info(f"Loaded fine-tuned DistilBERT model from {bert_model_dir}")
            else:
                logger.warning(f"Fine-tuned DistilBERT not found at {bert_model_dir}. Loading default 'distilbert-base-uncased' as fallback...")
                self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
                self.bert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
            
            self.bert_model.to(self.device)
            self.bert_model.eval()
        except Exception as e:
            logger.error(f"Failed to load DistilBERT model: {e}")

        # 2. Load Classical Model (optional fallback)
        try:
            if os.path.exists(classical_model_path):
                with open(classical_model_path, "rb") as f:
                    self.classical_model = pickle.load(f)
                logger.info(f"Loaded classical model wrapper from {classical_model_path}")
            else:
                logger.info(f"Classical model not found at {classical_model_path}. It will be trained during benchmark.")
        except Exception as e:
            logger.error(f"Failed to load classical model: {e}")

    def predict_bert(self, text: str) -> np.ndarray:
        """Returns [prob_ham, prob_spam] from DistilBERT."""
        if not self.bert_model or not self.tokenizer:
            return np.array([0.5, 0.5])
        
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            padding=True, max_length=128
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        
        with torch.no_grad():
            logits = self.bert_model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return probs

    def predict_classical(self, text: str) -> Optional[np.ndarray]:
        """Returns [prob_ham, prob_spam] from best classical model if available."""
        if not self.classical_model:
            return None
        try:
            probs = self.classical_model.predict_proba([text])[0]
            return probs
        except Exception as e:
            logger.error(f"Classical prediction failed: {e}")
            return None

    def get_token_importance(self, text: str) -> list[dict]:
        """
        Calculates gradient attribution per token for explainability.
        Higher score = token is more indicative of the positive class (Spam/Phishing).
        """
        if not self.bert_model or not self.tokenizer:
            return []

        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            padding=True, max_length=128
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        input_ids = enc["input_ids"][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # Clear gradients
        self.bert_model.zero_grad()
        
        # Get embedding inputs
        embeddings = self.bert_model.distilbert.embeddings(enc["input_ids"])
        embeddings.retain_grad()

        # Run forward pass using inputs_embeds
        logits = self.bert_model(
            inputs_embeds=embeddings,
            attention_mask=enc["attention_mask"]
        ).logits
        
        # We calculate gradient with respect to class 1 (Spam/Phishing)
        spam_score = logits[0, 1]
        spam_score.backward()

        # Gradient × Embedding norm to get an attribution proxy per token
        grad_norms = embeddings.grad[0].norm(dim=-1).detach().cpu().numpy()
        total_grad = grad_norms.sum() + 1e-9
        
        results = []
        for t, g in zip(tokens, grad_norms):
            # Skip special tokens
            if t not in ("[CLS]", "[SEP]", "[PAD]"):
                results.append({
                    "token": t,
                    "score": float(g / total_grad)
                })
        
        # Sort by importance
        return sorted(results, key=lambda x: -x["score"])

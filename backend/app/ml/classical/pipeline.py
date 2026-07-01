import os
import re
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import scipy.sparse as sp

# Safe NLTK downloads
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

URGENCY_WORDS = [
    "urgent", "immediately", "act now", "limited time", "expires",
    "verify", "confirm", "suspended", "account", "prize", "winner",
    "free", "click", "login", "password", "bank", "credit",
]

class FeatureEngineer:
    def __init__(self):
        self.ps = PorterStemmer()
        self.stop = set(stopwords.words("english"))
        self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    def clean_text(self, text: str) -> str:
        """Applies stemming and removes stopwords for standard bag-of-words/TF-IDF models."""
        if not isinstance(text, str):
            text = ""
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [self.ps.stem(w) for w in tokens if w.isalnum() and w not in self.stop]
        return " ".join(tokens)

    def extract_manual_features(self, text: str) -> np.ndarray:
        """Extracts statistical metadata features from the raw text."""
        if not isinstance(text, str):
            text = ""
        text_l = text.lower()
        urls = re.findall(r"https?://\S+|www\.\S+", text_l)
        
        feat = [
            float(bool(urls)),                                                # has_url
            float(len(urls)),                                                # url_count
            float(text.count("!")),                                           # exclamation_count
            float(sum(1 for c in text if c.isupper()) / max(len(text), 1)),   # caps_ratio
            float(sum(1 for w in URGENCY_WORDS if w in text_l)),             # urgency_count
            float(sum(1 for c in text if c.isdigit()) / max(len(text), 1)),   # digit_ratio
            float(len(text))                                                  # char_len
        ]
        return np.array(feat, dtype=np.float32)

    def fit_transform(self, texts: List[str]) -> sp.csr_matrix:
        """Fits TF-IDF and returns the combined TF-IDF + manual features sparse matrix."""
        cleaned_texts = [self.clean_text(t) for t in texts]
        tfidf_features = self.tfidf.fit_transform(cleaned_texts)
        
        manual_features = np.array([self.extract_manual_features(t) for t in texts])
        # Convert to CSR sparse matrix and concatenate
        manual_sparse = sp.csr_matrix(manual_features)
        combined = sp.hstack([tfidf_features, manual_sparse], format="csr")
        return combined

    def transform(self, texts: List[str]) -> sp.csr_matrix:
        """Transforms texts using the fitted vectorizer and concatenates manual features."""
        cleaned_texts = [self.clean_text(t) for t in texts]
        tfidf_features = self.tfidf.transform(cleaned_texts)
        
        manual_features = np.array([self.extract_manual_features(t) for t in texts])
        manual_sparse = sp.csr_matrix(manual_features)
        combined = sp.hstack([tfidf_features, manual_sparse], format="csr")
        return combined


class ClassicalModelWrapper:
    def __init__(self, model_type: str = "logistic_regression"):
        self.model_type = model_type.lower()
        self.fe = FeatureEngineer()
        
        if self.model_type == "logistic_regression":
            self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        elif self.model_type == "svm":
            self.model = SVC(kernel="linear", class_weight="balanced", probability=True)
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        elif self.model_type == "xgboost":
            # Compute class scale ratio for XGBoost balancing
            self.model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, X_texts: List[str], y: List[int]):
        """Trains the TF-IDF vectorizer and the ML model."""
        X_features = self.fe.fit_transform(X_texts)
        y_arr = np.array(y)
        
        if self.model_type == "xgboost":
            # Update scale_pos_weight dynamic balancer
            n_neg = sum(y_arr == 0)
            n_pos = sum(y_arr == 1)
            scale = n_neg / max(n_pos, 1)
            self.model.set_params(scale_pos_weight=scale)
            
        self.model.fit(X_features, y_arr)

    def predict(self, X_texts: List[str]) -> np.ndarray:
        """Predicts classes (0 or 1) for a list of texts."""
        X_features = self.fe.transform(X_texts)
        return self.model.predict(X_features)

    def predict_proba(self, X_texts: List[str]) -> np.ndarray:
        """Predicts probabilities [prob_ham, prob_spam] for a list of texts."""
        X_features = self.fe.transform(X_texts)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_features)
        else:
            # Fallback if probability is not supported (rare for these models)
            preds = self.model.predict(X_features)
            probs = np.zeros((len(preds), 2))
            probs[np.arange(len(preds)), preds] = 1.0
            return probs

    def evaluate(self, X_texts: List[str], y: List[int]) -> Dict[str, float]:
        """Evaluates model performance and returns metrics."""
        y_true = np.array(y)
        y_pred = self.predict(X_texts)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate ROC-AUC if predict_proba is available
        try:
            from sklearn.metrics import roc_auc_score
            probs = self.predict_proba(X_texts)[:, 1]
            auc = roc_auc_score(y_true, probs)
        except Exception:
            auc = 0.0

        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(auc)
        }

    def save(self, filepath: str):
        """Saves the fitted model wrapper using pickle."""
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> "ClassicalModelWrapper":
        """Loads a saved model wrapper."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

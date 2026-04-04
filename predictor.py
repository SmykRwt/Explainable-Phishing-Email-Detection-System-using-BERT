"""
predictor.py  –  BERT inference + LIME explainability + feature engineering
Used by app.py
"""

import re
import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ── Urgency / phishing patterns ──────────────────────────────────────────────
URGENCY_PATTERNS = {
    "🚨 Urgency detected":       r"\b(urgent|immediately|act now|limited time|expires?|asap|right now|don'?t delay)\b",
    "🔗 Suspicious link":        r"https?://\S+|www\.\S+",
    "🎁 Prize / reward bait":    r"\b(prize|winner|won|congratulation|reward|gift|claim|lucky)\b",
    "🔒 Account threat":         r"\b(suspend|block|restrict|verif|confirm|update|login|password|bank|credit|account)\b",
    "📞 Call to action":         r"\b(call|click|reply|text|visit|enter|subscribe|register)\b",
    "💰 Financial lure":         r"\b(free|cash|money|earn|profit|investment|£|\$|€|win)\b",
    "🎭 Impersonation pattern":  r"\b(dear (customer|user|member|friend)|your (account|subscription|service))\b",
}

LABEL_MAP = {0: "✅ Safe (Ham)", 1: "⚠️ Spam / Phishing"}
COLOR_MAP = {0: "green", 1: "red"}

class BERTPredictor:
    def __init__(self, model_dir: str = "./bert_model"):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model     = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    # ── Core prediction ───────────────────────────────────────────────────────
    def predict_proba(self, text: str) -> np.ndarray:
        """Returns [ham_prob, spam_prob]."""
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            padding=True, max_length=128
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return probs

    # ── Token-level attribution (Integrated Gradients lite) ──────────────────
    def get_token_importance(self, text: str) -> list[dict]:
        """
        Returns list of {token, score} where higher score = more suspicious.
        Uses gradient × embedding norm as a fast attribution proxy.
        """
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            padding=True, max_length=128
        )
        enc     = {k: v.to(self.device) for k, v in enc.items()}
        tokens  = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

        # Enable gradients for embeddings
        self.model.zero_grad()
        embeddings = self.model.distilbert.embeddings(enc["input_ids"])
        embeddings.retain_grad()

        logits = self.model(
            inputs_embeds=embeddings,
            attention_mask=enc["attention_mask"]
        ).logits
        spam_score = logits[0, 1]
        spam_score.backward()

        # grad norm per token
        grad_norms = embeddings.grad[0].norm(dim=-1).detach().cpu().numpy()
        total      = grad_norms.sum() + 1e-9
        results    = [
            {"token": t, "score": float(g / total)}
            for t, g in zip(tokens, grad_norms)
            if t not in ("[CLS]", "[SEP]", "[PAD]")
        ]
        return sorted(results, key=lambda x: -x["score"])[:15]

    # ── Rule-based pattern reasons ─────────────────────────────────────────────
    def get_reasons(self, text: str) -> list[str]:
        text_l  = text.lower()
        reasons = []
        for reason, pattern in URGENCY_PATTERNS.items():
            if re.search(pattern, text_l):
                reasons.append(reason)
        return reasons

    # ── URL feature extraction ────────────────────────────────────────────────
    def url_features(self, text: str) -> dict:
        urls = re.findall(r"https?://\S+|www\.\S+", text.lower())
        if not urls:
            return {}
        features = {}
        for url in urls[:3]:
            if len(url) > 60:
                features["⚠️ Unusually long URL detected"] = url
            # Domain mismatch heuristic: numbers in domain
            domain = re.sub(r"https?://", "", url).split("/")[0]
            if re.search(r"\d{3,}", domain):
                features["⚠️ Numeric domain (phishing indicator)"] = domain
            known_brands = ["paypal", "amazon", "google", "apple", "microsoft", "bank"]
            for brand in known_brands:
                if brand in url and brand not in domain.split(".")[0]:
                    features[f"⚠️ Brand impersonation ({brand})"] = domain
        return features

    # ── Full analysis ─────────────────────────────────────────────────────────
    def analyze(self, text: str) -> dict:
        probs      = self.predict_proba(text)
        label_idx  = int(np.argmax(probs))
        token_imp  = self.get_token_importance(text)
        reasons    = self.get_reasons(text)
        url_feats  = self.url_features(text)

        return {
            "label":         LABEL_MAP[label_idx],
            "label_idx":     label_idx,
            "confidence":    float(probs[label_idx]) * 100,
            "spam_prob":     float(probs[1]) * 100,
            "ham_prob":      float(probs[0]) * 100,
            "top_tokens":    token_imp[:8],
            "reasons":       reasons,
            "url_flags":     url_feats,
            "color":         COLOR_MAP[label_idx],
        }

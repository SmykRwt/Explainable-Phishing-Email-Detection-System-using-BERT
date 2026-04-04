"""
train_bert.py  –  Fine-tune DistilBERT on the SMS Spam dataset
Run: python train_bert.py
Outputs: ./bert_model/  (saved model + tokenizer)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os, re

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
MAX_LEN      = 128
BATCH_SIZE   = 16
EPOCHS       = 4
LR           = 2e-5
SAVE_DIR     = "./bert_model"
DATA_PATH    = "spam.csv"

# ── Feature Engineering helpers ─────────────────────────────────────────────
URGENCY_WORDS = [
    "urgent", "immediately", "act now", "limited time", "expires",
    "verify", "confirm", "suspended", "account", "prize", "winner",
    "free", "click", "login", "password", "bank", "credit",
]

def extract_features(text: str) -> dict:
    """Hand-crafted features that BERT alone might miss."""
    text_l = text.lower()
    urls    = re.findall(r"https?://\S+|www\.\S+", text_l)
    return {
        "has_url":        int(bool(urls)),
        "url_count":      len(urls),
        "exclamation":    text.count("!"),
        "caps_ratio":     sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "urgency_count":  sum(1 for w in URGENCY_WORDS if w in text_l),
        "digit_ratio":    sum(1 for c in text if c.isdigit()) / max(len(text), 1),
        "char_len":       len(text),
    }

# ── Dataset ──────────────────────────────────────────────────────────────────
class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True, max_length=MAX_LEN
        )
        self.labels    = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# ── Load & prep data ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    df["text"].values, df["label_num"].values,
    test_size=0.2, random_state=42, stratify=df["label_num"]
)

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
train_ds  = SpamDataset(X_train, y_train, tokenizer)
test_ds   = SpamDataset(X_test,  y_test,  tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# ── Model ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

# Class-imbalance aware loss weights  (ham=0.5, spam=3.2 approx)
n_ham, n_spam  = (df["label_num"] == 0).sum(), (df["label_num"] == 1).sum()
weights        = torch.tensor([1.0, n_ham / n_spam], dtype=torch.float).to(device)
loss_fn        = torch.nn.CrossEntropyLoss(weight=weights)

optimizer  = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
)

# ── Training loop ─────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = loss_fn(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}  |  avg loss: {avg_loss:.4f}")

# ── Evaluation ────────────────────────────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
        preds          = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n── Test Results ──")
print(classification_report(all_labels, all_preds, target_names=["Ham", "Spam"]))

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"\nModel saved to {SAVE_DIR}/")

"""
compare_models.py  –  Compare TF-IDF models vs DistilBERT on spam.csv
Run: python compare_models.py
Prints accuracy, precision, recall, F1 for each model.
Then saves the BEST model automatically.
"""

import pandas as pd
import numpy as np
import pickle
import re
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & preprocess data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Explainable Phishing Detector — Model Comparison")
print("=" * 60)

df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

print(f"\nDataset: {len(df)} rows  |  Ham: {(df.label_num==0).sum()}  Spam: {(df.label_num==1).sum()}")

# Classic NLP preprocessing (for TF-IDF models)
ps   = PorterStemmer()
stop = set(stopwords.words("english"))

def preprocess(text: str) -> str:
    text   = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [ps.stem(w) for w in tokens if w.isalnum() and w not in stop]
    return " ".join(tokens)

print("Preprocessing text for TF-IDF models ...")
df["proc"] = df["text"].apply(preprocess)

# Train / test split (one clean split for all)
X_proc_train, X_proc_test, X_raw_train, X_raw_test, y_train, y_test = train_test_split(
    df["proc"].values, df["text"].values, df["label_num"].values,
    test_size=0.2, random_state=42, stratify=df["label_num"]
)

# TF-IDF (bigrams, 5000 features — better than original 3000 / no bigrams)
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_proc_train)
X_test_tfidf  = tfidf.transform(X_proc_test)

# ─────────────────────────────────────────────────────────────────────────────
# 2. TF-IDF based models  (same approach as your original notebook)
# ─────────────────────────────────────────────────────────────────────────────
results = {}   # name → metrics dict

tfidf_models = {
    "Naive Bayes":          MultinomialNB(),
    "Logistic Regression":  LogisticRegression(max_iter=1000, class_weight="balanced"),
    "SVM":                  SVC(kernel="linear", class_weight="balanced", probability=True),
    "Random Forest":        RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
}

print("\n── TF-IDF Models ──────────────────────────────────────────")
for name, model in tfidf_models.items():
    t0 = time.time()
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    elapsed = time.time() - t0

    acc  = accuracy_score (y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score   (y_test, y_pred, pos_label=1, zero_division=0)
    f1   = f1_score       (y_test, y_pred, pos_label=1, zero_division=0)

    results[name] = dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                         time=elapsed, model_obj=model, type="tfidf")

    print(f"\n{name}")
    print(f"  Accuracy : {acc*100:.2f}%   |  Train time: {elapsed:.1f}s")
    print(f"  Precision: {prec*100:.2f}%  |  Recall: {rec*100:.2f}%  |  F1: {f1*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Ham","Spam"]))

# ─────────────────────────────────────────────────────────────────────────────
# 3. DistilBERT
# ─────────────────────────────────────────────────────────────────────────────
print("\n── DistilBERT ─────────────────────────────────────────────")
print("Fine-tuning DistilBERT (this takes a few minutes) ...")

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 128
BATCH_SIZE = 16
EPOCHS     = 4
LR         = 2e-5
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc    = tokenizer(list(texts), truncation=True, padding=True, max_length=MAX_LEN)
        self.labels = list(labels)
    def __len__(self):  return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

tokenizer  = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
train_ds   = SpamDataset(X_raw_train, y_train, tokenizer)
test_ds    = SpamDataset(X_raw_test,  y_test,  tokenizer)
train_ldr  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_ldr   = DataLoader(test_ds,  batch_size=BATCH_SIZE)

bert_model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
bert_model.to(device)

n_ham, n_spam  = (y_train == 0).sum(), (y_train == 1).sum()
weights        = torch.tensor([1.0, n_ham / n_spam], dtype=torch.float).to(device)
loss_fn        = torch.nn.CrossEntropyLoss(weight=weights)
optimizer      = AdamW(bert_model.parameters(), lr=LR)
total_steps    = len(train_ldr) * EPOCHS
scheduler      = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
)

t0 = time.time()
for epoch in range(EPOCHS):
    bert_model.train()
    total_loss = 0
    for batch in train_ldr:
        optimizer.zero_grad()
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labs = batch["labels"].to(device)
        out  = bert_model(input_ids=ids, attention_mask=mask)
        loss = loss_fn(out.logits, labs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}/{EPOCHS}  avg loss: {total_loss/len(train_ldr):.4f}")

bert_train_time = time.time() - t0

# Evaluate BERT
bert_model.eval()
all_preds, all_probs = [], []
with torch.no_grad():
    for batch in test_ldr:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out  = bert_model(input_ids=ids, attention_mask=mask)
        probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_preds.extend(preds)
        all_probs.extend(probs)

y_pred_bert = np.array(all_preds)
acc  = accuracy_score (y_test, y_pred_bert)
prec = precision_score(y_test, y_pred_bert, pos_label=1, zero_division=0)
rec  = recall_score   (y_test, y_pred_bert, pos_label=1, zero_division=0)
f1   = f1_score       (y_test, y_pred_bert, pos_label=1, zero_division=0)

results["DistilBERT"] = dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                              time=bert_train_time, model_obj=None, type="bert")

print(f"\nDistilBERT")
print(f"  Accuracy : {acc*100:.2f}%   |  Train time: {bert_train_time:.1f}s")
print(f"  Precision: {prec*100:.2f}%  |  Recall: {rec*100:.2f}%  |  F1: {f1*100:.2f}%")
print(classification_report(y_test, y_pred_bert, target_names=["Ham","Spam"]))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL COMPARISON")
print("=" * 60)
print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time':>8}")
print("-" * 72)

best_f1, best_name = 0, ""
for name, m in results.items():
    marker = " ◄ BEST" if m["f1"] == max(r["f1"] for r in results.values()) else ""
    print(
        f"{name:<22} {m['accuracy']*100:>9.2f}%"
        f" {m['precision']*100:>9.2f}%"
        f" {m['recall']*100:>9.2f}%"
        f" {m['f1']*100:>9.2f}%"
        f" {m['time']:>7.1f}s{marker}"
    )
    if m["f1"] > best_f1:
        best_f1, best_name = m["f1"], name

print("=" * 72)
print(f"\n🏆 Best model: {best_name}  (F1 = {best_f1*100:.2f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save best model
# ─────────────────────────────────────────────────────────────────────────────
import os
os.makedirs("./bert_model", exist_ok=True)

if best_name == "DistilBERT":
    bert_model.save_pretrained("./bert_model")
    tokenizer.save_pretrained("./bert_model")
    print("✅ DistilBERT saved to ./bert_model/")
else:
    best_obj = results[best_name]["model_obj"]
    pickle.dump(best_obj, open("best_model.pkl", "wb"))
    pickle.dump(tfidf,    open("tfidf_vectorizer.pkl", "wb"))
    print(f"✅ {best_name} saved to best_model.pkl + tfidf_vectorizer.pkl")

# Always save DistilBERT too (app.py always uses it)
if best_name != "DistilBERT":
    bert_model.save_pretrained("./bert_model")
    tokenizer.save_pretrained("./bert_model")
    print("✅ DistilBERT also saved to ./bert_model/ (used by app.py)")

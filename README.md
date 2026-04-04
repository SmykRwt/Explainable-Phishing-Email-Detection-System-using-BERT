# 🛡️ Explainable Phishing Email Detection System

DistilBERT-based spam/phishing detector with token-level explainability.

## 📁 Files
| File | Purpose |
|---|---|
| `train_bert.py` | Fine-tune DistilBERT on your spam.csv dataset |
| `predictor.py` | Inference engine + explainability + feature engineering |
| `app.py` | Streamlit web interface |
| `requirements.txt` | Dependencies |

## 🚀 Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Put your spam.csv in the same folder
#    (columns: v1=label, v2=text)

# 3. Train the model  (~10–20 min on CPU, ~3 min on GPU)
python train_bert.py

# 4. Launch the app
streamlit run app.py
```

## 🧠 Architecture

```
Input Text
    │
    ├──► DistilBERT (fine-tuned)  ──► Spam/Ham probability
    │         │
    │         └──► Gradient Attribution ──► Token importance chips
    │
    └──► Feature Engineering
              ├── URL length & domain mismatch
              ├── Urgency word patterns
              ├── Caps ratio, exclamation count
              └── Brand impersonation detection
```

## 🔍 Why BERT is better than TF-IDF + Random Forest

| | Old (TF-IDF + RF) | New (BERT) |
|---|---|---|
| Context understanding | ❌ Bag of words | ✅ Full sentence context |
| Short message detection | ❌ Needs 4-5 keywords | ✅ Works with 2-3 words |
| Preprocessing consistency | ❌ Train/predict mismatch | ✅ Tokenizer handles it |
| Explainability | ❌ None | ✅ Token attribution |
| URL awareness | ❌ None | ✅ Domain mismatch detection |

## 🐞 Bugs fixed from original code

1. **Preprocessing mismatch** — `app.py` used bare `re.sub` but model was trained on NLTK-stemmed tokens → fixed by BERT's own tokenizer
2. **No class balancing** — 6.5:1 ham:spam ratio made RF biased → fixed with `CrossEntropyLoss(weight=...)`
3. **No bigrams** — short spam messages missed → BERT captures n-gram context natively
4. **No URL/feature engineering** → added via `predictor.py`

"""
app.py  –  Explainable Phishing Email Detection System (BERT-based)
Run: streamlit run app.py

Requirements: pip install streamlit transformers torch
Also run train_bert.py first to generate ./bert_model/
"""

import streamlit as st
import re
from predictor import BERTPredictor

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🛡️ Phishing Detector",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
    .big-label { font-size: 1.3rem; font-weight: 700; margin-bottom: 4px; }
    .token-chip {
        display: inline-block; padding: 2px 10px; margin: 3px;
        border-radius: 12px; font-size: 0.85rem; font-weight: 600;
    }
    .reason-box {
        background: #000000; border-left: 4px solid #ffc107;
        color: #ffffff;
        padding: 8px 14px; margin: 6px 0; border-radius: 4px;
    }
    .url-box {
        background: #f8d7da; border-left: 4px solid #dc3545;
        padding: 8px 14px; margin: 6px 0; border-radius: 4px;
    }
    .safe-box {
        background: #0b2101; border-left: 4px solid #28a745;
        padding: 8px 14px; margin: 6px 0; border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return BERTPredictor("./bert_model")

predictor = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛡️ Explainable Phishing Email Detector")
st.markdown("Powered by **DistilBERT** · Feature Engineering · Token-level Explainability")
st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────
msg = st.text_area(
    "📩 Paste email / SMS text here:",
    height=180,
    placeholder="e.g. URGENT: Your account has been suspended. Click here to verify: http://secure-paypa1.com/verify",
)
analyze_btn = st.button("🔍 Analyze", type="primary")

# ── Analysis ──────────────────────────────────────────────────────────────────
if analyze_btn and msg.strip():
    full_text = msg
    result    = predictor.analyze(full_text)

    st.divider()
    # ── Verdict ───────────────────────────────────────────────────────────────
    label_color = "red" if result["label_idx"] == 1 else "green"
    st.markdown(
        f"<div class='big-label' style='color:{label_color};font-size:1.6rem'>"
        f"{result['label']}  —  {result['confidence']:.1f}% confidence"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Probability bars ──────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.metric("🚨 Spam probability",  f"{result['spam_prob']:.1f}%")
        st.progress(result["spam_prob"] / 100)
    with c2:
        st.metric("✅ Safe probability",  f"{result['ham_prob']:.1f}%")
        st.progress(result["ham_prob"] / 100)

    st.divider()

    # ── Reasons ───────────────────────────────────────────────────────────────
    st.subheader("🔍 Why this prediction?")

    if result["reasons"]:
        for reason in result["reasons"]:
            st.markdown(f"<div class='reason-box'>{reason}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='safe-box'>✅ No phishing patterns detected in text</div>", unsafe_allow_html=True)

    # ── URL flags ─────────────────────────────────────────────────────────────
    if result["url_flags"]:
        st.subheader("🌐 URL Analysis")
        for flag, detail in result["url_flags"].items():
            st.markdown(f"<div class='url-box'><b>{flag}</b>: <code>{detail}</code></div>", unsafe_allow_html=True)

    st.divider()

    # ── Token importance ──────────────────────────────────────────────────────
    st.subheader("🔬 Most Suspicious Words (Gradient Attribution)")
    st.caption("Larger / darker chip = stronger influence on the spam prediction")

    if result["top_tokens"]:
        max_score = result["top_tokens"][0]["score"] + 1e-9
        chips_html = ""
        for t in result["top_tokens"]:
            intensity = int(255 * (1 - t["score"] / max_score))
            if result["label_idx"] == 1:
                bg = "rgb(255," + str(intensity) + "," + str(intensity) + ")"
            else:
                bg = "rgb(" + str(intensity) + ",255," + str(intensity) + ")"
            word = t["token"].replace("##", "")
            score = t["score"]
            if not word.strip():
                continue
            chips_html += "<span class='token-chip' style='background:" + bg + ";' title='attribution: " + f"{score:.4f}" + "'>" + word + "</span>"
        st.markdown(chips_html, unsafe_allow_html=True)

    st.divider()
    st.caption("Model: DistilBERT fine-tuned on SMS Spam Collection · Explainability via gradient attribution · Feature engineering for URLs & urgency patterns")

elif analyze_btn:
    st.warning("Please enter some text to analyze.")

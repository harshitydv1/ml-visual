"""
Tab 1 — Text Cleaning
Show the preprocessing pipeline step-by-step with highlighted changes.
"""

import streamlit as st
from utils.nlp_helpers import clean_text


def render(text: str, lowercase: bool, remove_punct: bool, remove_sw: bool):
    st.markdown("## 🧹 Text Cleaning")
    st.info(
        "**What is text cleaning?**  \n"
        "Raw text contains noise — mixed casing, punctuation, extra spaces, and common "
        "words (stopwords) that add little meaning. Cleaning standardizes the text so "
        "downstream models focus on the *signal*, not the noise."
    )

    result = clean_text(text, lowercase, remove_punct, remove_sw)

    # ── Side-by-side: Original vs Cleaned ──
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📝 Original Text")
        st.code(result["original"], language=None)
    with col2:
        st.markdown("### ✨ Cleaned Text")
        st.code(result["cleaned"], language=None)

    # ── Step-by-step pipeline ──
    if result["steps"]:
        st.markdown("### 🔗 Pipeline Steps")
        for i, step in enumerate(result["steps"], 1):
            with st.expander(f"Step {i}: {step['name']}", expanded=(i == 1)):
                st.code(step["result"], language=None)

    # ── Character-level diff ──
    st.markdown("### 📊 Quick Stats")
    orig_len = len(result["original"])
    clean_len = len(result["cleaned"])
    removed = orig_len - clean_len
    c1, c2, c3 = st.columns(3)
    c1.metric("Original Length", f"{orig_len} chars")
    c2.metric("Cleaned Length", f"{clean_len} chars")
    c3.metric("Characters Removed", f"{removed}", delta=f"-{removed}", delta_color="inverse")

    st.success(
        "🔑 **Key takeaway:** Observe how each cleaning step progressively "
        "strips away noise while preserving meaningful content."
    )

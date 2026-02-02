"""
Tab 3 — Text Normalization
Side-by-side comparison of stemming vs lemmatization.
"""

import streamlit as st
import pandas as pd
from utils.nlp_helpers import tokenize, stem_and_lemmatize


def render(text: str):
    st.markdown("## 🔤 Text Normalization")
    st.info(
        "**What is normalization?**  \n"
        "Normalization reduces words to their base or root form. "
        "**Stemming** chops off suffixes using rules (fast but rough). "
        "**Lemmatization** uses vocabulary and grammar to find the true root (slower but accurate)."
    )

    tokens = tokenize(text, "Word")
    results = stem_and_lemmatize(tokens)

    # ── Comparison table ──
    st.markdown("### 🔍 Comparison Table")
    df = pd.DataFrame(results)
    df.columns = ["Original", "Stemmed", "Lemmatized"]

    # Highlight changed cells
    def highlight_changes(row):
        styles = [""] * 3
        if row["Stemmed"] != row["Original"].lower():
            styles[1] = "background-color: #FF6B6B33; font-weight: bold"
        if row["Lemmatized"] != row["Original"].lower():
            styles[2] = "background-color: #4ECDC433; font-weight: bold"
        return styles

    styled = df.style.apply(highlight_changes, axis=1)
    st.dataframe(styled, use_container_width=True, height=min(400, 40 + len(df) * 35))

    # ── Stats ──
    st.markdown("### 📊 Normalization Impact")
    stem_changed = sum(1 for r in results if r["stemmed"] != r["original"].lower())
    lemma_changed = sum(1 for r in results if r["lemmatized"] != r["original"].lower())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tokens", len(results))
    c2.metric("Changed by Stemming", stem_changed,
              delta=f"{stem_changed / max(len(results), 1) * 100:.0f}%")
    c3.metric("Changed by Lemmatization", lemma_changed,
              delta=f"{lemma_changed / max(len(results), 1) * 100:.0f}%")

    # ── Visual diff ──
    st.markdown("### 🎯 Words That Changed")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Stemming Changes**")
        for r in results:
            if r["stemmed"] != r["original"].lower():
                st.markdown(f"~~{r['original']}~~ → **{r['stemmed']}**")
    with col2:
        st.markdown("**Lemmatization Changes**")
        for r in results:
            if r["lemmatized"] != r["original"].lower():
                st.markdown(f"~~{r['original']}~~ → **{r['lemmatized']}**")

    st.success(
        "🔑 **Key takeaway:** Stemming is aggressive — it may produce non-words "
        "(e.g., 'running' → 'run', 'studies' → 'studi'). "
        "Lemmatization is smarter and returns real words."
    )

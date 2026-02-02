"""
Tab 5 — Vectorization (Bag of Words & TF-IDF)
Interactive heatmaps and numerical vector display.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from nltk.tokenize import sent_tokenize
from utils.nlp_helpers import vectorize


def render(original_text: str, cleaned_text: str, method: str):
    st.markdown("## 🔢 Vectorization")
    st.info(
        "**What is vectorization?**  \n"
        "Vectorization converts text into numerical arrays that models can process. "
        "**Bag of Words (BoW)** counts word occurrences. "
        "**TF-IDF** weighs words by how unique they are across documents — "
        "common words get lower scores."
    )

    # Use NLTK sentence tokenizer on the ORIGINAL text (which still has periods)
    sentences = [s.strip() for s in sent_tokenize(original_text) if s.strip()]

    if len(sentences) < 2:
        st.warning("⚠️ Please enter at least 2 sentences to see vectorization in action.")
        return

    # ── Vectorize ──
    result = vectorize(sentences, method)
    matrix = result["matrix"]
    features = result["feature_names"]

    st.markdown(f"### 📐 {method} Matrix")
    st.caption(f"Corpus: {len(sentences)} documents × {len(features)} features")

    # ── Heatmap ──
    doc_labels = [f"Doc {i+1}" for i in range(len(sentences))]
    df_matrix = pd.DataFrame(matrix, columns=features, index=doc_labels)

    fig = px.imshow(
        df_matrix,
        labels=dict(x="Feature (Token)", y="Document", color="Value"),
        color_continuous_scale="YlOrRd" if method == "BoW" else "Viridis",
        title=f"{method} Heatmap",
        aspect="auto",
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=max(300, len(sentences) * 60 + 100),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Raw vectors ──
    with st.expander("📋 View Raw Vectors"):
        for i, sentence in enumerate(sentences):
            st.markdown(f"**Doc {i+1}:** *\"{sentence}\"*")
            st.code(str(np.round(matrix[i], 4).tolist()), language=None)

    # ── Full matrix table ──
    with st.expander("📊 Full Matrix (Table View)"):
        st.dataframe(
            df_matrix.style.background_gradient(cmap="YlOrRd" if method == "BoW" else "viridis",
                                                  axis=None),
            use_container_width=True,
        )

    # ── Document sentences ──
    st.markdown("### 📄 Document Breakdown")
    for i, sentence in enumerate(sentences):
        st.markdown(f"**Doc {i+1}:** {sentence}")

    st.success(
        "🔑 **Key takeaway:** In BoW, values are raw counts. In TF-IDF, frequent words "
        "across all documents get *lower* scores, highlighting words that are unique "
        "to specific documents."
    )

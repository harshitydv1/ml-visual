"""
Tab 4 — Vocabulary Building
Display token-to-index mapping and frequency charts.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.nlp_helpers import tokenize, build_vocab


def render(text: str):
    st.markdown("## 📖 Vocabulary Building")
    st.info(
        "**What is a vocabulary?**  \n"
        "A vocabulary is a mapping from each unique token to a numerical index. "
        "This is how models 'see' words — as integers. "
        "The frequency of each token tells us which words dominate the text."
    )

    tokens = tokenize(text, "Word")
    vocab = build_vocab(tokens)

    # ── Vocab size ──
    st.markdown("### 📊 Vocabulary Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tokens", len(tokens))
    c2.metric("Vocabulary Size", len(vocab["token_to_index"]))
    c3.metric("Type-Token Ratio", f"{len(vocab['token_to_index']) / max(len(tokens), 1):.2f}")

    # ── Token-to-Index table ──
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🗂️ Token → Index Mapping")
        mapping_df = pd.DataFrame(
            list(vocab["token_to_index"].items()),
            columns=["Token", "Index"]
        )
        st.dataframe(mapping_df, use_container_width=True,
                      height=min(400, 40 + len(mapping_df) * 35))

    with col2:
        st.markdown("### 📊 Token Frequencies")
        freq = vocab["frequencies"]
        freq_df = pd.DataFrame(
            list(freq.items()), columns=["Token", "Frequency"]
        ).sort_values("Frequency", ascending=False)

        st.dataframe(freq_df, use_container_width=True,
                      height=min(400, 40 + len(freq_df) * 35))

    # ── Frequency bar chart ──
    st.markdown("### 📈 Top Token Frequencies")
    top_n = min(20, len(freq_df))
    top_df = freq_df.head(top_n)

    fig = px.bar(
        top_df, x="Token", y="Frequency",
        color="Frequency",
        color_continuous_scale="Viridis",
        title=f"Top {top_n} Most Frequent Tokens",
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_tickangle=-45,
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success(
        "🔑 **Key takeaway:** The Type-Token Ratio (TTR) measures lexical diversity. "
        "A ratio close to 1.0 means high diversity (many unique words); "
        "close to 0 means high repetition."
    )

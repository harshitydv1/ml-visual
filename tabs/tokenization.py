"""
Tab 2 — Tokenization
Display tokens as colored chips and show token statistics.
"""

import streamlit as st
from utils.nlp_helpers import tokenize


# Color palette for token chips
COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F8C471", "#82E0AA", "#F1948A", "#AED6F1", "#D7BDE2",
]


def _render_chips(tokens: list[str]):
    """Render tokens as colored HTML chips."""
    chips_html = ""
    for i, token in enumerate(tokens):
        color = COLORS[i % len(COLORS)]
        chips_html += (
            f'<span style="display:inline-block; background:{color}; color:#1a1a2e; '
            f'padding:6px 14px; margin:4px; border-radius:20px; '
            f'font-weight:600; font-size:0.9rem; '
            f'box-shadow: 0 2px 4px rgba(0,0,0,0.15);">'
            f'{token}</span>'
        )
    st.markdown(chips_html, unsafe_allow_html=True)


def render(text: str, token_method: str):
    st.markdown("## ✂️ Tokenization")
    st.info(
        "**What is tokenization?**  \n"
        "Tokenization splits raw text into smaller units called *tokens*. "
        "These can be words, sentences, or even subword pieces. "
        "It's the first step in converting text into a format models can understand."
    )

    tokens = tokenize(text, token_method)

    # ── Token chips ──
    st.markdown(f"### 🏷️ Tokens ({token_method} Tokenization)")
    _render_chips(tokens)

    # ── Stats ──
    st.markdown("### 📊 Token Statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tokens", len(tokens))
    c2.metric("Unique Tokens", len(set(tokens)))
    c3.metric("Avg Token Length", f"{sum(len(t) for t in tokens) / max(len(tokens), 1):.1f}")

    # ── Token list ──
    with st.expander("📋 View Token List"):
        for i, token in enumerate(tokens):
            st.text(f"[{i}] → \"{token}\"")

    st.success(
        "🔑 **Key takeaway:** Notice how different tokenization methods "
        "produce different numbers and types of tokens. Subword tokenization "
        "can handle unknown words by breaking them into pieces."
    )

"""
Tab 7 — Contextual Embeddings
Compare BERT embeddings of the same word in different contexts.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource(show_spinner="Loading BERT model (first time may take a minute)...")
def load_bert():
    """Load BERT tokenizer and model."""
    from transformers import AutoTokenizer, AutoModel
    import torch
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()
    return tokenizer, model


def get_word_embedding(sentence: str, target_word: str, tokenizer, model):
    """
    Get the contextual embedding of a target word within a sentence.
    Returns the average of all subword token embeddings for that word.
    """
    import torch

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state[0]  # (seq_len, 768)

    # Find subword tokens that belong to the target word
    target_lower = target_word.lower()
    target_indices = []
    for i, token in enumerate(tokens):
        clean_token = token.replace("##", "")
        if target_lower.startswith(clean_token) or clean_token == target_lower:
            target_indices.append(i)

    if not target_indices:
        # Fallback: search for partial matches
        for i, token in enumerate(tokens):
            if target_lower in token.replace("##", ""):
                target_indices.append(i)

    if not target_indices:
        return None, tokens

    embedding = hidden_states[target_indices].mean(dim=0).numpy()
    return embedding, tokens


def render(text: str):
    st.markdown("## 🌐 Contextual Embeddings")
    st.info(
        "**What are contextual embeddings?**  \n"
        "Unlike static embeddings (GloVe/Word2Vec), contextual embeddings from models "
        "like **BERT** generate *different* vectors for the same word depending on its "
        "surrounding context. For example, 'bank' in *'river bank'* vs *'bank account'* "
        "will have different embeddings."
    )

    try:
        tokenizer, model = load_bert()
    except Exception as e:
        st.error(f"Could not load BERT model: {e}")
        st.info("Install with: `pip install transformers torch`")
        return

    # ── User Input ──
    st.markdown("### ✏️ Compare the Same Word in Different Contexts")
    st.caption("Enter a target word and two sentences containing that word.")

    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        target_word = st.text_input("Target Word", value="bank", key="ctx_target")
    with col2:
        sentence1 = st.text_input("Sentence 1", value="I went to the river bank to fish.", key="ctx_s1")
    with col3:
        sentence2 = st.text_input("Sentence 2", value="I deposited money in the bank.", key="ctx_s2")

    if not target_word or not sentence1 or not sentence2:
        st.warning("Please fill in all three fields.")
        return

    # ── Get embeddings ──
    with st.spinner("Computing contextual embeddings..."):
        emb1, tokens1 = get_word_embedding(sentence1, target_word, tokenizer, model)
        emb2, tokens2 = get_word_embedding(sentence2, target_word, tokenizer, model)

    if emb1 is None or emb2 is None:
        st.error(f"Could not find '{target_word}' in one or both sentences. Try different wording.")
        return

    # ── Cosine similarity ──
    cos_sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]

    st.markdown("### 📊 Embedding Comparison")

    c1, c2, c3 = st.columns(3)
    c1.metric("Embedding Dimensions", emb1.shape[0])
    c2.metric("Cosine Similarity", f"{cos_sim:.4f}")
    c3.metric(
        "Context Impact",
        "High" if cos_sim < 0.85 else "Medium" if cos_sim < 0.95 else "Low",
        delta=f"{(1 - cos_sim) * 100:.1f}% difference",
    )

    # ── Embedding vector comparison ──
    st.markdown("### 📈 Embedding Dimensions Comparison")
    n_dims = min(50, len(emb1))  # Show first 50 dims for readability
    dim_df = pd.DataFrame({
        "Dimension": list(range(n_dims)),
        f"Sentence 1": emb1[:n_dims],
        f"Sentence 2": emb2[:n_dims],
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dim_df["Dimension"], y=dim_df["Sentence 1"],
        name=f"'{sentence1[:40]}...'",
        marker_color="#4ECDC4", opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        x=dim_df["Dimension"], y=dim_df["Sentence 2"],
        name=f"'{sentence2[:40]}...'",
        marker_color="#FF6B6B", opacity=0.7,
    ))
    fig.update_layout(
        barmode="overlay",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title=f"First {n_dims} Embedding Dimensions for '{target_word}'",
        xaxis_title="Dimension Index",
        yaxis_title="Value",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Difference heatmap ──
    st.markdown("### 🔥 Dimension-wise Difference")
    diff = np.abs(emb1 - emb2)
    top_diff_indices = np.argsort(diff)[-20:][::-1]
    top_diff_df = pd.DataFrame({
        "Dimension": top_diff_indices,
        "Sentence 1 Value": emb1[top_diff_indices].round(4),
        "Sentence 2 Value": emb2[top_diff_indices].round(4),
        "Absolute Difference": diff[top_diff_indices].round(4),
    })

    fig_diff = px.bar(
        top_diff_df, x="Dimension", y="Absolute Difference",
        color="Absolute Difference",
        color_continuous_scale="Reds",
        title="Top 20 Dimensions with Largest Differences",
    )
    fig_diff.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    st.plotly_chart(fig_diff, use_container_width=True)

    # ── Token display ──
    with st.expander("🔤 BERT Tokenization"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Sentence 1 tokens:** `{tokens1}`")
        with col2:
            st.markdown(f"**Sentence 2 tokens:** `{tokens2}`")

    st.success(
        "🔑 **Key takeaway:** If cosine similarity is high (~1.0), BERT sees the word "
        "similarly in both contexts. A lower similarity means the word's meaning differs "
        "significantly based on context — this is the power of contextual embeddings!"
    )

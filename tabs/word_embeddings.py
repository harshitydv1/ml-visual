"""
Tab 6 — Word Embeddings
Visualize local distributional word vectors using PCA, nearest neighbors, and cosine similarity.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from utils.nlp_helpers import tokenize


def build_cooccurrence_embeddings(tokens: list[str], window_size: int = 2) -> tuple[list[str], np.ndarray]:
    """Create simple distributional embeddings from word co-occurrences in the input text."""
    normalized_tokens = [token.lower() for token in tokens if token.strip()]
    vocabulary = list(dict.fromkeys(normalized_tokens))
    index = {word: idx for idx, word in enumerate(vocabulary)}
    vectors = np.zeros((len(vocabulary), len(vocabulary)), dtype=float)

    for position, word in enumerate(normalized_tokens):
        word_index = index[word]
        start = max(0, position - window_size)
        end = min(len(normalized_tokens), position + window_size + 1)
        for neighbor_position in range(start, end):
            if neighbor_position == position:
                continue
            neighbor_word = normalized_tokens[neighbor_position]
            neighbor_index = index[neighbor_word]
            distance = abs(neighbor_position - position)
            vectors[word_index, neighbor_index] += 1.0 / distance

    return vocabulary, vectors


def render(text: str):
    st.markdown("## 🧠 Word Embeddings")
    st.info(
        "**What are word embeddings?**  \n"
        "Unlike BoW/TF-IDF (sparse, high-dimensional), word embeddings map each word "
        "to a dense vector so related words end up near each other. "
        "This tab builds a lightweight local embedding from word co-occurrence patterns in the input text, "
        "which keeps deployment self-contained and fast."
    )

    tokens = tokenize(text, "Word")
    valid_tokens = [token for token in tokens if token.strip()]

    if len(valid_tokens) < 2:
        st.warning("⚠️ Not enough tokens to build word embeddings. Try entering a longer text.")
        return

    unique_tokens, vectors = build_cooccurrence_embeddings(valid_tokens)

    if len(unique_tokens) < 2:
        st.warning("⚠️ Not enough unique words to visualize embeddings.")
        return

    nonzero_rows = vectors.sum(axis=1) > 0
    unique_tokens = [token for token, keep in zip(unique_tokens, nonzero_rows) if keep]
    vectors = vectors[nonzero_rows]

    if len(unique_tokens) < 2:
        st.warning("⚠️ The selected text does not provide enough context for embeddings.")
        return

    st.markdown("### 🗺️ Word Embedding Space (PCA → 2D)")
    n_components = 2 if vectors.shape[1] >= 2 else 1
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(vectors)

    scatter_data = {
        "Word": unique_tokens,
        "PC1": coords[:, 0],
    }
    if n_components == 2:
        scatter_data["PC2"] = coords[:, 1]
    else:
        scatter_data["PC2"] = np.zeros(len(unique_tokens))

    scatter_df = pd.DataFrame(scatter_data)

    fig = px.scatter(
        scatter_df,
        x="PC1",
        y="PC2",
        text="Word",
        title="Words in 2D Embedding Space",
        color_discrete_sequence=["#4ECDC4"],
    )
    fig.update_traces(
        textposition="top center",
        marker=dict(size=12, line=dict(width=1, color="#1a1a2e")),
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔍 Nearest Neighbors")
    selected = st.selectbox("Select a word to find its nearest neighbors:", unique_tokens)
    if selected:
        selected_index = unique_tokens.index(selected)
        similarities = cosine_similarity(vectors[[selected_index]], vectors)[0]
        neighbor_frame = pd.DataFrame(
            {
                "Word": unique_tokens,
                "Cosine Similarity": similarities,
            }
        )
        neighbor_frame = neighbor_frame[neighbor_frame["Word"] != selected]
        neighbor_frame = neighbor_frame.sort_values("Cosine Similarity", ascending=False).head(10)
        neighbor_frame["Cosine Similarity"] = neighbor_frame["Cosine Similarity"].round(4)

        fig_nn = px.bar(
            neighbor_frame,
            x="Word",
            y="Cosine Similarity",
            color="Cosine Similarity",
            color_continuous_scale="Tealgrn",
            title=f"Top 10 Nearest Neighbors of '{selected}'",
        )
        fig_nn.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        st.plotly_chart(fig_nn, use_container_width=True)

    st.markdown("### 🔥 Cosine Similarity Heatmap")
    sim_matrix = cosine_similarity(vectors)
    sim_df = pd.DataFrame(sim_matrix, columns=unique_tokens, index=unique_tokens)

    fig_sim = px.imshow(
        sim_df,
        color_continuous_scale="RdYlGn",
        title="Pairwise Cosine Similarity",
        aspect="auto",
    )
    fig_sim.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=max(400, len(unique_tokens) * 30 + 100),
    )
    st.plotly_chart(fig_sim, use_container_width=True)

    st.success(
        "🔑 **Key takeaway:** Words that appear in similar contexts get similar vectors. "
        "This local embedding captures relationships from the current text without requiring external model downloads."
    )

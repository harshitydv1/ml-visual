"""
Tab 6 — Word Embeddings
Visualize dense word vectors using PCA, nearest neighbors, and cosine similarity.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from utils.nlp_helpers import tokenize


@st.cache_resource(show_spinner="Loading word embeddings model...")
def load_embedding_model():
    """Load a lightweight gensim embedding model."""
    import gensim.downloader as api
    # glove-wiki-gigaword-50 is ~66 MB — smallest available
    model = api.load("glove-wiki-gigaword-50")
    return model


def render(text: str):
    st.markdown("## 🧠 Word Embeddings")
    st.info(
        "**What are word embeddings?**  \n"
        "Unlike BoW/TF-IDF (sparse, high-dimensional), word embeddings map each word "
        "to a *dense* vector of fixed size (e.g., 50 dimensions). "
        "Words with similar meanings end up *close together* in this vector space. "
        "We use pre-trained **GloVe** embeddings (50 dimensions) here."
    )

    try:
        model = load_embedding_model()
    except Exception as e:
        st.error(f"Could not load embedding model: {e}")
        st.info("Run `pip install gensim` and ensure you have internet for the first download.")
        return

    tokens = tokenize(text, "Word")
    # Filter to tokens present in the model vocabulary
    valid_tokens = [t.lower() for t in tokens if t.lower() in model]
    invalid_tokens = [t for t in tokens if t.lower() not in model]

    if len(valid_tokens) < 2:
        st.warning("⚠️ Not enough tokens found in the GloVe vocabulary. Try using more common words.")
        return

    unique_tokens = list(dict.fromkeys(valid_tokens))  # preserve order, remove dups
    vectors = np.array([model[t] for t in unique_tokens])

    # ── 2D scatter via PCA ──
    st.markdown("### 🗺️ Word Embedding Space (PCA → 2D)")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    scatter_df = pd.DataFrame({
        "Word": unique_tokens,
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
    })

    fig = px.scatter(
        scatter_df, x="PC1", y="PC2", text="Word",
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

    # ── Nearest neighbors ──
    st.markdown("### 🔍 Nearest Neighbors")
    selected = st.selectbox("Select a word to find its nearest neighbors:", unique_tokens)
    if selected:
        neighbors = model.most_similar(selected, topn=10)
        neighbor_df = pd.DataFrame(neighbors, columns=["Word", "Cosine Similarity"])
        neighbor_df["Cosine Similarity"] = neighbor_df["Cosine Similarity"].round(4)

        fig_nn = px.bar(
            neighbor_df, x="Word", y="Cosine Similarity",
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

    # ── Cosine similarity matrix ──
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

    # ── Stats ──
    if invalid_tokens:
        with st.expander(f"⚠️ {len(set(invalid_tokens))} tokens not in GloVe vocabulary"):
            st.write(", ".join(set(invalid_tokens)))

    st.success(
        "🔑 **Key takeaway:** Words with similar meanings cluster together. "
        "For example, 'king' and 'queen' will be close in this space. "
        "The cosine similarity heatmap shows which word pairs are most related."
    )

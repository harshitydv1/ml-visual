"""
ML Visual — NLP Text Processing Pipeline Visualizer
A Streamlit app that walks through the full NLP pipeline with
interactive visualizations and intuitive explanations.
"""

import streamlit as st

# ── Page config ──
st.set_page_config(
    page_title="ML Visual — NLP Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for premium look ──
st.markdown("""
<style>
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Main header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 20px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e0e0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #2a2a4a;
    }

    /* Info/Success boxes */
    .stAlert {
        border-radius: 12px;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
    }

    /* Hide Streamlit footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Default example text ──
DEFAULT_TEXT = (
    "Natural Language Processing (NLP) is a fascinating field of artificial intelligence. "
    "It enables computers to understand, interpret, and generate human language. "
    "NLP powers many applications we use daily, including search engines, "
    "chatbots, translation services, and sentiment analysis tools. "
    "The field has been revolutionized by deep learning and transformer models, "
    "which can capture complex linguistic patterns and contextual relationships."
)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🧠 ML Visual")
    st.markdown("**NLP Pipeline Explorer**")
    st.markdown("---")

    # ── Input text ──
    st.markdown("### 📝 Input Text")
    input_text = st.text_area(
        "Enter your text below:",
        value=DEFAULT_TEXT,
        height=200,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ── Preprocessing toggles ──
    st.markdown("### ⚙️ Preprocessing")
    lowercase = st.toggle("Lowercase", value=True)
    remove_punct = st.toggle("Remove Punctuation", value=True)
    remove_stopwords = st.toggle("Remove Stopwords", value=False)

    st.markdown("---")

    # ── Tokenization method ──
    st.markdown("### ✂️ Tokenization")
    token_method = st.selectbox(
        "Tokenization Type",
        ["Word", "Sentence", "Subword"],
        index=0,
    )

    st.markdown("---")

    # ── Vectorization method ──
    st.markdown("### 🔢 Vectorization")
    vec_method = st.selectbox(
        "Vectorization Method",
        ["BoW", "TF-IDF"],
        index=0,
    )

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<p class="main-header">NLP Text Processing Pipeline</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Explore each step of the NLP pipeline with interactive visualizations</p>', unsafe_allow_html=True)

# Prepare cleaned text for downstream tabs
from utils.nlp_helpers import clean_text
cleaned_result = clean_text(input_text, lowercase, remove_punct, remove_stopwords)
cleaned_text = cleaned_result["cleaned"]

# ── Tabs ──
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🧹 Text Cleaning",
    "✂️ Tokenization",
    "🔤 Normalization",
    "📖 Vocabulary",
    "🔢 Vectorization",
    "🧠 Word Embeddings",
    "🌐 Contextual Embeddings",
])

# Import tab modules
from tabs import text_cleaning, tokenization, normalization, vocabulary
from tabs import vectorization, word_embeddings, contextual_embeddings

with tab1:
    text_cleaning.render(input_text, lowercase, remove_punct, remove_stopwords)

with tab2:
    tokenization.render(cleaned_text, token_method)

with tab3:
    normalization.render(cleaned_text)

with tab4:
    vocabulary.render(cleaned_text)

with tab5:
    vectorization.render(input_text, cleaned_text, vec_method)

with tab6:
    word_embeddings.render(cleaned_text)

with tab7:
    contextual_embeddings.render(cleaned_text)

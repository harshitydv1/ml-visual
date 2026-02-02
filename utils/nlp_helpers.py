"""
Shared NLP utility functions used across all tabs.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

# ---------------------------------------------------------------------------
# Ensure NLTK data is available
# ---------------------------------------------------------------------------

def ensure_nltk_data():
    """Download required NLTK datasets if not already present."""
    resources = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


ensure_nltk_data()

# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str, lowercase: bool = True, remove_punct: bool = True,
               remove_stopwords_flag: bool = False) -> dict:
    """
    Apply cleaning steps and return a dict with intermediate results.
    """
    steps = []
    current = text

    # Step 1 — Lowercase
    if lowercase:
        current = current.lower()
        steps.append({"name": "Lowercase", "result": current})

    # Step 2 — Remove punctuation
    if remove_punct:
        current = current.translate(str.maketrans("", "", string.punctuation))
        steps.append({"name": "Remove Punctuation", "result": current})

    # Step 3 — Collapse whitespace
    current = re.sub(r"\s+", " ", current).strip()
    steps.append({"name": "Normalize Whitespace", "result": current})

    # Step 4 — Remove stopwords (operates on tokens, returns joined string)
    if remove_stopwords_flag:
        stop_words = set(stopwords.words("english"))
        tokens = current.split()
        filtered = [t for t in tokens if t not in stop_words]
        current = " ".join(filtered)
        steps.append({"name": "Remove Stopwords", "result": current})

    return {"original": text, "cleaned": current, "steps": steps}


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(text: str, method: str = "Word") -> list:
    """Tokenize text using the chosen method."""
    if method == "Word":
        return word_tokenize(text)
    elif method == "Sentence":
        return sent_tokenize(text)
    elif method == "Subword":
        # Lazy import to avoid loading transformers unless needed
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            return tokenizer.tokenize(text)
        except Exception:
            return word_tokenize(text)  # fallback
    return word_tokenize(text)


# ---------------------------------------------------------------------------
# Stemming & Lemmatization
# ---------------------------------------------------------------------------

def stem_and_lemmatize(tokens: list[str]) -> dict:
    """Return stemmed and lemmatized versions of each token."""
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    results = []
    for token in tokens:
        results.append({
            "original": token,
            "stemmed": stemmer.stem(token),
            "lemmatized": lemmatizer.lemmatize(token),
        })
    return results


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def build_vocab(tokens: list[str]) -> dict:
    """Build token→index mapping and frequency counts."""
    from collections import Counter
    freq = Counter(tokens)
    sorted_tokens = sorted(set(tokens))
    token_to_index = {t: i for i, t in enumerate(sorted_tokens)}
    return {"token_to_index": token_to_index, "frequencies": dict(freq)}


# ---------------------------------------------------------------------------
# Vectorization
# ---------------------------------------------------------------------------

def vectorize(corpus: list[str], method: str = "BoW") -> dict:
    """
    Convert a list of documents to numerical vectors.
    Returns matrix, feature names, and the raw array.
    """
    if method == "TF-IDF":
        vec = TfidfVectorizer()
    else:
        vec = CountVectorizer()

    matrix = vec.fit_transform(corpus)
    feature_names = vec.get_feature_names_out()
    return {
        "matrix": matrix.toarray(),
        "feature_names": list(feature_names),
    }

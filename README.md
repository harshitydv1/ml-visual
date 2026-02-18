# 🧠 ML Visual — NLP Text Processing Pipeline

An interactive **Streamlit** application that walks you through the complete NLP pipeline — from raw text all the way to contextual embeddings — with intuitive visualizations and hands-on controls.

---

## ✨ Features

The app is organized into **7 interactive tabs**, each covering a key stage in the NLP lifecycle:

| Tab | Description |
|-----|-------------|
| 🧹 **Text Cleaning** | Lowercase, punctuation removal, and stopword filtering with live before/after diff |
| ✂️ **Tokenization** | Word, sentence, and subword tokenization with token-level highlights |
| 🔤 **Normalization** | Stemming and lemmatization side-by-side comparison |
| 📖 **Vocabulary** | Frequency distributions and vocabulary statistics |
| 🔢 **Vectorization** | Bag-of-Words and TF-IDF matrix visualizations |
| 🧠 **Word Embeddings** | Static word embeddings (Word2Vec via Gensim) with similarity exploration |
| 🌐 **Contextual Embeddings** | Transformer-based (HuggingFace) contextual representations |

The **sidebar** lets you tune every preprocessing option globally — changes cascade instantly through all downstream stages.

---

## 🏗️ Project Structure

```
ml-visual/
├── app.py                  # Main Streamlit entry point
├── download_nltk.py        # Pre-download NLTK resources (run once)
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version pin (3.12.10)
├── tabs/
│   ├── text_cleaning.py
│   ├── tokenization.py
│   ├── normalization.py
│   ├── vocabulary.py
│   ├── vectorization.py
│   ├── word_embeddings.py
│   └── contextual_embeddings.py
└── utils/
    └── nlp_helpers.py      # Shared text-processing utilities
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.12+**
- `pip`

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd ml-visual
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK resources (one-time setup)

```bash
python download_nltk.py
```

This downloads `punkt_tab`, `stopwords`, `wordnet`, and `averaged_perceptron_tagger_eng`.

### 5. Run the app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 📦 Dependencies

| Library | Purpose |
|---------|---------|
| `streamlit` | Web app framework |
| `nltk` | Tokenization, stopwords, stemming, lemmatization |
| `scikit-learn` | BoW & TF-IDF vectorization |
| `gensim` | Word2Vec word embeddings |
| `transformers` | HuggingFace contextual embeddings |
| `torch` | PyTorch backend for transformers |
| `plotly` | Interactive charts |
| `matplotlib` | Static visualizations |
| `numpy` / `pandas` | Data manipulation |

---

## ☁️ Deployment

### Render

1. Set the **start command** to:
   ```
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
2. Add a **pre-deploy command** (or build command):
   ```
   python download_nltk.py
   ```
3. Set the **Python version** in `runtime.txt` (set to `python-3.12.10`).

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss any major changes.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

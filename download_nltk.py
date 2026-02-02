"""
Download required NLTK data at startup (needed for Render deployment).
Run this once to pre-download all NLTK resources.
"""

import nltk

resources = [
    "punkt_tab",
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger_eng",
]

for r in resources:
    nltk.download(r)
    print(f"Downloaded: {r}")

print("All NLTK resources downloaded successfully!")

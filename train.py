"""
Train a TF-IDF + Logistic Regression sentiment classifier.

Dataset: NLTK movie_reviews corpus (1 000 positive + 1 000 negative reviews).
Artifacts saved to models/ as classifier.pkl and vectorizer.pkl.

Usage:
    python train.py
"""

import os
import pickle

import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

MODELS_DIR = "models"


def download_data() -> None:
    nltk.download("movie_reviews", quiet=True)


def load_corpus() -> tuple[list[str], list[str]]:
    """Load all reviews and their labels from the NLTK corpus."""
    texts, labels = [], []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            texts.append(movie_reviews.raw(fileid))
            labels.append(category)  # 'pos' or 'neg'
    return texts, labels


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=15_000,
        ngram_range=(1, 2),       # unigrams + bigrams
        stop_words="english",
        sublinear_tf=True,         # log-scale TF to dampen very frequent terms
        min_df=2,                  # ignore terms that appear in fewer than 2 docs
    )


def train() -> None:
    print("Downloading NLTK corpus...")
    download_data()

    print("Loading reviews...")
    texts, labels = load_corpus()
    print(f"  {len(texts)} documents loaded ({labels.count('pos')} pos / {labels.count('neg')} neg)")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("Fitting TF-IDF vectorizer...")
    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,} terms")

    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(max_iter=1_000, C=1.0, solver="lbfgs")
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*40}")
    print(f"Test accuracy: {acc:.1%}")
    print(f"{'='*40}")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(f"{MODELS_DIR}/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(f"{MODELS_DIR}/classifier.pkl", "wb") as f:
        pickle.dump(clf, f)

    print(f"Model artifacts saved to {MODELS_DIR}/")


if __name__ == "__main__":
    train()

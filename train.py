"""
Train a TF-IDF + Logistic Regression sentiment classifier.

Dataset: IMDb movie reviews via HuggingFace datasets library.
  - 25 000 training reviews (pos/neg balanced)
  - No external download needed on HF Spaces — dataset is cached on HF servers

Artifacts saved to models/ as classifier.pkl and vectorizer.pkl.

Usage:
    python train.py
"""

import os
import pickle

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

MODELS_DIR = "models"


def load_corpus() -> tuple[list[str], list[str]]:
    """Load IMDb reviews from HuggingFace datasets."""
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb", split="train", trust_remote_code=False)
    texts  = dataset["text"]
    # IMDb labels: 0 = neg, 1 = pos — convert to string labels
    labels = ["pos" if l == 1 else "neg" for l in dataset["label"]]
    return texts, labels


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=15_000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        min_df=2,
    )


def train() -> None:
    texts, labels = load_corpus()
    print(f"  {len(texts)} documents ({labels.count('pos')} pos / {labels.count('neg')} neg)")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("Fitting TF-IDF vectorizer...")
    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,} terms")

    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(max_iter=1_000, C=1.0, solver="lbfgs")
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n{'='*40}")
    print(f"Test accuracy: {acc:.1%}")
    print(f"{'='*40}")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(f"{MODELS_DIR}/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(f"{MODELS_DIR}/classifier.pkl", "wb") as f:
        pickle.dump(clf, f)

    print(f"Model saved to {MODELS_DIR}/")


if __name__ == "__main__":
    train()

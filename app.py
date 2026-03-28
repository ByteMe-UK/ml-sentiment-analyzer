"""
Sentiment Analyser — Gradio web interface.

Loads the pre-trained TF-IDF + LogisticRegression model and serves
sentiment predictions via an interactive Gradio UI.

Run locally:    python app.py
HuggingFace:    entry point is this file (app.py)
"""

import os
import pickle

import numpy as np
import gradio as gr

from train import train

MODELS_DIR = "models"
VECTORIZER_PATH = f"{MODELS_DIR}/vectorizer.pkl"
CLASSIFIER_PATH = f"{MODELS_DIR}/classifier.pkl"

LABEL_MAP = {"pos": "Positive 😊", "neg": "Negative 😞"}


def _ensure_model() -> None:
    """Train if model artifacts are missing (e.g. on HuggingFace Spaces cold start)."""
    if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(CLASSIFIER_PATH)):
        print("Model not found — training now...")
        train()
        print("Training complete.")


def _load_model() -> tuple:
    _ensure_model()
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(CLASSIFIER_PATH, "rb") as f:
        clf = pickle.load(f)
    return vectorizer, clf


vectorizer, clf = _load_model()


def predict(text: str) -> dict[str, float]:
    """Return confidence scores for each sentiment class."""
    if not text.strip():
        return {LABEL_MAP["pos"]: 0.0, LABEL_MAP["neg"]: 0.0}

    vec = vectorizer.transform([text])
    probas = clf.predict_proba(vec)[0]

    return {
        LABEL_MAP[cls]: float(prob)
        for cls, prob in zip(clf.classes_, probas)
    }


# ── Gradio UI ────────────────────────────────────────────────────────────────

EXAMPLES = [
    ["An absolute masterpiece. The performances were stunning and the story left me speechless."],
    ["Dreadful. Poorly written, badly acted, and a complete waste of my evening."],
    ["It was okay — nothing special, but not terrible either. Decent popcorn movie."],
    ["Genuinely one of the funniest films I've seen in years. Loved every minute."],
    ["The plot was a mess and the CGI looked cheap. I nearly walked out."],
]

with gr.Blocks(theme=gr.themes.Soft(), title="Sentiment Analyser") as demo:
    gr.Markdown(
        """
        # 🎬 Sentiment Analyser
        Classify any text as **Positive** or **Negative** using a trained ML model.

        **Model:** TF-IDF vectoriser + Logistic Regression
        **Trained on:** NLTK movie_reviews corpus (2 000 labelled reviews)
        **Accuracy:** ~85% on held-out test set
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                lines=5,
                placeholder="Type or paste any text here — a review, tweet, comment...",
                label="Input Text",
            )
            submit_btn = gr.Button("Analyse Sentiment", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=2, label="Sentiment Confidence")

    gr.Examples(
        examples=EXAMPLES,
        inputs=text_input,
        label="Try an example",
    )

    submit_btn.click(fn=predict, inputs=text_input, outputs=label_output)
    text_input.submit(fn=predict, inputs=text_input, outputs=label_output)

    gr.Markdown(
        """
        ---
        Built by [Laksh Menroy](https://github.com/lakshmenroy) · Part of the [ByteMe-UK](https://github.com/ByteMe-UK) portfolio
        """
    )

if __name__ == "__main__":
    demo.launch()

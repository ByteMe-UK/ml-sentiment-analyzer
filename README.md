# 🎬 ML Sentiment Analyser

A machine learning app that classifies any text as **Positive** or **Negative** — trained on 2 000 movie reviews using **TF-IDF + Logistic Regression**, served via a **Gradio** web interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-FF7C00?logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Live Demo

> 🔗 **[Try it on Hugging Face Spaces →](https://huggingface.co/spaces/bytemeuk/ml-sentiment-analyzer)**
>
> Type any text — a review, tweet, or comment — and get an instant sentiment prediction.

## 📸 Screenshots

> _Add a screenshot of the Gradio UI here after deployment_

## ✨ Features

- **Positive / Negative classification** with confidence scores
- **No GPU required** — fast CPU inference
- **Auto-trains on first run** — no pre-built binary needed
- **5 built-in examples** to explore immediately
- **Clean Gradio UI** — works in any browser

## 🧠 How It Works

```
Input text
    └─▶ TF-IDF Vectoriser (15k features, unigrams + bigrams)
            └─▶ Logistic Regression classifier
                    └─▶ Confidence scores: { Positive: 0.87, Negative: 0.13 }
```

**Training data:** NLTK `movie_reviews` corpus — 1 000 positive + 1 000 negative film reviews
**Test accuracy:** ~85%

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.10+ | Core language |
| scikit-learn | TF-IDF vectoriser + Logistic Regression |
| NLTK | Movie reviews training corpus |
| Gradio | Web interface |
| Hugging Face Spaces | Deployment |

## 📦 Getting Started

```bash
# Clone the repo
git clone https://github.com/ByteMe-UK/ml-sentiment-analyzer.git
cd ml-sentiment-analyzer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model (downloads NLTK corpus automatically)
python train.py

# Run the app
python app.py
```

Open `http://localhost:7860` in your browser.

> **Note:** If you skip `train.py`, the app will auto-train on first launch — it just takes ~30 seconds.

## 📁 Project Structure

```
ml-sentiment-analyzer/
├── app.py          ← Gradio web interface (entry point)
├── train.py        ← Training script — builds and saves model artifacts
├── models/         ← Saved model artifacts (gitignored)
│   ├── vectorizer.pkl
│   └── classifier.pkl
├── requirements.txt
├── LICENSE
└── README.md
```

## 🚢 Deployment (Hugging Face Spaces)

1. Push this repo to GitHub
2. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → **Create new Space**
3. Select **Gradio** as the SDK
4. Link your GitHub repo (`ByteMe-UK/ml-sentiment-analyzer`)
5. Set entry point to `app.py`
6. Click **Deploy** — Spaces auto-installs from `requirements.txt`

The app auto-trains on first boot since `models/` is gitignored.

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

**Part of the [ByteMe-UK](https://github.com/ByteMe-UK) portfolio collection.**

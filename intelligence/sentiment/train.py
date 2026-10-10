"""Train Outhit's own financial-news sentiment model.

Data: "Twitter Financial News Sentiment" (zeroshot on HuggingFace, MIT licence),
~12k finance headlines/tweets labelled bearish / bullish / neutral. We train on
the official train split and report metrics on the untouched validation split.

Model: TF-IDF (word 1-2 grams + character 3-5 grams) -> logistic regression.
Small, fast (<1 ms per headline), fully explainable, and runs offline once trained.

Run:  python -m ai.sentiment.train
"""

import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_URL = "https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/resolve/main/{split}.csv"
SPLITS = {"train": "sent_train", "valid": "sent_valid"}
LABELS = {0: "bearish", 1: "bullish", 2: "neutral"}

MODEL_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
METRICS_PATH = MODEL_DIR / "sentiment_metrics.json"


def clean(text: str) -> str:
    """Same normalisation at train and inference time: drop URLs, cashtags' $ and a trailing ' - Source'."""
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+-\s+[^-]{2,60}$", "", text)  # news-feed titles end with " - Publisher"
    text = text.replace("$", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def load(split: str) -> pd.DataFrame:
    cache = MODEL_DIR / f"{SPLITS[split]}.csv"
    if not cache.exists():
        r = httpx.get(DATA_URL.format(split=SPLITS[split]), timeout=60, follow_redirects=True)
        r.raise_for_status()
        cache.write_bytes(r.content)
    df = pd.read_csv(io.BytesIO(cache.read_bytes()))
    df["text"] = df["text"].astype(str).map(clean)
    return df


def build() -> Pipeline:
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=120_000)),
    ])
    return Pipeline([("features", features), ("clf", LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"))])


def main() -> dict:
    MODEL_DIR.mkdir(exist_ok=True)
    train, valid = load("train"), load("valid")
    model = build().fit(train["text"], train["label"])
    pred = model.predict(valid["text"])

    names = [LABELS[i] for i in sorted(LABELS)]
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "zeroshot/twitter-financial-news-sentiment (MIT)",
        "train_size": int(len(train)),
        "test_size": int(len(valid)),
        "accuracy": float(accuracy_score(valid["label"], pred)),
        "macro_f1": float(f1_score(valid["label"], pred, average="macro")),
        "per_class": {
            LABELS[int(k)]: {m: float(v) for m, v in vals.items()}
            for k, vals in classification_report(valid["label"], pred, output_dict=True).items()
            if k.isdigit()
        },
        "confusion_matrix": {"labels": names, "matrix": confusion_matrix(valid["label"], pred, labels=sorted(LABELS)).tolist()},
        "baseline_accuracy_majority_class": float((valid["label"] == valid["label"].mode()[0]).mean()),
    }
    joblib.dump({"model": model, "labels": LABELS}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    m = main()
    print(f"accuracy {m['accuracy']:.3f}  macro-F1 {m['macro_f1']:.3f}  (majority-class baseline {m['baseline_accuracy_majority_class']:.3f})")
    for k, v in m["per_class"].items():
        print(f"  {k:8s} precision {v['precision']:.2f} recall {v['recall']:.2f} f1 {v['f1-score']:.2f}")

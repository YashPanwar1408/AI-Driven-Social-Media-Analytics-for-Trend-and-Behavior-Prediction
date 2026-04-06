from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SentimentModel:
    pipeline: Pipeline
    label_names: list[str]

    def predict(self, text: str) -> tuple[str, float]:
        """Return (label, score) for the input text.

        The score is the predicted class probability (0..1) when available.
        """

        text = (text or "").strip()
        if not text:
            raise ValueError("text must be non-empty")

        # scikit-learn pipelines accept an iterable of strings
        pred = self.pipeline.predict([text])[0]

        # Prefer probabilities (LogisticRegression supports predict_proba)
        score = 1.0
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba([text])[0]
            pred_int = int(pred)
            if 0 <= pred_int < len(probs):
                score = float(probs[pred_int])
            else:
                score = float(np.max(probs))

        label = self.label_names[int(pred)] if int(pred) < len(self.label_names) else str(pred)
        return label, score


def _read_sentiment140_slice(
    csv_path: Path, *, nrows: int | None, skiprows: int
) -> pd.DataFrame:
    """Read a slice containing only sentiment and text."""

    df = pd.read_csv(
        csv_path,
        header=None,
        usecols=[0, 5],
        encoding="latin-1",
        nrows=nrows,
        skiprows=skiprows,
        low_memory=False,
    )
    df.columns = ["sentiment", "text"]
    return df


def train_sentiment_tfidf_model(
    *,
    csv_path: Path,
    sample_size: int,
    positive_start_row: int,
    max_features: int = 50_000,
    random_state: int = 42,
) -> SentimentModel:
    """Train a TF-IDF + Logistic Regression sentiment model on a balanced sample.

    This workspace's Sentiment140 file is typically ordered (negatives then positives),
    so we read half from the beginning and half from the positive section.
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    neg_n = sample_size // 2
    pos_n = sample_size - neg_n

    df_neg = _read_sentiment140_slice(csv_path, nrows=neg_n, skiprows=0)
    df_pos = _read_sentiment140_slice(csv_path, nrows=pos_n, skiprows=positive_start_row)

    df = pd.concat([df_neg, df_pos], ignore_index=True)
    df["text"] = df["text"].fillna("").astype(str)
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    df = df.dropna(subset=["sentiment"]).copy()

    # Prepare labels. For Sentiment140: {0,4} -> {0,1}
    unique = sorted(df["sentiment"].astype(int).unique().tolist())
    if set(unique).issubset({0, 4}):
        y = (df["sentiment"].astype(int) == 4).astype(int)
        label_names = ["negative", "positive"]
    else:
        # Generic multiclass: map each label to an index
        mapping = {label: i for i, label in enumerate(unique)}
        y = df["sentiment"].astype(int).map(mapping).astype(int)
        label_names = [str(label) for label in unique]

    X = df["text"]

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=max_features, lowercase=True)),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1000,
                    solver="saga",
                    random_state=random_state,
                ),
            ),
        ]
    )

    pipeline.fit(X, y)
    return SentimentModel(pipeline=pipeline, label_names=label_names)


def save_sentiment_artifact(model: SentimentModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model.pipeline, "label_names": model.label_names}, path)


def load_sentiment_artifact(path: Path) -> SentimentModel:
    obj = joblib.load(path)
    return SentimentModel(pipeline=obj["pipeline"], label_names=list(obj["label_names"]))


def load_or_train_sentiment_model(
    *,
    csv_path: Path,
    artifact_path: Path,
    auto_train: bool,
    sample_size: int,
    positive_start_row: int,
) -> SentimentModel:
    """Load sentiment model artifact, optionally training it if missing."""

    if artifact_path.exists():
        logger.info("Loading sentiment model artifact: %s", artifact_path)
        return load_sentiment_artifact(artifact_path)

    if not auto_train:
        raise FileNotFoundError(
            f"Sentiment model artifact not found: {artifact_path}. "
            "Run scripts/train_artifacts.py to create it (or set AUTO_TRAIN_ARTIFACTS=1)."
        )

    logger.warning("Sentiment artifact missing; training a new model (auto-train enabled)")
    model = train_sentiment_tfidf_model(
        csv_path=csv_path,
        sample_size=sample_size,
        positive_start_row=positive_start_row,
    )
    save_sentiment_artifact(model, artifact_path)
    return model

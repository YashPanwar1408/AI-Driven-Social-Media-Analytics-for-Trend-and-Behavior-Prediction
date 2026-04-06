from __future__ import annotations

"""Train and save model artifacts used by the FastAPI backend.

This creates files under ./artifacts/:
- sentiment_tfidf.joblib  (TF-IDF + Logistic Regression)
- hashtag_lstm.pt         (PyTorch LSTM forecaster)

Run:
  d:/aiml-social-analytics/.venv/Scripts/python.exe scripts/train_artifacts.py
"""

import logging

# Allow running this script directly (python scripts/train_artifacts.py)
# by adding the repository root to sys.path.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.config import get_settings
from api.services.forecast import save_lstm_artifact, train_lstm_artifact
from api.services.sentiment import save_sentiment_artifact, train_sentiment_tfidf_model


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    settings = get_settings()
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1) Sentiment TF-IDF model
    sentiment_model = train_sentiment_tfidf_model(
        csv_path=settings.csv_path,
        sample_size=settings.sentiment_sample_size,
        positive_start_row=settings.sentiment140_positive_start_row,
    )
    save_sentiment_artifact(sentiment_model, settings.sentiment_artifact)
    print(f"Saved sentiment artifact -> {settings.sentiment_artifact}")

    # 2) LSTM forecaster
    lstm_model, lstm_meta = train_lstm_artifact(
        csv_path=settings.csv_path,
        nrows=settings.lstm_nrows,
        group_freq=settings.lstm_group_freq,
        target_hashtag=settings.lstm_target_hashtag,
        lookback=settings.lstm_lookback,
        hidden_size=settings.lstm_hidden_size,
        num_layers=settings.lstm_num_layers,
        dropout=settings.lstm_dropout,
        epochs=settings.lstm_epochs,
        lr=settings.lstm_learning_rate,
        batch_size=settings.lstm_batch_size,
    )
    save_lstm_artifact(lstm_model, lstm_meta, settings.lstm_artifact)
    print(f"Saved LSTM artifact -> {settings.lstm_artifact}")


if __name__ == "__main__":
    main()

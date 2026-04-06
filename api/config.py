from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_optional_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value


@dataclass(frozen=True)
class Settings:
    # Paths
    csv_path: Path
    artifacts_dir: Path

    # Artifacts
    sentiment_artifact: Path
    lstm_artifact: Path

    # Data loading defaults
    trends_nrows: int | None

    # Optional auto-training (off by default for production)
    auto_train_artifacts: bool

    # Training defaults (used only when auto_train_artifacts=True)
    sentiment_sample_size: int
    sentiment140_positive_start_row: int

    lstm_nrows: int | None
    lstm_group_freq: str
    lstm_target_hashtag: str | None
    lstm_lookback: int
    lstm_hidden_size: int
    lstm_num_layers: int
    lstm_dropout: float
    lstm_epochs: int
    lstm_learning_rate: float
    lstm_batch_size: int


def get_settings() -> Settings:
    """Read app settings from environment variables (with safe defaults)."""

    csv_path = Path(_env_str("SOCIAL_CSV_PATH", "data/training.1600000.processed.noemoticon.csv"))
    artifacts_dir = Path(_env_str("ARTIFACTS_DIR", "artifacts"))

    sentiment_artifact = artifacts_dir / "sentiment_tfidf.joblib"
    lstm_artifact = artifacts_dir / "hashtag_lstm.pt"

    trends_nrows = _env_optional_int("TRENDS_NROWS", 200_000)

    auto_train_artifacts = _env_bool("AUTO_TRAIN_ARTIFACTS", False)

    sentiment_sample_size = _env_int("SENTIMENT_SAMPLE_SIZE", 100_000)
    sentiment140_positive_start_row = _env_int("SENTIMENT140_POSITIVE_START_ROW", 800_000)

    lstm_nrows = _env_optional_int("LSTM_NROWS", 200_000)
    lstm_group_freq = _env_str("LSTM_GROUP_FREQ", "D")
    lstm_target_hashtag = os.getenv("LSTM_TARGET_HASHTAG")  # optional

    lstm_lookback = _env_int("LSTM_LOOKBACK", 14)
    lstm_hidden_size = _env_int("LSTM_HIDDEN_SIZE", 64)
    lstm_num_layers = _env_int("LSTM_NUM_LAYERS", 2)
    lstm_dropout = float(_env_str("LSTM_DROPOUT", "0.1"))
    lstm_epochs = _env_int("LSTM_EPOCHS", 60)
    lstm_learning_rate = float(_env_str("LSTM_LEARNING_RATE", "0.001"))
    lstm_batch_size = _env_int("LSTM_BATCH_SIZE", 32)

    return Settings(
        csv_path=csv_path,
        artifacts_dir=artifacts_dir,
        sentiment_artifact=sentiment_artifact,
        lstm_artifact=lstm_artifact,
        trends_nrows=trends_nrows,
        auto_train_artifacts=auto_train_artifacts,
        sentiment_sample_size=sentiment_sample_size,
        sentiment140_positive_start_row=sentiment140_positive_start_row,
        lstm_nrows=lstm_nrows,
        lstm_group_freq=lstm_group_freq,
        lstm_target_hashtag=lstm_target_hashtag,
        lstm_lookback=lstm_lookback,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        lstm_dropout=lstm_dropout,
        lstm_epochs=lstm_epochs,
        lstm_learning_rate=lstm_learning_rate,
        lstm_batch_size=lstm_batch_size,
    )

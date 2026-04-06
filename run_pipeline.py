
"""Master end-to-end execution pipeline.

🎯 Goal
-------
Run the entire project end-to-end with ONE command.

This pipeline orchestrates the project you already built:
- Data preprocessing
- TF-IDF sentiment model (saved to /artifacts)
- BERT sentiment model (optional, saved to /artifacts)
- Trend detection
- LSTM forecasting (optional, saved to /artifacts)
- Visualizations (saved to /outputs/plots)

Commands
--------
Fast (recommended):
  python run_pipeline.py --fast
  - Skips heavy stages (BERT + LSTM)

Full:
  python run_pipeline.py --full
  - Runs everything (can be slow on CPU)

Outputs
-------
Everything is written under ./outputs/
- outputs/data/                preprocessed sample CSV
- outputs/trends/              trend JSON/CSV
- outputs/metrics/             model metrics JSON
- outputs/forecast/            sample forecast JSON (full mode)
- outputs/logs/pipeline.log     execution log
- outputs/plots/               PNG plots

Why this script forces Matplotlib to Agg
----------------------------------------
Your environment can be non-interactive (no GUI). Using the 'Agg' backend prevents
"FigureCanvasAgg is non-interactive" warnings and ensures plots are always saved.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Matplotlib backend fix (MUST be set before importing pyplot anywhere)
# ---------------------------------------------------------------------------

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# Ensure repo root is on sys.path so imports work even if executed from elsewhere.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Project imports (existing modules)
# ---------------------------------------------------------------------------

from api.config import get_settings
from api.services.sentiment import load_or_train_sentiment_model
from api.services.trends import TrendsService

import eda_social_media
import visualizations


# ---------------------------------------------------------------------------
# Configuration (kept simple + beginner-friendly)
# ---------------------------------------------------------------------------

# Preprocessing with NLTK is relatively slow; we do it on a sample.
PREPROCESS_NROWS = 20_000

# Evaluation sample for TF-IDF (balanced) used to compute metrics for plots.
EVAL_SAMPLE_SIZE = 20_000

# Trend detection parameters
TRENDS_TOP_N = 10
TRENDS_GROUP_FREQ = "D"
TRENDS_WINDOW = 60

# Plot parameters
PLOT_TOP_HASHTAGS = 5
PLOT_WINDOW_PERIODS = 60


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """JSON serializer that can handle datetimes and Paths."""

    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _setup_logging(outputs_dir: Path) -> logging.Logger:
    logs_dir = outputs_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "pipeline.log"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=handlers,
        force=True,
    )

    logger = logging.getLogger("run_pipeline")
    logger.info("Logging to %s", log_path)
    return logger


def _load_balanced_sentiment_sample(
    *,
    csv_path: Path,
    sample_size: int,
    positive_start_row: int,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, list[str]]:
    """Load a balanced (neg/pos) sample for evaluating sentiment models.

    Important: Sentiment140 training data is typically ordered: negatives then positives.
    This loader avoids the "single-class" problem by taking half from the start and
    half from the known positive section.

    Returns:
      X: text
      y: integer labels (0/1)
      label_names: human-readable labels
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    neg_n = sample_size // 2
    pos_n = sample_size - neg_n

    def _read_slice(*, nrows: int, skiprows: int) -> pd.DataFrame:
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

    df_neg = _read_slice(nrows=neg_n, skiprows=0)
    df_pos = _read_slice(nrows=pos_n, skiprows=positive_start_row)

    df = pd.concat([df_neg, df_pos], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    df["text"] = df["text"].fillna("").astype(str)
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    df = df.dropna(subset=["sentiment"]).copy()

    unique = sorted(df["sentiment"].astype(int).unique().tolist())
    if set(unique).issubset({0, 4}):
        y = (df["sentiment"].astype(int) == 4).astype(int)
        label_names = ["negative", "positive"]
    else:
        # Generic fallback: treat each unique label as a class id
        mapping = {label: i for i, label in enumerate(unique)}
        y = df["sentiment"].astype(int).map(mapping).astype(int)
        label_names = [str(label) for label in unique]

    X = df["text"]
    return X, y, label_names


def _compute_classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute accuracy/precision/recall/F1 in a sensible way for binary vs multiclass."""

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    y_true_arr = y_true.to_numpy()
    y_pred_arr = y_pred.to_numpy()

    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))

    labels = pd.unique(y_true)
    is_binary = len(labels) == 2

    if is_binary:
        precision = float(precision_score(y_true_arr, y_pred_arr, average="binary", pos_label=1, zero_division=0))
        recall = float(recall_score(y_true_arr, y_pred_arr, average="binary", pos_label=1, zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred_arr, average="binary", pos_label=1, zero_division=0))
    else:
        precision = float(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
        recall = float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def _maybe_train_or_load_bert(
    *,
    full: bool,
    artifacts_dir: Path,
    csv_path: Path,
    positive_start_row: int,
    outputs_dir: Path,
    logger: logging.Logger,
) -> dict[str, float] | None:
    """Train/load BERT artifact and return evaluation metrics.

    - In --fast mode: skip entirely.
    - In --full mode: train if missing; otherwise load.

    The artifact is saved to:
      artifacts/bert_sentiment/
    """

    if not full:
        logger.info("[BERT] Skipped (fast mode)")
        return None

    bert_dir = artifacts_dir / "bert_sentiment"
    meta_path = bert_dir / "meta.json"

    # Lazy imports so --fast runs without Transformers installed.
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        import bert_sentiment_model as bert_mod
    except Exception as exc:
        raise RuntimeError(
            "BERT stage requires torch + transformers. Install dependencies and rerun with --full."
        ) from exc

    cfg = bert_mod.Config(
        csv_path=csv_path,
        sample_size=10_000,
        sentiment140_positive_start_row=positive_start_row,
        epochs=1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[BERT] Device: %s", device)

    # Prepare evaluation dataset (balanced)
    df = bert_mod.load_dataset(cfg)
    df = bert_mod.handle_missing_values(df)
    y, label_names = bert_mod.prepare_labels(df)
    X = df["text"].astype(str)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    # Load or train
    if (bert_dir / "config.json").exists():
        logger.info("[BERT] Loading artifact from %s", bert_dir)
        tokenizer = AutoTokenizer.from_pretrained(bert_dir, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            label_names = list(meta.get("label_names", label_names))
    else:
        logger.info("[BERT] Training new model (this can be slow on CPU)")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=len(label_names),
        )

        train_loader, test_loader = bert_mod.make_dataloaders(
            cfg,
            tokenizer,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        bert_mod.train_bert(cfg, model, train_loader, device)

        # Save artifact for reuse
        bert_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(bert_dir)
        tokenizer.save_pretrained(bert_dir)

        meta = {
            "label_names": label_names,
            "base_model": cfg.model_name,
            "max_length": cfg.max_length,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("[BERT] Saved artifact -> %s", bert_dir)

    # Evaluate
    train_loader, test_loader = bert_mod.make_dataloaders(
        cfg,
        tokenizer,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    y_true_np, y_pred_np = bert_mod.predict_bert(model.to(device), test_loader, device)
    metrics = bert_mod.compute_metrics(y_true_np, y_pred_np)

    # Save a short report for viva
    report_path = outputs_dir / "metrics" / "bert_metrics.json"
    _save_json(report_path, {"metrics": metrics, "label_names": label_names})
    logger.info("[BERT] Metrics saved -> %s", report_path)

    return metrics


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(*, full: bool) -> None:
    settings = get_settings()

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logging(outputs_dir)

    logger.info("Pipeline mode: %s", "FULL" if full else "FAST")
    logger.info("CSV path: %s", settings.csv_path)
    logger.info("Artifacts dir: %s", settings.artifacts_dir)

    # Create output subfolders (nice and organized)
    (outputs_dir / "data").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "trends").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "forecast").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "plots").mkdir(parents=True, exist_ok=True)  # used by visualizations.py

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Load dataset (sample for analytics/plots/trends)
    # ------------------------------------------------------------------
    logger.info("[1/7] Loading dataset (nrows=%s)", settings.trends_nrows)
    df = visualizations.load_sentiment140(settings.csv_path, nrows=settings.trends_nrows)
    logger.info("Loaded rows: %s", len(df))

    # ------------------------------------------------------------------
    # 2) Preprocessing (create clean_text on a smaller sample)
    # ------------------------------------------------------------------
    logger.info("[2/7] Preprocessing text (sample_nrows=%s)", PREPROCESS_NROWS)
    df_prep = df.head(PREPROCESS_NROWS).copy()
    try:
        df_prep = eda_social_media.add_clean_text_column(
            df_prep,
            text_col="text",
            output_col="clean_text",
        )
        out_csv = outputs_dir / "data" / "preprocessed_sample.csv"
        df_prep[["sentiment", "timestamp", "text", "clean_text"]].to_csv(
            out_csv, index=False, encoding="utf-8"
        )
        logger.info("Saved preprocessed sample -> %s", out_csv)
    except Exception:
        logger.exception("Preprocessing failed")
        raise

    # ------------------------------------------------------------------
    # 3) Train/load TF-IDF model (artifact used by FastAPI)
    # ------------------------------------------------------------------
    logger.info("[3/7] Training/loading TF-IDF sentiment model")
    try:
        sentiment_model = load_or_train_sentiment_model(
            csv_path=settings.csv_path,
            artifact_path=settings.sentiment_artifact,
            auto_train=True,
            sample_size=settings.sentiment_sample_size,
            positive_start_row=settings.sentiment140_positive_start_row,
        )
        logger.info("TF-IDF artifact ready: %s", settings.sentiment_artifact)
    except Exception:
        logger.exception("TF-IDF model stage failed")
        raise

    # Evaluate TF-IDF for the model-comparison plot
    logger.info("Evaluating TF-IDF model for metrics plot")
    X_eval, y_eval, label_names = _load_balanced_sentiment_sample(
        csv_path=settings.csv_path,
        sample_size=EVAL_SAMPLE_SIZE,
        positive_start_row=settings.sentiment140_positive_start_row,
    )

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_eval,
        y_eval,
        test_size=0.2,
        random_state=42,
        stratify=y_eval,
    )

    y_pred = pd.Series(sentiment_model.pipeline.predict(X_test), index=y_test.index)
    tfidf_metrics = _compute_classification_metrics(y_test, y_pred)

    metrics_payload: dict[str, Any] = {
        "tfidf": {"metrics": tfidf_metrics, "label_names": label_names},
    }

    # ------------------------------------------------------------------
    # 4) Train/load BERT model (optional heavy stage)
    # ------------------------------------------------------------------
    logger.info("[4/7] BERT stage")
    bert_metrics = None
    try:
        bert_metrics = _maybe_train_or_load_bert(
            full=full,
            artifacts_dir=settings.artifacts_dir,
            csv_path=settings.csv_path,
            positive_start_row=settings.sentiment140_positive_start_row,
            outputs_dir=outputs_dir,
            logger=logger,
        )
        if bert_metrics is not None:
            metrics_payload["bert"] = {"metrics": bert_metrics}
    except Exception:
        logger.exception("BERT stage failed")
        raise

    # ------------------------------------------------------------------
    # 5) Trend detection (counts + top hashtags)
    # ------------------------------------------------------------------
    logger.info("[5/7] Trend detection")
    try:
        trends_service = TrendsService(csv_path=settings.csv_path, nrows=settings.trends_nrows)
        trends_summary = trends_service.get_trends(
            top_n=TRENDS_TOP_N,
            group_freq=TRENDS_GROUP_FREQ,
            window=TRENDS_WINDOW,
        )

        # Save summary JSON
        trends_json = outputs_dir / "trends" / f"trends_{TRENDS_GROUP_FREQ}.json"
        _save_json(trends_json, trends_summary)
        logger.info("Saved trends summary -> %s", trends_json)

        # Save raw counts CSV (useful for debugging and viva)
        counts_df = trends_service.get_counts_dataframe(TRENDS_GROUP_FREQ)
        counts_csv = outputs_dir / "trends" / f"hashtag_counts_{TRENDS_GROUP_FREQ}.csv"
        counts_df.to_csv(counts_csv, index=False, encoding="utf-8")
        logger.info("Saved hashtag counts -> %s", counts_csv)
    except Exception:
        logger.exception("Trend detection failed")
        raise

    # ------------------------------------------------------------------
    # 6) Train/load LSTM model (optional heavy stage)
    # ------------------------------------------------------------------
    logger.info("[6/7] LSTM stage")
    if full:
        try:
            from api.services.forecast import forecast as forecast_fn
            from api.services.forecast import load_or_train_lstm

            lstm_model, lstm_artifact = load_or_train_lstm(
                csv_path=settings.csv_path,
                nrows=settings.lstm_nrows,
                artifact_path=settings.lstm_artifact,
                auto_train=True,
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
            logger.info("LSTM artifact ready: %s", settings.lstm_artifact)

            # Save a sample forecast output (nice demo artifact)
            cached_counts = None
            try:
                cached_counts = trends_service.get_counts_dataframe(lstm_artifact.group_freq)
            except Exception:
                cached_counts = None

            forecast_result = forecast_fn(
                model=lstm_model,
                artifact=lstm_artifact,
                csv_path=settings.csv_path,
                nrows=settings.lstm_nrows,
                steps=14,
                counts=cached_counts,
            )

            forecast_json = outputs_dir / "forecast" / "forecast.json"
            _save_json(forecast_json, forecast_result)
            logger.info("Saved sample forecast -> %s", forecast_json)

            # Save LSTM metadata (helpful for viva)
            lstm_meta_json = outputs_dir / "forecast" / "lstm_artifact_meta.json"
            _save_json(lstm_meta_json, asdict(lstm_artifact))
        except Exception:
            logger.exception("LSTM stage failed")
            raise
    else:
        logger.info("[LSTM] Skipped (fast mode)")

    # ------------------------------------------------------------------
    # 7) Visualizations (saved as PNG files)
    # ------------------------------------------------------------------
    logger.info("[7/7] Generating visualizations")
    try:
        # 1) Sentiment distribution
        visualizations.plot_sentiment_distribution(df)

        # 2) Trend over time (line plot) - use the already computed counts
        visualizations.plot_hashtag_trends(
            counts_df,
            top_n=PLOT_TOP_HASHTAGS,
            window_periods=PLOT_WINDOW_PERIODS,
            title=f"Top {PLOT_TOP_HASHTAGS} hashtag trends ({TRENDS_GROUP_FREQ})",
        )

        # 3) Model comparison (TF-IDF always, BERT only in full mode)
        plot_metrics: dict[str, dict[str, float]] = {"TF-IDF": tfidf_metrics}
        if bert_metrics is not None:
            plot_metrics["BERT"] = bert_metrics

        visualizations.plot_model_comparison(
            plot_metrics,
            metrics_to_plot=("accuracy", "f1"),
            title="Model comparison (accuracy & F1)",
        )

        # Save metrics JSON used for the bar chart
        metrics_json = outputs_dir / "metrics" / "model_metrics.json"
        _save_json(metrics_json, metrics_payload)
        logger.info("Saved metrics -> %s", metrics_json)

    except Exception:
        logger.exception("Visualization stage failed")
        raise

    logger.info("✅ Pipeline completed successfully")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full social analytics pipeline")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fast",
        action="store_true",
        help="Run a fast pipeline (skip BERT + LSTM)",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Run the full pipeline (includes BERT + LSTM; slow on CPU)",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Default behavior: FAST (safer for beginners)
    full = bool(args.full)
    run_pipeline(full=full)


if __name__ == "__main__":
    main()

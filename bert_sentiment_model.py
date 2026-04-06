"""BERT sentiment model (Transformers) + TF-IDF baseline comparison.

This script trains two models on the same train/test split:
1) TF-IDF + Logistic Regression baseline (fast)
2) Fine-tuned BERT sequence classifier (slower, usually stronger)

Dataset in this workspace:
- data/training.1600000.processed.noemoticon.csv (Sentiment140-style)
  Columns (no header): sentiment, id, timestamp, query, user, text
  Labels: 0=negative, 4=positive

Why the script loads a balanced sample by default:
- The Sentiment140 training file is typically ordered with all negatives first,
  then all positives. If you read only the first N rows, you may get only one class.

Requirements:
- pip install torch transformers pandas scikit-learn

Notes:
- You have CPU-only torch installed. BERT fine-tuning on CPU is slow.
  For faster runs:
  - Reduce SAMPLE_SIZE (e.g., 5_000)
  - Reduce EPOCHS to 1
  - Use a smaller BERT model (MODEL_NAME = "prajjwal1/bert-tiny")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Config:
    # Data
    csv_path: Path = Path("data") / "training.1600000.processed.noemoticon.csv"

    # Balanced sampling (recommended for this ordered Sentiment140 file)
    # Set to None to read the full dataset (very large)
    sample_size: int | None = 20_000
    sentiment140_positive_start_row: int = 800_000

    # Train/test split
    test_size: float = 0.2
    random_state: int = 42

    # Baseline model
    tfidf_max_features: int = 50_000

    # BERT model
    # Use a small BERT by default so CPU training is feasible.
    # This particular checkpoint also loads cleanly without extra tokenizer deps.
    # Switch to "bert-base-uncased" if you have a GPU and want higher accuracy.
    model_name: str = "google/bert_uncased_L-2_H-128_A-2"
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # Logging
    log_every_n_steps: int = 50


def set_seed(seed: int) -> None:
    """Make results more reproducible."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_sentiment140_slice(
    csv_path: Path, *, nrows: int | None = None, skiprows: int = 0
) -> pd.DataFrame:
    """Read a slice of a Sentiment140 CSV (only sentiment + text columns)."""

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


def load_dataset(cfg: Config) -> pd.DataFrame:
    """Load the dataset (balanced sample by default)."""

    if not cfg.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {cfg.csv_path.resolve()}")

    if cfg.sample_size is None:
        df = _read_sentiment140_slice(cfg.csv_path)
        return df

    # Balanced sample: half negatives from top, half positives from the positive section.
    neg_n = cfg.sample_size // 2
    pos_n = cfg.sample_size - neg_n

    df_neg = _read_sentiment140_slice(cfg.csv_path, nrows=neg_n, skiprows=0)
    df_pos = _read_sentiment140_slice(
        cfg.csv_path,
        nrows=pos_n,
        skiprows=cfg.sentiment140_positive_start_row,
    )

    df = pd.concat([df_neg, df_pos], ignore_index=True)
    df = df.sample(frac=1.0, random_state=cfg.random_state).reset_index(drop=True)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing text and drop missing sentiment."""

    df = df.copy()
    df["text"] = df["text"].fillna("")
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    df = df.dropna(subset=["sentiment"])
    return df


def prepare_labels(df: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    """Convert the dataset's sentiment labels to model-friendly integers.

    Supports:
    - Sentiment140 binary labels {0,4} -> {0,1}
    - Optional 3-class {0,2,4} -> {0,1,2}

    Returns:
    - y: integer labels
    - label_names: human-readable names aligned with label ids
    """

    unique = sorted(df["sentiment"].astype(int).unique().tolist())

    # Sentiment140 binary
    if set(unique).issubset({0, 4}):
        y = (df["sentiment"].astype(int) == 4).astype(int)
        label_names = ["negative", "positive"]
        return y, label_names

    # Common 3-class convention
    if set(unique).issubset({0, 2, 4}):
        mapping = {0: 0, 2: 1, 4: 2}
        y = df["sentiment"].astype(int).map(mapping).astype(int)
        label_names = ["negative", "neutral", "positive"]
        return y, label_names

    # Generic fallback: treat each unique label as a class id
    unique_sorted = sorted(unique)
    mapping = {label: i for i, label in enumerate(unique_sorted)}
    y = df["sentiment"].astype(int).map(mapping).astype(int)
    label_names = [str(label) for label in unique_sorted]
    return y, label_names


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute accuracy/precision/recall/F1 with sensible averaging."""

    accuracy = accuracy_score(y_true, y_pred)

    labels = np.unique(y_true)
    is_binary = len(labels) == 2

    if is_binary:
        precision = precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    else:
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def print_eval(name: str, y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> dict[str, float]:
    """Print required metrics + classification report."""

    metrics = compute_metrics(y_true, y_pred)

    print(f"\n=== {name} Evaluation ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    return metrics


def train_tfidf_baseline(
    cfg: Config,
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    label_names: list[str],
) -> dict[str, float]:
    """Train a simple TF-IDF + Logistic Regression model and evaluate it."""

    baseline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=cfg.tfidf_max_features, lowercase=True)),
            (
                "logreg",
                # 'saga' supports sparse text features and multiclass.
                LogisticRegression(max_iter=1000, solver="saga", random_state=cfg.random_state),
            ),
        ]
    )

    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)

    return print_eval("TF-IDF + Logistic Regression", y_test.to_numpy(), y_pred, label_names)


class TextDataset(Dataset):
    """Torch Dataset that tokenizes text on-the-fly for BERT."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenize:
        # - truncation=True: cut off very long posts
        # - padding='max_length': makes every example the same length for batching
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # encoded is a dict of tensors with an extra batch dim (1, seq_len)
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def make_dataloaders(
    cfg: Config,
    tokenizer: AutoTokenizer,
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
) -> tuple[DataLoader, DataLoader]:
    """Create DataLoaders for training and evaluation."""

    train_ds = TextDataset(
        texts=X_train.astype(str).tolist(),
        labels=y_train.astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )
    test_ds = TextDataset(
        texts=X_test.astype(str).tolist(),
        labels=y_test.astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, test_loader


def train_bert(
    cfg: Config,
    model: AutoModelForSequenceClassification,
    train_loader: DataLoader,
    device: torch.device,
) -> None:
    """Fine-tune BERT with a simple PyTorch training loop."""

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item())

            if step % cfg.log_every_n_steps == 0 or step == len(train_loader):
                avg_loss = running_loss / step
                print(
                    f"Epoch {epoch}/{cfg.epochs} | Step {step}/{len(train_loader)} | Loss {avg_loss:.4f}"
                )

        elapsed = time.time() - start
        print(f"Epoch {epoch} finished in {elapsed:.1f}s")


@torch.no_grad()
def predict_bert(
    model: AutoModelForSequenceClassification,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return (y_true, y_pred)."""

    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []

    for batch in data_loader:
        labels = batch["labels"].numpy().tolist()
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

        outputs = model(**inputs)
        preds = outputs.logits.argmax(dim=1).cpu().numpy().tolist()

        y_true.extend(labels)
        y_pred.extend(preds)

    return np.array(y_true), np.array(y_pred)


def explain_bert_vs_tfidf() -> None:
    """Brief explanation (in plain English) of why BERT often beats TF-IDF."""

    print("\n=== Why BERT often performs better than TF-IDF ===")
    print(
        "- TF-IDF is a bag-of-words representation: it mostly counts which words appear.\n"
        "  It ignores word order and has limited ability to model context (e.g., negation).\n"
        "- BERT reads text as a sequence and builds *contextual* embeddings: the same word\n"
        "  can have different meaning depending on surrounding words.\n"
        "- BERT is pretrained on large corpora, so it starts with strong language knowledge\n"
        "  and needs fewer labeled examples to generalize well (transfer learning).\n"
        "- BERT uses subword tokenization, helping with misspellings and rare words.\n\n"
        "Caveat: BERT is much more compute-heavy; on CPU or with tiny models, gains may be smaller."
    )


def main() -> None:
    cfg = Config()
    set_seed(cfg.random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) Load + clean dataset
    df = load_dataset(cfg)
    df = handle_missing_values(df)

    # 2) Prepare labels
    y, label_names = prepare_labels(df)
    X = df["text"].astype(str)

    label_counts = y.value_counts()
    print("\nLabel distribution:")
    print(label_counts)
    if label_counts.shape[0] < 2:
        raise ValueError(
            "Only one class found in the loaded data. "
            "Increase sample_size or set sample_size=None to load more rows."
        )

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )
    print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")

    # --- Baseline model ---
    tfidf_metrics = train_tfidf_baseline(
        cfg, X_train, y_train, X_test, y_test, label_names
    )

    # --- BERT fine-tuning ---
    print("\nLoading tokenizer + pretrained BERT...")
    # Use a *slow* tokenizer for maximum compatibility (no `tokenizers`/protobuf requirements).
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False)

    num_labels = len(label_names)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=num_labels,
    )

    train_loader, test_loader = make_dataloaders(
        cfg, tokenizer, X_train, y_train, X_test, y_test
    )

    print("\nFine-tuning BERT...")
    train_bert(cfg, model, train_loader, device)

    y_true, y_pred = predict_bert(model, test_loader, device)
    bert_metrics = print_eval("BERT (fine-tuned)", y_true, y_pred, label_names)

    # --- Comparison summary ---
    print("\n=== Summary (higher is better) ===")
    print(
        "Model\t\tAccuracy\tPrecision\tRecall\t\tF1\n"
        f"TF-IDF\t\t{tfidf_metrics['accuracy']:.4f}\t\t{tfidf_metrics['precision']:.4f}\t\t{tfidf_metrics['recall']:.4f}\t\t{tfidf_metrics['f1']:.4f}\n"
        f"BERT\t\t{bert_metrics['accuracy']:.4f}\t\t{bert_metrics['precision']:.4f}\t\t{bert_metrics['recall']:.4f}\t\t{bert_metrics['f1']:.4f}"
    )

    explain_bert_vs_tfidf()


if __name__ == "__main__":
    main()

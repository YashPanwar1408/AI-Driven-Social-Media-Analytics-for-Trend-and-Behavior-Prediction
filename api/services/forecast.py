from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .hashtag_data import (
    build_hashtag_series,
    compute_hashtag_counts,
    extract_hashtag_rows,
    load_timestamp_text_csv_sentiment140,
)

logger = logging.getLogger(__name__)


# ----------------------------
# Scaling + sequences
# ----------------------------


def minmax_scale(values: np.ndarray, *, vmin: float, vmax: float) -> np.ndarray:
    """Scale values to [0, 1] using provided min/max."""

    if vmax - vmin < 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def minmax_fit(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Fit min-max scaling on values and return (scaled, vmin, vmax)."""

    vmin = float(values.min())
    vmax = float(values.max())
    return minmax_scale(values, vmin=vmin, vmax=vmax), vmin, vmax


def minmax_inverse(scaled: np.ndarray, *, vmin: float, vmax: float) -> np.ndarray:
    if vmax - vmin < 1e-9:
        return np.full_like(scaled, fill_value=vmin, dtype=np.float32)
    return (scaled * (vmax - vmin) + vmin).astype(np.float32)


def make_sequences(series: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i : i + lookback])
        y.append(series[i + lookback])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # LSTM input: (batch, seq_len, features)
        x = self.X[idx].unsqueeze(-1)
        return x, self.y[idx]


# ----------------------------
# Model
# ----------------------------


class LSTMForecaster(nn.Module):
    def __init__(self, *, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        pred = self.fc(last)
        return pred.squeeze(-1)


@dataclass(frozen=True)
class LSTMArtifact:
    hashtag: str
    group_freq: str
    lookback: int

    hidden_size: int
    num_layers: int
    dropout: float

    vmin: float
    vmax: float


def _choose_hashtag(counts: pd.DataFrame, *, lookback: int, target: str | None) -> str:
    if target:
        return target.lower().lstrip("#")

    # Pick a hashtag with enough periods to support training
    min_periods = lookback + 6
    summary = (
        counts.groupby("hashtag")
        .agg(total_count=("count", "sum"), n_periods=("period_start", "nunique"))
        .sort_values(["total_count", "n_periods"], ascending=[False, False])
    )
    candidates = summary[summary["n_periods"] >= min_periods]
    if candidates.empty:
        raise ValueError(
            f"No hashtag has enough history (need >= {min_periods} periods). "
            "Increase nrows, reduce lookback, or use weekly grouping."
        )

    return str(candidates.index[0])


def _load_counts(
    *,
    csv_path: Path,
    nrows: int | None,
    group_freq: str,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    raw = load_timestamp_text_csv_sentiment140(csv_path, nrows=nrows)
    hashtag_rows = extract_hashtag_rows(raw)
    counts = compute_hashtag_counts(hashtag_rows, group_freq=group_freq)
    return counts


def train_lstm_artifact(
    *,
    csv_path: Path,
    nrows: int | None,
    group_freq: str,
    target_hashtag: str | None,
    lookback: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int = 42,
    train_ratio: float = 0.8,
) -> tuple[LSTMForecaster, LSTMArtifact]:
    """Train an LSTM forecaster on one hashtag's grouped counts."""

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    group_freq = group_freq.upper()
    if group_freq not in {"D", "W"}:
        raise ValueError("group_freq must be 'D' or 'W'")

    counts = _load_counts(csv_path=csv_path, nrows=nrows, group_freq=group_freq)
    chosen = _choose_hashtag(counts, lookback=lookback, target=target_hashtag)

    series_obj = build_hashtag_series(counts, group_freq=group_freq, hashtag=chosen)
    series = series_obj.series

    if len(series) <= lookback + 6:
        raise ValueError(
            f"Not enough time points ({len(series)}) for lookback={lookback}. "
            "Increase nrows, reduce lookback, or change grouping."
        )

    values = series.values.astype(np.float32)
    scaled, vmin, vmax = minmax_fit(values)

    X, y = make_sequences(scaled, lookback)

    # Time-based split
    train_size = int(len(scaled) * train_ratio)
    train_end = max(1, train_size - lookback)

    X_train, y_train = X[:train_end], y[:train_end]

    # Shuffle training sequences (safe, because each sample is a fixed window)
    idx = rng.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    train_loader = DataLoader(
        SequenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            logger.info("LSTM epoch %s/%s | train MSE %.6f", epoch, epochs, float(np.mean(losses)))

    artifact = LSTMArtifact(
        hashtag=chosen,
        group_freq=group_freq,
        lookback=lookback,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        vmin=vmin,
        vmax=vmax,
    )

    return model, artifact


def save_lstm_artifact(model: LSTMForecaster, artifact: LSTMArtifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact": artifact.__dict__,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def load_lstm_artifact(path: Path) -> tuple[LSTMForecaster, LSTMArtifact]:
    payload = torch.load(path, map_location="cpu")
    meta = payload["artifact"]
    artifact = LSTMArtifact(**meta)
    model = LSTMForecaster(
        hidden_size=artifact.hidden_size,
        num_layers=artifact.num_layers,
        dropout=artifact.dropout,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, artifact


def load_or_train_lstm(
    *,
    csv_path: Path,
    nrows: int | None,
    artifact_path: Path,
    auto_train: bool,
    group_freq: str,
    target_hashtag: str | None,
    lookback: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    lr: float,
    batch_size: int,
) -> tuple[LSTMForecaster, LSTMArtifact]:
    if artifact_path.exists():
        logger.info("Loading LSTM artifact: %s", artifact_path)
        return load_lstm_artifact(artifact_path)

    if not auto_train:
        raise FileNotFoundError(
            f"LSTM artifact not found: {artifact_path}. "
            "Run scripts/train_artifacts.py to create it (or set AUTO_TRAIN_ARTIFACTS=1)."
        )

    logger.warning("LSTM artifact missing; training a new model (auto-train enabled)")
    model, artifact = train_lstm_artifact(
        csv_path=csv_path,
        nrows=nrows,
        group_freq=group_freq,
        target_hashtag=target_hashtag,
        lookback=lookback,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )
    save_lstm_artifact(model, artifact, artifact_path)
    return model, artifact


@torch.no_grad()
def _predict_scaled(model: LSTMForecaster, X: np.ndarray, device: torch.device) -> np.ndarray:
    model = model.to(device)
    model.eval()
    xb = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
    preds = model(xb).cpu().numpy().astype(np.float32)
    return preds


def forecast(
    *,
    model: LSTMForecaster,
    artifact: LSTMArtifact,
    csv_path: Path,
    nrows: int | None,
    steps: int,
    counts: pd.DataFrame | None = None,
    train_ratio: float = 0.8,
) -> dict:
    """Build history + test predictions + future forecast for the trained hashtag."""

    if counts is None:
        counts = _load_counts(csv_path=csv_path, nrows=nrows, group_freq=artifact.group_freq)
    series_obj = build_hashtag_series(
        counts,
        group_freq=artifact.group_freq,
        hashtag=artifact.hashtag,
    )
    series = series_obj.series

    values = series.values.astype(np.float32)
    scaled = minmax_scale(values, vmin=artifact.vmin, vmax=artifact.vmax)

    lookback = artifact.lookback
    X, y = make_sequences(scaled, lookback)

    # Time-based split
    train_size = int(len(scaled) * train_ratio)
    train_end = max(1, train_size - lookback)

    X_test, y_test = X[train_end:], y[train_end:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_pred_scaled = _predict_scaled(model, X_test, device=device)

    test_pred = minmax_inverse(test_pred_scaled, vmin=artifact.vmin, vmax=artifact.vmax)
    test_actual = minmax_inverse(y_test, vmin=artifact.vmin, vmax=artifact.vmax)

    # Align timestamps: test targets start at series index [train_size]
    test_index = series.index[train_size : train_size + len(test_pred)]

    # Future forecast (recursive)
    last_window = scaled[-lookback:].copy()
    future_scaled: list[float] = []

    window = last_window
    for _ in range(steps):
        pred = float(_predict_scaled(model, window[np.newaxis, :], device=device)[0])
        future_scaled.append(pred)
        window = np.roll(window, shift=-1)
        window[-1] = pred

    future_pred = minmax_inverse(np.array(future_scaled, dtype=np.float32), vmin=artifact.vmin, vmax=artifact.vmax)

    last_period = series.index[-1].to_period(artifact.group_freq)
    future_periods = pd.period_range(last_period + 1, periods=steps, freq=artifact.group_freq)
    future_index = future_periods.to_timestamp(how="start")

    history = [
        {"period_start": ts.to_pydatetime(), "count": int(val)}
        for ts, val in series.items()
    ]

    test_predictions = [
        {
            "period_start": ts.to_pydatetime(),
            "actual_count": int(round(float(a))),
            "predicted_count": float(max(0.0, p)),
        }
        for ts, a, p in zip(test_index, test_actual, test_pred, strict=False)
    ]

    forecast_points = [
        {"period_start": ts.to_pydatetime(), "predicted_count": float(max(0.0, p))}
        for ts, p in zip(future_index, future_pred, strict=False)
    ]

    return {
        "hashtag": f"#{artifact.hashtag}",
        "group_freq": artifact.group_freq,
        "lookback": artifact.lookback,
        "model": "pytorch_lstm",
        "history": history,
        "test_predictions": test_predictions,
        "forecast": forecast_points,
    }

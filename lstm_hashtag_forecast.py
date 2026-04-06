"""LSTM time-series forecasting for hashtag trends (PyTorch).

Goal
----
Predict future hashtag frequency (trend) over time.

What this script does
---------------------
1) Load social media CSV with a `text` column and a `timestamp` column
2) Extract hashtags from text
3) Build a time series of hashtag counts per day/week
4) Prepare *sequence data* (sliding window) for an LSTM:
     X = counts from the previous LOOKBACK periods
     y = count in the next period
5) Train an LSTM model in PyTorch
6) Predict on a test split + forecast FUTURE_STEPS periods ahead
7) Plot actual vs predicted values

Works with this workspace's dataset:
- data/training.1600000.processed.noemoticon.csv (Sentiment140-style)

Tips (important)
----------------
- This dataset is large; start with a smaller NROWS while experimenting.
- LSTMs need enough time points. If you get "not enough data", try:
  - Increase NROWS
  - Use weekly grouping (GROUP_FREQ="W")
  - Reduce LOOKBACK
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ----------------------------
# Configuration
# ----------------------------


@dataclass(frozen=True)
class Config:
    # Data
    csv_path: Path = Path("data") / "training.1600000.processed.noemoticon.csv"

    # Load fewer rows for faster iteration. Set to None to load all rows.
    nrows: int | None = 200_000

    # Time grouping:
    # - "D" for daily
    # - "W" for weekly
    group_freq: str = "D"

    # Which hashtag to forecast.
    # - Set to None to automatically pick the top hashtag from loaded data.
    # - Otherwise set to something like "fb" (without the leading #)
    target_hashtag: str | None = None

    # Sequence length: how many past periods the LSTM sees to predict the next one
    lookback: int = 14

    # Forecast horizon (how many future periods to predict)
    future_steps: int = 14

    # Train/test split ratio (by time; no shuffling)
    train_ratio: float = 0.8

    # LSTM model + training
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 60

    # Reproducibility
    seed: int = 42


# ----------------------------
# Hashtag extraction utilities
# ----------------------------


HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")


def load_sentiment140_csv(cfg: Config) -> pd.DataFrame:
    """Load the Sentiment140-style CSV (no header) into a DataFrame."""

    if not cfg.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {cfg.csv_path.resolve()}")

    names = ["sentiment", "id", "timestamp", "query", "user", "text"]

    df = pd.read_csv(
        cfg.csv_path,
        header=None,
        names=names,
        encoding="latin-1",
        nrows=cfg.nrows,
        low_memory=False,
    )

    return df


def parse_timestamp(ts: pd.Series) -> pd.Series:
    """Parse timestamps into pandas datetime.

    Sentiment140 timestamps look like:
      "Mon Apr 06 22:19:45 PDT 2009"

    Pandas doesn't reliably parse timezone abbreviations (PDT, PST, etc.),
    so we remove the timezone token and parse with a known format.
    """

    raw = ts.astype(str).str.strip()

    sent140_like = raw.str.match(
        r"^[A-Za-z]{3} [A-Za-z]{3} \d{1,2} \d{2}:\d{2}:\d{2} [A-Za-z]{2,5} \d{4}$"
    )

    if sent140_like.any():
        cleaned = raw.where(
            ~sent140_like,
            raw.str.replace(r"\s[A-Za-z]{2,5}\s(\d{4})$", r" \1", regex=True),
        )
        return pd.to_datetime(cleaned, format="%a %b %d %H:%M:%S %Y", errors="coerce")

    return pd.to_datetime(raw, errors="coerce")


def extract_hashtags(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hashtags and return a long-form DataFrame with one hashtag per row."""

    work = df[["timestamp", "text"]].copy()
    work["text"] = work["text"].fillna("")

    # Find all hashtags per post (list of strings)
    work["hashtag"] = work["text"].astype(str).str.findall(HASHTAG_RE)

    # Normalize: lowercase hashtags so #AI and #ai are counted together
    work["hashtag"] = work["hashtag"].apply(lambda tags: [t.lower() for t in tags])

    # Explode: one row per hashtag
    work = work.explode("hashtag").dropna(subset=["hashtag"])  # remove posts without hashtags

    return work


def build_hashtag_time_series(cfg: Config, hashtag_rows: pd.DataFrame) -> pd.Series:
    """Build a regular (no missing dates) time series of counts for one hashtag.

    Important details:
    - We group by day/week using pandas Periods (robust for weekly grouping).
    - We reindex across the *full time range* present in the loaded data so missing periods
      become explicit zeros.
    - If cfg.target_hashtag is None, we automatically pick a hashtag that has enough periods
      to train with the chosen lookback.
    """

    df = hashtag_rows.copy()
    df["timestamp_parsed"] = parse_timestamp(df["timestamp"])
    df = df.dropna(subset=["timestamp_parsed"])

    # Create a time bucket using Periods ("D"=daily, "W"=weekly, etc.)
    df["period"] = df["timestamp_parsed"].dt.to_period(cfg.group_freq)

    # Count hashtags per period
    counts = df.groupby(["period", "hashtag"]).size().reset_index(name="count")

    # Full period index present in the loaded data (used for zero-filling)
    full_period_index = pd.period_range(
        counts["period"].min(), counts["period"].max(), freq=cfg.group_freq
    )

    # Choose which hashtag to forecast
    if cfg.target_hashtag is None:
        # Require at least (lookback + 6) periods so there are train/test sequences.
        min_periods = cfg.lookback + 6

        summary = (
            counts.groupby("hashtag")
            .agg(total_count=("count", "sum"), n_periods=("period", "nunique"))
            .sort_values(["total_count", "n_periods"], ascending=[False, False])
        )

        candidates = summary[summary["n_periods"] >= min_periods]
        if candidates.empty:
            raise ValueError(
                "No hashtag has enough time points for the current settings. "
                f"Need at least {min_periods} periods. Try increasing nrows, "
                "reducing lookback, or changing group_freq."
            )

        chosen = candidates.index[0]
        print(f"Auto-selected hashtag with enough history: #{chosen}")
    else:
        chosen = cfg.target_hashtag.lower().lstrip("#")
        print(f"Using target hashtag: #{chosen}")

    # Build a series for the chosen hashtag and reindex across the full time range
    chosen_counts = counts[counts["hashtag"] == chosen].set_index("period")["count"]
    chosen_counts = chosen_counts.reindex(full_period_index, fill_value=0).sort_index()

    # Convert PeriodIndex -> DatetimeIndex for plotting
    series = chosen_counts.copy()
    series.index = series.index.to_timestamp(how="start")
    series.name = chosen
    return series


# ----------------------------
# Sequence dataset utilities
# ----------------------------


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def minmax_scale(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Scale values to [0, 1]. Returns (scaled, min, max)."""

    vmin = float(values.min())
    vmax = float(values.max())

    if vmax - vmin < 1e-9:
        # All values are the same; avoid division by zero.
        scaled = np.zeros_like(values, dtype=np.float32)
        return scaled, vmin, vmax

    scaled = (values - vmin) / (vmax - vmin)
    return scaled.astype(np.float32), vmin, vmax


def minmax_inverse(scaled: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Invert min-max scaling back to original units."""

    if vmax - vmin < 1e-9:
        return np.full_like(scaled, fill_value=vmin, dtype=np.float32)

    return (scaled * (vmax - vmin) + vmin).astype(np.float32)


def make_sequences(series: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences for supervised learning.

    Example with lookback=3:
    series: [x0, x1, x2, x3, x4]
    X:      [[x0,x1,x2], [x1,x2,x3]]
    y:      [x3, x4]

    Shapes:
    - X: (num_samples, lookback)
    - y: (num_samples,)
    """

    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i : i + lookback])
        y.append(series[i + lookback])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class SequenceDataset(Dataset):
    """Torch dataset for (sequence -> next value) regression."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # LSTM expects input shape: (batch, seq_len, features)
        x = self.X[idx].unsqueeze(-1)  # add feature dimension (1)
        y = self.y[idx]
        return x, y


# ----------------------------
# LSTM model
# ----------------------------


class LSTMForecaster(nn.Module):
    """Simple LSTM forecaster that predicts the next value in a series."""

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
        # x shape: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        # Use the last timestep's output
        last = out[:, -1, :]
        pred = self.fc(last)
        return pred.squeeze(-1)  # shape: (batch,)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    """Train the LSTM model with Mean Squared Error loss."""

    model.to(device)
    model.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

        # Print occasionally (keeps output readable)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:>3}/{epochs} | Train MSE: {np.mean(losses):.6f}")


@torch.no_grad()
def predict_model(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Predict y for a batch of sequences."""

    model.to(device)
    model.eval()

    xb = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
    preds = model(xb).cpu().numpy().astype(np.float32)
    return preds


def forecast_future(
    model: nn.Module,
    last_window: np.ndarray,
    *,
    steps: int,
    device: torch.device,
) -> np.ndarray:
    """Iteratively forecast the next `steps` values.

    We predict one step ahead, append that prediction to the window,
    then predict the next, and so on.
    """

    window = last_window.astype(np.float32).copy()
    preds = []

    for _ in range(steps):
        pred = predict_model(model, window[np.newaxis, :], device=device)[0]
        preds.append(pred)
        window = np.roll(window, shift=-1)
        window[-1] = pred

    return np.array(preds, dtype=np.float32)


# ----------------------------
# Plotting
# ----------------------------


def plot_actual_vs_predicted(
    series: pd.Series,
    *,
    lookback: int,
    train_size: int,
    test_preds: np.ndarray,
    future_preds: np.ndarray,
    title: str,
) -> None:
    """Plot actual series, test predictions, and future forecast."""

    plt.figure(figsize=(12, 5))

    # Actual series
    plt.plot(series.index, series.values, label="Actual", linewidth=2)

    # Align test predictions with their corresponding timestamps.
    # If we use the first `train_size` points for training, then test targets start at index `train_size`.
    pred_start = train_size
    pred_index = series.index[pred_start : pred_start + len(test_preds)]
    plt.plot(pred_index, test_preds, label="Predicted (test)", linewidth=2)

    # Future predictions start after the last observed timestamp
    last_time = series.index[-1]
    freq = series.index.freq or pd.infer_freq(series.index) or "D"
    future_index = pd.date_range(last_time, periods=len(future_preds) + 1, freq=freq)[1:]
    plt.plot(future_index, future_preds, label="Forecast (future)", linestyle="--")

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Hashtag count")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) Load social media dataset
    df = load_sentiment140_csv(cfg)

    # 2) Extract hashtags from text
    hashtag_rows = extract_hashtags(df)

    if hashtag_rows.empty:
        print("No hashtags found in the loaded rows. Increase nrows or try another dataset.")
        return

    # 3) Build time-series trend data for one hashtag
    series = build_hashtag_time_series(cfg, hashtag_rows)
    print(f"Time points: {len(series)} (from {series.index.min().date()} to {series.index.max().date()})")

    if len(series) <= cfg.lookback + 5:
        raise ValueError(
            f"Not enough time points ({len(series)}) for lookback={cfg.lookback}. "
            "Try weekly grouping, increase nrows, or reduce lookback."
        )

    # Convert to numpy values
    values = series.values.astype(np.float32)

    # 4) Prepare sequence data
    # Scaling helps LSTMs learn more easily.
    scaled, vmin, vmax = minmax_scale(values)
    X, y = make_sequences(scaled, lookback=cfg.lookback)

    # Time-based split (no shuffling)
    train_size = int(len(scaled) * cfg.train_ratio)

    # Because sequences predict the next point, we split sequences by their time position.
    # Sequence i predicts point (i + lookback), so we want those predicted points to be < train_size.
    train_end = max(1, train_size - cfg.lookback)

    X_train, y_train = X[:train_end], y[:train_end]
    X_test, y_test = X[train_end:], y[train_end:]

    print(f"Train sequences: {len(X_train):,} | Test sequences: {len(X_test):,}")

    train_loader = DataLoader(
        SequenceDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True
    )

    # 5) Build LSTM model
    model = LSTMForecaster(
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )

    # 6) Train model
    train_model(model, train_loader, epochs=cfg.epochs, lr=cfg.learning_rate, device=device)

    # 7) Predict on test set + forecast future
    test_preds_scaled = predict_model(model, X_test, device=device)

    # Invert scaling so the plot is in real hashtag counts
    test_preds = minmax_inverse(test_preds_scaled, vmin, vmax)

    # The last window is the final LOOKBACK values from the FULL series
    last_window_scaled = scaled[-cfg.lookback :]
    future_scaled = forecast_future(
        model, last_window_scaled, steps=cfg.future_steps, device=device
    )
    future_preds = minmax_inverse(future_scaled, vmin, vmax)

    # 8) Plot: actual vs predicted + future forecast
    title = f"Hashtag trend forecast: #{series.name} (group={cfg.group_freq})"
    plot_actual_vs_predicted(
        series,
        lookback=cfg.lookback,
        train_size=train_size,
        test_preds=test_preds,
        future_preds=future_preds,
        title=title,
    )


if __name__ == "__main__":
    main()

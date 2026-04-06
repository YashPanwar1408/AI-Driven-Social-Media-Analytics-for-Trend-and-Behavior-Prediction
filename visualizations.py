"""Visualization helpers for the social analytics project.

Creates 3 plots:
1) Sentiment distribution (bar chart)
2) Hashtag trend over time (line plot)
3) Model comparison (bar chart)

This script is beginner-friendly and works with the Sentiment140-style CSV in this repo:
  data/training.1600000.processed.noemoticon.csv

Run:
  d:/aiml-social-analytics/.venv/Scripts/python.exe visualizations.py

Outputs:
    If Matplotlib is running in a non-interactive backend (like Agg), plots are saved to:
        outputs/plots/
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd

# Seaborn is optional (it just makes plots prettier). If it isn't installed,
# the script will still work using matplotlib only.
try:
    import seaborn as sns  # type: ignore

    _HAS_SEABORN = True
except Exception:
    sns = None
    _HAS_SEABORN = False


def _backend_supports_show() -> bool:
    """Return True if the current Matplotlib backend can display GUI windows."""

    backend = str(plt.get_backend()).lower()
    # Common non-interactive backends.
    return backend not in {"agg", "pdf", "ps", "svg", "cairo", "template"}


PLOTS_DIR = Path("outputs") / "plots"


def _finalize_plot(*, filename: str) -> None:
    """Save the current figure and (if possible) show it."""

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_path}")

    if _backend_supports_show():
        plt.show()
    # Always close to avoid accumulating open figures.
    plt.close()


# ----------------------------
# Configuration
# ----------------------------

CSV_PATH = Path("data") / "training.1600000.processed.noemoticon.csv"

# Load fewer rows for faster experimentation (set to None for full dataset)
NROWS: int | None = 200_000

# Trend settings
# Choose which "trend over time" line plot you want:
# - "sentiment": average sentiment over time (0=negative, 1=positive)
# - "hashtags": top hashtag counts over time
TREND_KIND = "hashtags"  # "sentiment" or "hashtags"

GROUP_FREQ = "D"  # "D"=daily, "W"=weekly
TOP_HASHTAGS = 5  # only used for TREND_KIND="hashtags"
WINDOW_PERIODS = 60  # show only the last N periods in the line plot


# ----------------------------
# Data loading
# ----------------------------


def load_sentiment140(csv_path: Path, *, nrows: int | None) -> pd.DataFrame:
    """Load the Sentiment140-style CSV (no header).

    Columns:
      sentiment, id, timestamp, query, user, text
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    names = ["sentiment", "id", "timestamp", "query", "user", "text"]

    df = pd.read_csv(
        csv_path,
        header=None,
        names=names,
        encoding="latin-1",
        nrows=nrows,
        low_memory=False,
    )

    return df


# ----------------------------
# 1) Sentiment distribution
# ----------------------------


def plot_sentiment_distribution(
    df: pd.DataFrame,
    *,
    sentiment_col: str = "sentiment",
    title: str = "Sentiment distribution",
) -> None:
    """Plot a bar chart showing how many samples belong to each sentiment."""

    # Convert to numeric when possible
    s = pd.to_numeric(df[sentiment_col], errors="coerce")

    # Map common sentiment label encodings to readable names
    label_map = {0: "negative", 2: "neutral", 4: "positive"}
    labels = s.map(label_map).fillna(s.astype("Int64").astype(str))

    counts = labels.value_counts().sort_index()

    # Seaborn style (optional)
    if _HAS_SEABORN:
        sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(title)
    plt.xlabel("Sentiment")
    plt.ylabel("Number of posts")
    plt.tight_layout()
    _finalize_plot(filename="sentiment_distribution.png")


# ----------------------------
# 2) Trend over time (hashtags)
# ----------------------------


HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")


def parse_timestamp(ts: pd.Series) -> pd.Series:
    """Parse timestamps into datetimes.

    Sentiment140 timestamps look like:
      "Mon Apr 06 22:19:45 PDT 2009"

    Pandas doesn't reliably parse timezone abbreviations like PDT, so we remove the timezone
    token before the year and parse using a known format. For other formats, fall back.
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
        parsed = pd.to_datetime(cleaned, format="%a %b %d %H:%M:%S %Y", errors="coerce")
    else:
        parsed = pd.to_datetime(raw, errors="coerce")

    return parsed


def build_hashtag_counts_over_time(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    text_col: str = "text",
    group_freq: str = "D",
) -> pd.DataFrame:
    """Extract hashtags and count them by time bucket.

    Logic:
    - Parse timestamp
    - Extract hashtags with regex (list per row)
    - Normalize by lowercasing
    - Explode to one hashtag per row
    - Group by (period_start, hashtag) and count
    """

    work = df[[timestamp_col, text_col]].copy()
    work[text_col] = work[text_col].fillna("")

    work["timestamp_parsed"] = parse_timestamp(work[timestamp_col])
    work = work.dropna(subset=["timestamp_parsed"])

    # Extract hashtags (list per row)
    work["hashtag"] = work[text_col].astype(str).str.findall(HASHTAG_RE)
    work["hashtag"] = work["hashtag"].apply(lambda tags: [t.lower() for t in tags])

    # One hashtag per row
    work = work.explode("hashtag").dropna(subset=["hashtag"])

    # Time bucketing
    period = work["timestamp_parsed"].dt.to_period(group_freq)
    work["period_start"] = period.dt.start_time

    counts = (
        work.groupby(["period_start", "hashtag"]).size().reset_index(name="count")
    )

    return counts


def plot_sentiment_trend_over_time(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    sentiment_col: str = "sentiment",
    group_freq: str = "D",
    window_periods: int = 60,
    title: str = "Sentiment trend over time",
) -> None:
    """Plot sentiment trend over time as a line graph.

    For Sentiment140-style labels:
      0 -> negative, 4 -> positive

    We map those to 0.0 and 1.0 and plot the mean per time bucket (i.e. positive rate).
    If label 2 exists (neutral), we map it to 0.5.
    """

    ts = parse_timestamp(df[timestamp_col])
    sentiment_raw = pd.to_numeric(df[sentiment_col], errors="coerce")
    sentiment_score = sentiment_raw.map({0: 0.0, 2: 0.5, 4: 1.0})

    work = pd.DataFrame({"timestamp": ts, "score": sentiment_score}).dropna()
    if work.empty:
        print("No timestamp/sentiment data available to plot.")
        return

    period = work["timestamp"].dt.to_period(group_freq)
    work["period_start"] = period.dt.start_time

    series = work.groupby("period_start")["score"].mean().sort_index()
    if len(series) > window_periods:
        series = series.iloc[-window_periods:]

    if _HAS_SEABORN:
        sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5))
    plt.plot(series.index, series.values)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Avg sentiment (0=negative, 1=positive)")
    plt.ylim(0, 1)
    plt.tight_layout()
    _finalize_plot(filename="sentiment_trend.png")


def plot_hashtag_trends(
    counts: pd.DataFrame,
    *,
    top_n: int = 5,
    window_periods: int = 60,
    title: str = "Hashtag trends over time",
) -> None:
    """Plot top hashtag counts over time as a line chart."""

    if counts.empty:
        print("No hashtag counts available to plot.")
        return

    # Pick top hashtags overall
    totals = counts.groupby("hashtag")["count"].sum().sort_values(ascending=False)
    top_tags = totals.head(top_n).index.tolist()

    # Keep only top tags
    subset = counts[counts["hashtag"].isin(top_tags)].copy()

    # Keep only a recent time window
    periods = subset["period_start"].drop_duplicates().sort_values()
    if len(periods) > window_periods:
        keep_from = periods.iloc[-window_periods]
        subset = subset[subset["period_start"] >= keep_from]

    # Pivot for plotting: rows=time, columns=hashtag, values=count
    pivot = subset.pivot_table(
        index="period_start",
        columns="hashtag",
        values="count",
        aggfunc="sum",
        fill_value=0,
    ).sort_index()

    if _HAS_SEABORN:
        sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5))
    for tag in pivot.columns:
        plt.plot(pivot.index, pivot[tag], label=f"#{tag}")

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Number of posts")
    plt.legend()
    plt.tight_layout()
    _finalize_plot(filename="hashtag_trends.png")


# ----------------------------
# 3) Model comparison
# ----------------------------


def plot_model_comparison(
    metrics: dict[str, dict[str, float]],
    *,
    metrics_to_plot: tuple[str, ...] = ("accuracy", "f1"),
    title: str = "Model comparison",
) -> None:
    """Plot a bar chart comparing model metrics.

    `metrics` example:
      {
        "TF-IDF": {"accuracy": 0.76, "f1": 0.77},
        "BERT":   {"accuracy": 0.82, "f1": 0.83},
      }
    """

    rows = []
    for model_name, model_metrics in metrics.items():
        row = {"model": model_name}
        for m in metrics_to_plot:
            row[m] = float(model_metrics.get(m, float("nan")))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")

    if _HAS_SEABORN:
        sns.set_theme(style="whitegrid")

    ax = df[list(metrics_to_plot)].plot(kind="bar", figsize=(8, 4))
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    _finalize_plot(filename="model_comparison.png")


def main() -> None:
    # Load data
    df = load_sentiment140(CSV_PATH, nrows=NROWS)

    # 1) Sentiment distribution
    plot_sentiment_distribution(df)

    # 2) Trend over time
    if TREND_KIND == "sentiment":
        plot_sentiment_trend_over_time(
            df,
            group_freq=GROUP_FREQ,
            window_periods=WINDOW_PERIODS,
            title=f"Sentiment trend ({GROUP_FREQ})",
        )
    elif TREND_KIND == "hashtags":
        counts = build_hashtag_counts_over_time(df, group_freq=GROUP_FREQ)
        plot_hashtag_trends(
            counts,
            top_n=TOP_HASHTAGS,
            window_periods=WINDOW_PERIODS,
            title=f"Top {TOP_HASHTAGS} hashtag trends ({GROUP_FREQ})",
        )
    else:
        raise ValueError(f"Unknown TREND_KIND={TREND_KIND!r}; use 'sentiment' or 'hashtags'")

    # 3) Model comparison
    # Replace these placeholder numbers with your real results.
    # Tip: copy the metrics printed by baseline_sentiment_model.py / bert_sentiment_model.py.
    example_metrics = {
        "TF-IDF": {"accuracy": 0.76, "f1": 0.77},
        "BERT": {"accuracy": 0.78, "f1": 0.79},
    }
    plot_model_comparison(example_metrics, metrics_to_plot=("accuracy", "f1"))


if __name__ == "__main__":
    main()

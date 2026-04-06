"""Detect hashtag trends in social media data.

This script shows a simple, beginner-friendly approach to trending topics:
1) Extract hashtags from the `text` column
2) Count hashtag frequency
3) Group counts by time (daily or weekly)
4) Identify top trending hashtags
5) Visualize hashtag trends with line plots

It is compatible with the Sentiment140-style CSV in this workspace:
  data/training.1600000.processed.noemoticon.csv

That file has *no header* and 6 columns:
  sentiment, id, timestamp, query, user, text

If your CSV already has headers (e.g., text, sentiment, timestamp), set HAS_HEADER=True
and update the column names below.
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


# --- Configuration (edit these if your file/columns differ) ---
CSV_PATH = Path("data") / "training.1600000.processed.noemoticon.csv"

# Load fewer rows for faster iteration while experimenting.
# Set to None to load the full file.
NROWS: int | None = 200_000

# This workspace's Sentiment140 file has NO header row.
HAS_HEADER = False

# Column names
COL_TIMESTAMP = "timestamp"
COL_TEXT = "text"

# Hashtag extraction settings
TOP_N = 10  # how many top hashtags to report/plot

# Choose how to group time:
# - "D" for daily
# - "W" for weekly
GROUP_FREQ = "D"


# --- Hashtag regex ---
# Matches hashtags like: #AI, #MachineLearning, #covid19
# We capture only the word part (no leading '#')
HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the CSV into a DataFrame."""

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    if HAS_HEADER:
        return pd.read_csv(csv_path, nrows=NROWS)

    # Sentiment140-style file: no header and latin-1 encoding
    names = ["sentiment", "id", COL_TIMESTAMP, "query", "user", COL_TEXT]
    df = pd.read_csv(
        csv_path,
        header=None,
        names=names,
        encoding="latin-1",
        nrows=NROWS,
        low_memory=False,
    )

    return df


def parse_timestamp(df: pd.DataFrame, timestamp_col: str) -> pd.Series:
    """Parse timestamps into pandas datetime.

    For Sentiment140, timestamps look like:
      "Mon Apr 06 22:19:45 PDT 2009"

    Pandas may not understand timezone abbreviations like PDT, so we remove them
    and parse with a known format. For other datasets, we fall back to generic parsing.
    """

    raw = df[timestamp_col].astype(str).str.strip()

    # Detect Sentiment140-style strings
    sent140_like = raw.str.match(
        r"^[A-Za-z]{3} [A-Za-z]{3} \d{1,2} \d{2}:\d{2}:\d{2} [A-Za-z]{2,5} \d{4}$"
    )

    if sent140_like.any():
        # Remove the timezone token before the year ("... PDT 2009" -> "... 2009")
        cleaned = raw.where(
            ~sent140_like,
            raw.str.replace(r"\s[A-Za-z]{2,5}\s(\d{4})$", r" \1", regex=True),
        )
        return pd.to_datetime(cleaned, format="%a %b %d %H:%M:%S %Y", errors="coerce")

    # Fallback for ISO timestamps, etc.
    return pd.to_datetime(raw, errors="coerce")


def extract_hashtags(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Extract hashtags from the text column.

    Logic:
    - Use regex to find all hashtag matches in each post
    - Convert to lowercase for consistent counting (#AI and #ai become the same)
    - Explode lists of hashtags into one hashtag per row for easy aggregation
    """

    df = df.copy()

    # Fill missing values so string operations are safe
    df[text_col] = df[text_col].fillna("")

    # Find all hashtags in each row (returns a list per row)
    df["hashtags"] = df[text_col].astype(str).str.findall(HASHTAG_RE)

    # Convert each hashtag to lowercase
    df["hashtags"] = df["hashtags"].apply(lambda tags: [t.lower() for t in tags])

    # One hashtag per row
    exploded = df.explode("hashtags").rename(columns={"hashtags": "hashtag"})

    # Drop rows where there was no hashtag
    exploded = exploded.dropna(subset=["hashtag"])

    return exploded


def count_hashtags_over_time(
    hashtag_df: pd.DataFrame,
    *,
    timestamp_col: str,
    group_freq: str,
) -> pd.DataFrame:
    """Count hashtags grouped by a time frequency.

    `group_freq` examples:
    - "D" = daily
    - "W" = weekly
    """

    df = hashtag_df.copy()

    # Convert timestamps to datetimes
    df["timestamp_parsed"] = parse_timestamp(df, timestamp_col)

    # Remove rows with invalid timestamps
    df = df.dropna(subset=["timestamp_parsed"])

    # Create a period start time so we can group by day/week
    # Example: all posts on the same day will have the same period_start.
    period = df["timestamp_parsed"].dt.to_period(group_freq)
    df["period_start"] = period.dt.start_time

    counts = (
        df.groupby(["period_start", "hashtag"])  # group by time + hashtag
        .size()  # count rows
        .reset_index(name="count")
        .sort_values(["period_start", "count"], ascending=[True, False])
    )

    return counts


def get_top_hashtags(counts: pd.DataFrame, top_n: int) -> list[str]:
    """Return the top N hashtags overall (by total count)."""

    total = counts.groupby("hashtag")["count"].sum().sort_values(ascending=False)
    return total.head(top_n).index.tolist()


def print_top_trending_latest_period(counts: pd.DataFrame, top_n: int) -> None:
    """Print top hashtags in the most recent time bucket."""

    if counts.empty:
        print("No hashtag counts to analyze.")
        return

    latest_period = counts["period_start"].max()
    latest = counts[counts["period_start"] == latest_period].sort_values(
        "count", ascending=False
    )

    print(f"\n--- Top {top_n} hashtags in latest period ({latest_period.date()}) ---")
    print(latest.head(top_n).to_string(index=False))


def plot_trends(counts: pd.DataFrame, hashtags_to_plot: list[str]) -> None:
    """Plot hashtag frequency trends over time using line plots."""

    if not hashtags_to_plot:
        print("No hashtags to plot.")
        return

    # Keep only selected hashtags
    subset = counts[counts["hashtag"].isin(hashtags_to_plot)].copy()

    # Convert long-form counts to wide format:
    # rows = period_start, columns = hashtag, values = count
    pivot = subset.pivot_table(
        index="period_start",
        columns="hashtag",
        values="count",
        aggfunc="sum",
        fill_value=0,
    ).sort_index()

    plt.figure(figsize=(10, 5))

    for hashtag in pivot.columns:
        plt.plot(pivot.index, pivot[hashtag], label=f"#{hashtag}")

    plt.title("Hashtag trends over time")
    plt.xlabel("Time")
    plt.ylabel("Number of posts")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    # 1) Load dataset
    df = load_dataset(CSV_PATH)
    print("Loaded rows:", len(df))

    # 2) Extract hashtags from text
    hashtag_df = extract_hashtags(df, text_col=COL_TEXT)

    if hashtag_df.empty:
        print("No hashtags found in this dataset (at least in the rows loaded).")
        return

    # 3) Count hashtag frequency grouped by time (daily/weekly)
    counts = count_hashtags_over_time(
        hashtag_df,
        timestamp_col=COL_TIMESTAMP,
        group_freq=GROUP_FREQ,
    )

    # 4) Identify top trending topics
    top_hashtags = get_top_hashtags(counts, top_n=TOP_N)

    print(f"\n--- Top {TOP_N} hashtags overall ---")
    for i, tag in enumerate(top_hashtags, start=1):
        print(f"{i:>2}. #{tag}")

    print_top_trending_latest_period(counts, top_n=TOP_N)

    # 5) Visualize trends
    plot_trends(counts, hashtags_to_plot=top_hashtags)


if __name__ == "__main__":
    main()

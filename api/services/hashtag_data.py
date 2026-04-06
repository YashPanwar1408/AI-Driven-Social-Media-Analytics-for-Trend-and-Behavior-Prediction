from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


# Hashtag regex: captures the token without the leading '#'
HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")


def load_timestamp_text_csv_sentiment140(csv_path: Path, *, nrows: int | None) -> pd.DataFrame:
    """Load only timestamp and text columns from the Sentiment140-style CSV."""

    df = pd.read_csv(
        csv_path,
        header=None,
        usecols=[2, 5],
        encoding="latin-1",
        nrows=nrows,
        low_memory=False,
    )
    df.columns = ["timestamp", "text"]
    df["text"] = df["text"].fillna("")
    return df


def parse_timestamp(ts: pd.Series) -> pd.Series:
    """Parse timestamps into pandas datetime.

    Sentiment140 strings look like:
      "Mon Apr 06 22:19:45 PDT 2009"

    Pandas doesn't reliably parse timezone abbreviations like PDT, so we remove
    the timezone token and parse with a known format.

    For non-Sentiment140 data, we fall back to generic parsing.
    """

    raw = ts.astype(str).str.strip()

    sent140_like = raw.str.match(
        r"^[A-Za-z]{3} [A-Za-z]{3} \d{1,2} \d{2}:\d{2}:\d{2} [A-Za-z]{2,5} \d{4}$"
    )

    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    if sent140_like.any():
        cleaned = raw.where(
            ~sent140_like,
            raw.str.replace(r"\s[A-Za-z]{2,5}\s(\d{4})$", r" \1", regex=True),
        )
        parsed = pd.to_datetime(cleaned, format="%a %b %d %H:%M:%S %Y", errors="coerce")

    # Fallback for anything that didn't parse
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(raw[missing], errors="coerce")

    return parsed


def extract_hashtag_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return a long-form DataFrame with one hashtag per row.

    Output columns:
    - timestamp_parsed (datetime)
    - hashtag (lowercase string)
    """

    work = df[["timestamp", "text"]].copy()
    work["timestamp_parsed"] = parse_timestamp(work["timestamp"])
    work = work.dropna(subset=["timestamp_parsed"])

    # Extract hashtags: list per row
    work["hashtag"] = work["text"].astype(str).str.findall(HASHTAG_RE)
    work["hashtag"] = work["hashtag"].apply(lambda tags: [t.lower() for t in tags])

    # Explode: one row per hashtag
    work = work.explode("hashtag").dropna(subset=["hashtag"])

    return work[["timestamp_parsed", "hashtag"]]


def compute_hashtag_counts(
    hashtag_rows: pd.DataFrame, *, group_freq: str
) -> pd.DataFrame:
    """Compute hashtag counts grouped by time period.

    Returns a DataFrame with columns:
    - period_start (datetime)
    - hashtag (string)
    - count (int)
    """

    df = hashtag_rows.copy()
    df["period"] = df["timestamp_parsed"].dt.to_period(group_freq)
    df["period_start"] = df["period"].dt.start_time

    counts = (
        df.groupby(["period_start", "hashtag"]).size().reset_index(name="count")
    )

    return counts


@dataclass(frozen=True)
class HashtagSeries:
    hashtag: str
    group_freq: str
    series: pd.Series  # index=period_start timestamps, values=int counts


def build_hashtag_series(
    counts: pd.DataFrame,
    *,
    group_freq: str,
    hashtag: str,
) -> HashtagSeries:
    """Create a continuous time series for a given hashtag (missing periods filled with 0)."""

    chosen = hashtag.lower().lstrip("#")

    subset = counts[counts["hashtag"] == chosen].copy()
    if subset.empty:
        raise ValueError(f"Hashtag #{chosen} not found")

    start_period = subset["period_start"].min().to_period(group_freq)
    end_period = subset["period_start"].max().to_period(group_freq)

    full_periods = pd.period_range(start_period, end_period, freq=group_freq)
    full_index = full_periods.to_timestamp(how="start")

    series = subset.set_index("period_start")["count"].sort_index()
    series = series.reindex(full_index, fill_value=0).astype(int)
    series.name = chosen

    return HashtagSeries(hashtag=chosen, group_freq=group_freq, series=series)

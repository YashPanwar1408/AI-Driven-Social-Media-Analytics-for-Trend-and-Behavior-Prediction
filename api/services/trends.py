from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import pandas as pd

from .hashtag_data import (
    compute_hashtag_counts,
    extract_hashtag_rows,
    load_timestamp_text_csv_sentiment140,
)

logger = logging.getLogger(__name__)


@dataclass
class TrendsService:
    csv_path: Path
    nrows: int | None

    # Cache: group_freq -> counts dataframe
    _counts_cache: dict[str, pd.DataFrame]

    def __init__(self, *, csv_path: Path, nrows: int | None) -> None:
        self.csv_path = csv_path
        self.nrows = nrows
        self._counts_cache = {}

    def _get_counts(self, group_freq: str) -> pd.DataFrame:
        group_freq = group_freq.upper()
        if group_freq not in {"D", "W"}:
            raise ValueError("group_freq must be 'D' (daily) or 'W' (weekly)")

        if group_freq in self._counts_cache:
            return self._counts_cache[group_freq]

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path.resolve()}")

        logger.info("Computing hashtag counts (freq=%s, nrows=%s)", group_freq, self.nrows)
        raw = load_timestamp_text_csv_sentiment140(self.csv_path, nrows=self.nrows)
        hashtag_rows = extract_hashtag_rows(raw)
        counts = compute_hashtag_counts(hashtag_rows, group_freq=group_freq)

        # Keep a stable ordering
        counts = counts.sort_values(["period_start", "count"], ascending=[True, False])

        self._counts_cache[group_freq] = counts
        return counts

    def get_counts_dataframe(self, group_freq: str) -> pd.DataFrame:
        """Return the aggregated hashtag counts DataFrame for internal reuse.

        This is useful for other services (e.g., forecasting) to avoid re-reading
        and re-parsing the CSV multiple times.
        """

        return self._get_counts(group_freq)

    def get_trends(
        self,
        *,
        top_n: int,
        group_freq: str,
        window: int,
    ) -> dict:
        """Return trending hashtags and a small timeseries window for plotting."""

        counts = self._get_counts(group_freq)
        if counts.empty:
            return {
                "group_freq": group_freq,
                "top_overall": [],
                "latest_period_start": None,
                "top_latest": [],
                "series": {},
            }

        total = counts.groupby("hashtag")["count"].sum().sort_values(ascending=False)
        top_hashtags = total.head(top_n)

        latest_period = counts["period_start"].max()
        latest = counts[counts["period_start"] == latest_period].sort_values(
            "count", ascending=False
        )

        # Limit the timeseries window to the last `window` periods
        periods = counts["period_start"].drop_duplicates().sort_values()
        if window < len(periods):
            keep_from = periods.iloc[-window]
            counts_window = counts[counts["period_start"] >= keep_from]
        else:
            counts_window = counts

        series: dict[str, list[dict]] = {}
        for tag in top_hashtags.index.tolist():
            subset = counts_window[counts_window["hashtag"] == tag][["period_start", "count"]]
            series[f"#{tag}"] = [
                {"period_start": row.period_start.to_pydatetime(), "count": int(row.count)}
                for row in subset.itertuples(index=False)
            ]

        return {
            "group_freq": group_freq,
            "top_overall": [
                {"hashtag": f"#{tag}", "count": int(cnt)}
                for tag, cnt in top_hashtags.items()
            ],
            "latest_period_start": latest_period.to_pydatetime(),
            "top_latest": [
                {"hashtag": f"#{row.hashtag}", "count": int(row.count)}
                for row in latest.head(top_n).itertuples(index=False)
            ],
            "series": series,
        }

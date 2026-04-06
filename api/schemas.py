from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# -----------------
# /sentiment
# -----------------


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)


class SentimentResponse(BaseModel):
    label: str
    score: float = Field(..., ge=0.0, le=1.0)
    model: str


# -----------------
# /trends
# -----------------


class HashtagCount(BaseModel):
    hashtag: str
    count: int = Field(..., ge=0)


class TrendPoint(BaseModel):
    period_start: datetime
    count: int = Field(..., ge=0)


class TrendsResponse(BaseModel):
    group_freq: Literal["D", "W"]
    top_overall: list[HashtagCount]
    latest_period_start: datetime | None
    top_latest: list[HashtagCount]
    series: dict[str, list[TrendPoint]]


# -----------------
# /predict
# -----------------


class PredictRequest(BaseModel):
    hashtag: str | None = Field(
        default=None,
        description="Hashtag to forecast, without the leading '#'. If omitted, uses the trained model's hashtag.",
    )
    steps: int = Field(default=14, ge=1, le=60)


class TestPredictionPoint(BaseModel):
    period_start: datetime
    actual_count: int = Field(..., ge=0)
    predicted_count: float = Field(..., ge=0.0)


class ForecastPoint(BaseModel):
    period_start: datetime
    predicted_count: float = Field(..., ge=0.0)


class PredictResponse(BaseModel):
    hashtag: str
    group_freq: Literal["D", "W"]
    lookback: int = Field(..., ge=1)
    model: str

    history: list[TrendPoint]
    test_predictions: list[TestPredictionPoint]
    forecast: list[ForecastPoint]

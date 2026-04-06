from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request

from .config import get_settings
from .schemas import (
    PredictRequest,
    PredictResponse,
    SentimentRequest,
    SentimentResponse,
    TrendsResponse,
)
from .services.forecast import forecast, load_or_train_lstm
from .services.sentiment import load_or_train_sentiment_model
from .services.trends import TrendsService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Basic logging setup (works for local dev and simple deployments)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    settings = get_settings()
    app.state.settings = settings

    # Always initialize trends service (it doesn't require trained artifacts).
    app.state.trends_service = TrendsService(csv_path=settings.csv_path, nrows=settings.trends_nrows)

    # Sentiment model (TF-IDF)
    app.state.sentiment_model = None
    app.state.sentiment_error = None
    try:
        app.state.sentiment_model = load_or_train_sentiment_model(
            csv_path=settings.csv_path,
            artifact_path=settings.sentiment_artifact,
            auto_train=settings.auto_train_artifacts,
            sample_size=settings.sentiment_sample_size,
            positive_start_row=settings.sentiment140_positive_start_row,
        )
    except Exception as exc:  # keep app runnable for /trends
        app.state.sentiment_error = str(exc)
        logger.exception("Failed to initialize sentiment model")

    # LSTM forecaster
    app.state.lstm_model = None
    app.state.lstm_artifact = None
    app.state.lstm_error = None
    try:
        model, artifact = load_or_train_lstm(
            csv_path=settings.csv_path,
            nrows=settings.lstm_nrows,
            artifact_path=settings.lstm_artifact,
            auto_train=settings.auto_train_artifacts,
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
        app.state.lstm_model = model
        app.state.lstm_artifact = artifact
    except Exception as exc:  # keep app runnable for /trends and /sentiment
        app.state.lstm_error = str(exc)
        logger.exception("Failed to initialize LSTM forecaster")

    yield


app = FastAPI(title="Social Analytics API", version="1.0.0", lifespan=lifespan)

# Safe defaults in case lifespan is disabled or not yet executed (e.g., some test setups).
app.state.settings = None
app.state.trends_service = None
app.state.sentiment_model = None
app.state.sentiment_error = "Sentiment model not initialized"
app.state.lstm_model = None
app.state.lstm_artifact = None
app.state.lstm_error = "Forecast model not initialized"


@app.post("/sentiment", response_model=SentimentResponse)
def sentiment(req: SentimentRequest, request: Request) -> SentimentResponse:
    state = request.app.state
    model = getattr(state, "sentiment_model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=getattr(state, "sentiment_error", None)
            or "Sentiment model not loaded. Run scripts/train_artifacts.py.",
        )

    try:
        label, score = model.predict(req.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SentimentResponse(label=label, score=score, model="tfidf_logreg")


@app.get("/trends", response_model=TrendsResponse)
def trends(
    request: Request,
    top_n: int = Query(10, ge=1, le=50),
    group_freq: Literal["D", "W"] = Query("D"),
    window: int = Query(60, ge=1, le=365),
) -> TrendsResponse:
    state = request.app.state
    service: TrendsService | None = getattr(state, "trends_service", None)
    if service is None:
        # Fallback: create a service lazily (useful in certain test setups)
        settings = get_settings()
        service = TrendsService(csv_path=settings.csv_path, nrows=settings.trends_nrows)

    try:
        result = service.get_trends(top_n=top_n, group_freq=group_freq, window=window)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return TrendsResponse(**result)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    state = request.app.state
    model = getattr(state, "lstm_model", None)
    artifact = getattr(state, "lstm_artifact", None)
    if model is None or artifact is None:
        raise HTTPException(
            status_code=503,
            detail=getattr(state, "lstm_error", None)
            or "Forecast model not loaded. Run scripts/train_artifacts.py.",
        )

    # This backend forecasts the specific hashtag the LSTM was trained on.
    if req.hashtag is not None:
        requested = req.hashtag.lower().lstrip("#")
        if requested != artifact.hashtag:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"This server's LSTM model was trained for '#{artifact.hashtag}'. "
                    f"Requested '#{requested}'. Train another model artifact to support it."
                ),
            )

    # Reuse the cached counts from TrendsService if available.
    cached_counts = None
    trends_service = getattr(state, "trends_service", None)
    if trends_service is not None:
        try:
            cached_counts = trends_service.get_counts_dataframe(artifact.group_freq)
        except Exception:
            cached_counts = None

    result = forecast(
        model=model,
        artifact=artifact,
        csv_path=state.settings.csv_path if getattr(state, "settings", None) else get_settings().csv_path,
        nrows=state.settings.lstm_nrows if getattr(state, "settings", None) else get_settings().lstm_nrows,
        steps=req.steps,
        counts=cached_counts,
    )

    return PredictResponse(**result)

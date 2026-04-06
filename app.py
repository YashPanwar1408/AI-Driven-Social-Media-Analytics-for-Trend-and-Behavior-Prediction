"""Streamlit frontend for the Social Media Analytics project.

✅ Requirements satisfied
------------------------
- Uses existing artifacts only (NO retraining):
  - artifacts/sentiment_tfidf.joblib
  - artifacts/bert_sentiment/
  - artifacts/hashtag_lstm.pt
- Uses existing outputs (NO recomputation):
  - outputs/trends/
  - outputs/forecast/forecast.json
  - outputs/plots/*.png
- Sidebar navigation:
  - Sentiment Analysis
  - Trends Dashboard
  - Forecast Prediction
  - Project Overview

Run
---
  streamlit run app.py
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Paths (all relative to repo root)
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
OUTPUTS_DIR = REPO_ROOT / "outputs"

SENTIMENT_TFIDF_PATH = ARTIFACTS_DIR / "sentiment_tfidf.joblib"
BERT_DIR = ARTIFACTS_DIR / "bert_sentiment"
LSTM_PATH = ARTIFACTS_DIR / "hashtag_lstm.pt"

TRENDS_JSON_PATH = OUTPUTS_DIR / "trends" / "trends_D.json"
FORECAST_JSON_PATH = OUTPUTS_DIR / "forecast" / "forecast.json"

PLOT_SENTIMENT_DIST = OUTPUTS_DIR / "plots" / "sentiment_distribution.png"
PLOT_TRENDS = OUTPUTS_DIR / "plots" / "hashtag_trends.png"
PLOT_MODEL_COMPARISON = OUTPUTS_DIR / "plots" / "model_comparison.png"


# -----------------------------------------------------------------------------
# Streamlit page setup
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Social Media Analytics",
    page_icon="📊",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _file_status(path: Path) -> tuple[bool, str]:
    exists = path.exists()
    return exists, ("✅ Found" if exists else "❌ Missing")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_trends_json() -> dict:
    return _read_json(TRENDS_JSON_PATH)


@st.cache_data(show_spinner=False)
def load_forecast_json() -> dict:
    return _read_json(FORECAST_JSON_PATH)


@st.cache_resource(show_spinner=False)
def load_tfidf_model():
    """Load TF-IDF sentiment artifact (no retraining)."""

    from api.services.sentiment import load_sentiment_artifact

    return load_sentiment_artifact(SENTIMENT_TFIDF_PATH)


@st.cache_resource(show_spinner=False)
def load_bert_model_and_tokenizer():
    """Load BERT sentiment artifact from artifacts/bert_sentiment (no retraining)."""

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR)
    model.eval()

    meta_path = BERT_DIR / "meta.json"
    label_names = ["negative", "positive"]
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            label_names = list(meta.get("label_names", label_names))
        except Exception:
            label_names = ["negative", "positive"]

    return model, tokenizer, label_names


@st.cache_resource(show_spinner=False)
def load_lstm_artifact():
    """Load the LSTM artifact (weights + metadata) from artifacts/hashtag_lstm.pt."""

    from api.services.forecast import load_lstm_artifact as _load

    return _load(LSTM_PATH)


def predict_with_bert(text: str) -> tuple[str, float]:
    """Run BERT inference and return (label, confidence)."""

    import torch

    model, tokenizer, label_names = load_bert_model_and_tokenizer()

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        out = model(**inputs)
        probs = torch.softmax(out.logits, dim=-1).squeeze(0)
        pred_id = int(torch.argmax(probs).item())
        confidence = float(probs[pred_id].item())

    label = label_names[pred_id] if 0 <= pred_id < len(label_names) else str(pred_id)
    return label, confidence


# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "😊 Sentiment Analysis",
        "🔥 Trends Dashboard",
        "📈 Forecast Prediction",
        "🧾 Project Overview",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("📦 File Status")

for label, path in [
    ("TF‑IDF artifact", SENTIMENT_TFIDF_PATH),
    ("BERT artifact dir", BERT_DIR),
    ("LSTM artifact", LSTM_PATH),
    ("Trends JSON", TRENDS_JSON_PATH),
    ("Forecast JSON", FORECAST_JSON_PATH),
    ("Trends plot", PLOT_TRENDS),
    ("Model comparison plot", PLOT_MODEL_COMPARISON),
]:
    ok, status = _file_status(path)
    st.sidebar.write(f"- {status}: {label}")

st.sidebar.markdown("---")
st.sidebar.caption("Run pipeline to generate missing files:\n`python run_pipeline.py --fast` or `--full`")


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------

if page.startswith("😊"):
    st.title("😊 Sentiment Analysis")
    st.caption("Predict sentiment using saved artifacts (no retraining).")

    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        st.subheader("📝 Input")
        text = st.text_area(
            "Enter a social media post",
            height=140,
            placeholder="Example: I love this new phone! Battery life is amazing.",
        )

        model_choice = st.radio(
            "Choose model",
            ["TF‑IDF (fast)", "BERT (deep learning)"],
            horizontal=True,
        )

        predict_clicked = st.button("🔍 Predict", type="primary")

    with col_right:
        st.subheader("✅ Output")

        if predict_clicked:
            if not text.strip():
                st.warning("Please enter some text.")
            else:
                try:
                    with st.spinner("Loading model and predicting..."):
                        if model_choice.startswith("TF"):
                            if not SENTIMENT_TFIDF_PATH.exists():
                                raise FileNotFoundError(
                                    "TF-IDF artifact is missing. Run: python run_pipeline.py --fast"
                                )
                            model = load_tfidf_model()
                            label, score = model.predict(text)
                        else:
                            if not BERT_DIR.exists():
                                raise FileNotFoundError(
                                    "BERT artifact folder is missing. Run: python run_pipeline.py --full"
                                )
                            label, score = predict_with_bert(text)

                    pretty = "Positive" if label.lower().startswith("pos") else "Negative"
                    st.metric("Sentiment", pretty)
                    st.metric("Confidence", f"{score * 100:.2f}%")
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

    st.markdown("---")
    st.subheader("📊 Model Comparison")
    if PLOT_MODEL_COMPARISON.exists():
        st.image(str(PLOT_MODEL_COMPARISON), caption="Model comparison (Accuracy & F1)", use_container_width=True)
    else:
        st.info("Model comparison plot not found. Run: python run_pipeline.py --fast")


elif page.startswith("🔥"):
    st.title("🔥 Trends Dashboard")
    st.caption("Shows trends from saved outputs in outputs/trends/ and outputs/plots/.")

    if not TRENDS_JSON_PATH.exists():
        st.error("Missing trends file outputs/trends/trends_D.json. Run: python run_pipeline.py --fast")
    else:
        with st.spinner("Loading trend outputs..."):
            trends = load_trends_json()

        top_overall = trends.get("top_overall", [])
        top_latest = trends.get("top_latest", [])
        latest_period = trends.get("latest_period_start")

        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.subheader("🏆 Top Hashtags (Overall)")
            if top_overall:
                df = pd.DataFrame(top_overall)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.write("No hashtags found in this window.")

        with c2:
            st.subheader("🕒 Top Hashtags (Latest Period)")
            st.write(f"Latest period start: **{latest_period}**")
            if top_latest:
                df2 = pd.DataFrame(top_latest)
                st.dataframe(df2, use_container_width=True, hide_index=True)
            else:
                st.write("No hashtags found in the latest period.")

        st.markdown("---")
        st.subheader("📉 Trend Graph")
        if PLOT_TRENDS.exists():
            st.image(str(PLOT_TRENDS), caption="Hashtag trends (from outputs/plots)", use_container_width=True)
        else:
            st.info("Trend plot not found. Run: python run_pipeline.py --fast")


elif page.startswith("📈"):
    st.title("📈 Forecast Prediction")
    st.caption("Loads predictions from outputs/forecast/forecast.json (no retraining).")

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("🧠 LSTM Artifact")
        if not LSTM_PATH.exists():
            st.error("Missing artifacts/hashtag_lstm.pt. Run: python run_pipeline.py --full")
        else:
            with st.spinner("Loading LSTM artifact..."):
                model, artifact = load_lstm_artifact()

            st.metric("Trained Hashtag", f"#{artifact.hashtag}")
            st.metric("Frequency", artifact.group_freq)
            st.metric("Lookback", str(artifact.lookback))
            st.caption("(Model is loaded from artifacts/hashtag_lstm.pt; no training in UI)")

            with st.expander("Show artifact metadata"):
                st.json(asdict(artifact))

    with right:
        st.subheader("📄 Forecast Output")
        if not FORECAST_JSON_PATH.exists():
            st.error("Missing outputs/forecast/forecast.json. Run: python run_pipeline.py --full")
        else:
            with st.spinner("Loading forecast output..."):
                forecast = load_forecast_json()

            forecast_points = forecast.get("forecast", [])
            test_points = forecast.get("test_predictions", [])

            if forecast_points:
                df_forecast = pd.DataFrame(forecast_points)
                st.dataframe(df_forecast, use_container_width=True, hide_index=True)

                st.subheader("📉 Forecast Graph")
                # If an external graph image exists, display it; otherwise plot from JSON.
                forecast_plot_path = OUTPUTS_DIR / "plots" / "forecast.png"
                if forecast_plot_path.exists():
                    st.image(str(forecast_plot_path), caption="Forecast graph", use_container_width=True)
                else:
                    # Plot from JSON (still satisfies "show graph" requirement).
                    chart_df = df_forecast.copy()
                    chart_df["period_start"] = pd.to_datetime(chart_df["period_start"], errors="coerce")
                    chart_df = chart_df.dropna(subset=["period_start"]).set_index("period_start")
                    st.line_chart(chart_df["predicted_count"], height=320)

            else:
                st.write("No forecast points found in forecast.json")

            with st.expander("Show test predictions (actual vs predicted)"):
                if test_points:
                    df_test = pd.DataFrame(test_points)
                    st.dataframe(df_test, use_container_width=True, hide_index=True)
                else:
                    st.write("No test prediction points available.")


else:
    st.title("🧾 Project Overview")
    st.caption("High-level explanation of models and architecture.")

    c1, c2 = st.columns([2, 1], gap="large")

    with c1:
        st.subheader("🧠 TF‑IDF vs BERT")
        st.markdown(
            """
- **TF‑IDF + Logistic Regression** is a strong baseline for sentiment analysis. It is fast and works well for many text classification tasks.
- **BERT** is a Transformer model that captures context and word order, but it is heavier to run and usually needs more compute (GPU) and tuning to outperform a strong baseline.
            """
        )

        st.subheader("📈 LSTM Forecasting")
        st.markdown(
            """
- Hashtags are aggregated into a **time series** (daily/weekly counts).
- An **LSTM** learns from a sliding window of past counts (lookback) to predict future values.
- Forecasts help estimate how a trend may behave in upcoming days.
            """
        )

        st.subheader("🏗️ Architecture")
        st.markdown(
            """
- **Offline pipeline**: `run_pipeline.py` generates artifacts and outputs.
- **Artifacts**: stored in `artifacts/` and reused (no retraining).
- **FastAPI**: serves `/sentiment`, `/trends`, `/predict` using saved artifacts.
- **Streamlit UI**: this app reads artifacts + outputs and provides an interactive dashboard.
            """
        )

    with c2:
        st.subheader("📊 Key Plots")
        if PLOT_SENTIMENT_DIST.exists():
            st.image(str(PLOT_SENTIMENT_DIST), caption="Sentiment distribution", use_container_width=True)
        if PLOT_MODEL_COMPARISON.exists():
            st.image(str(PLOT_MODEL_COMPARISON), caption="Model comparison", use_container_width=True)

    st.markdown("---")
    st.subheader("📌 How to Run")
    st.code(
        """# 1) Run the pipeline
python run_pipeline.py --fast

# 2) Start Streamlit
streamlit run app.py

# (Optional) Start the API
python -m uvicorn api.main:app --reload
""",
        language="bash",
    )

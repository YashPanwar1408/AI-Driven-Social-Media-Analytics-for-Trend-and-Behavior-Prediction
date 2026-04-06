"""Smoke-test script for the FastAPI backend.

What it does
------------
- Calls all endpoints once:
  - POST /sentiment
  - GET  /trends
  - POST /predict
- Prints status codes and JSON responses.

How to run
----------
  d:/aiml-social-analytics/.venv/Scripts/python.exe test_api.py

Notes
-----
- This uses FastAPI's TestClient, so you do NOT need to start a server.
- The API is configured to load models from ./artifacts by default.
  If artifacts are missing, endpoints may return 503.
  Create artifacts via:
    d:/aiml-social-analytics/.venv/Scripts/python.exe scripts/train_artifacts.py
  or run the pipeline:
    d:/aiml-social-analytics/.venv/Scripts/python.exe run_pipeline.py --fast
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from api.main import app


def _print_response(name: str, r) -> None:
    print("\n" + "=" * 80)
    print(name)
    print("Status:", r.status_code)
    try:
        payload = r.json()
        print(json.dumps(payload, indent=2, default=str)[:4000])
    except Exception:
        print(r.text[:4000])


def main() -> None:
    with TestClient(app) as client:
        # 1) Sentiment
        r1 = client.post(
            "/sentiment",
            json={
                "text": "I love this project! The results are awesome.",
            },
        )
        _print_response("POST /sentiment", r1)

        # 2) Trends
        r2 = client.get(
            "/trends",
            params={
                "top_n": 5,
                "group_freq": "D",
                "window": 30,
            },
        )
        _print_response("GET /trends", r2)

        # 3) Forecast
        r3 = client.post(
            "/predict",
            json={
                "steps": 5,
            },
        )
        _print_response("POST /predict", r3)


if __name__ == "__main__":
    main()

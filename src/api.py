import logging
import os
import secrets
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import APIKeyHeader

from src.predict import load_model_bundle, predict_features
from src.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)


logger = logging.getLogger("uvicorn.error")

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.getenv("CLV_API_KEY", "")

    if len(api_key) < 16:
        raise RuntimeError(
            "Set CLV_API_KEY to a value of at least 16 characters "
            "before starting the API."
        )

    app.state.api_key = api_key
    app.state.bundle = load_model_bundle()

    logger.info(
        "CLV model loaded: version %s",
        app.state.bundle["metadata"]["model_version"],
    )

    yield

    app.state.bundle = None


app = FastAPI(
    title="Customer Lifetime Value Prediction API",
    description=(
        "Predict customer purchase value over the next 90 days "
        "from historical customer features."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def require_api_key(
    request: Request,
    supplied_key: str | None = Depends(api_key_header),
) -> None:
    expected_key = request.app.state.api_key

    if supplied_key is None or not secrets.compare_digest(
        supplied_key.encode("utf-8"),
        expected_key.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key.",
        )


def run_predictions(
    customers: list[PredictionRequest],
    request: Request,
) -> list[PredictionResponse]:
    bundle = request.app.state.bundle

    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready.",
        )

    frame = pd.DataFrame(
        [customer.features.model_dump() for customer in customers]
    )

    try:
        predictions = predict_features(bundle, frame)
    except Exception:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from None

    metadata = bundle["metadata"]

    return [
        PredictionResponse(
            customer_id=customer.customer_id,
            predicted_clv_90d=round(float(prediction), 2),
            currency=metadata["currency"],
            prediction_horizon_days=metadata[
                "prediction_horizon_days"
            ],
            model_version=metadata["model_version"],
        )
        for customer, prediction in zip(
            customers,
            predictions,
            strict=True,
        )
    ]


@app.get("/health")
def health(request: Request):
    bundle = request.app.state.bundle

    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready.",
        )

    return {
        "status": "ready",
        "model_version": bundle["metadata"]["model_version"],
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(require_api_key)],
)
def predict(
    payload: PredictionRequest,
    request: Request,
):
    return run_predictions([payload], request)[0]


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    dependencies=[Depends(require_api_key)],
)
def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
):
    return BatchPredictionResponse(
        predictions=run_predictions(payload.customers, request)
    )
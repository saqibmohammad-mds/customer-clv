import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.predict import load_model_bundle, predict_features


TEST_API_KEY = "test-only-key-for-clv-api"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("CLV_API_KEY", TEST_API_KEY)

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def headers():
    return {"X-API-Key": TEST_API_KEY}


@pytest.fixture
def payload():
    return {
        "customer_id": "C00001",
        "features": {
            "recency_days": 10,
            "tenure_days": 400,
            "orders_180d": 12,
            "spend_180d": 1200.0,
            "avg_order_value_180d": 100.0,
            "active_months_180d": 6,
            "spend_last_30d": 250.0,
            "spend_previous_30d": 200.0,
            "spend_change_30d": 50.0,
        },
    }


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


@pytest.mark.parametrize("key", [None, "incorrect-key"])
def test_rejects_missing_or_wrong_key(client, payload, key):
    request_headers = {} if key is None else {"X-API-Key": key}

    response = client.post(
        "/predict",
        json=payload,
        headers=request_headers,
    )

    assert response.status_code == 401


def test_prediction_matches_saved_model(client, headers, payload):
    response = client.post(
        "/predict",
        json=payload,
        headers=headers,
    )

    assert response.status_code == 200

    bundle = load_model_bundle()
    frame = pd.DataFrame([payload["features"]])
    expected = round(float(predict_features(bundle, frame)[0]), 2)

    result = response.json()

    assert result["customer_id"] == payload["customer_id"]
    assert result["predicted_clv_90d"] == expected
    assert result["currency"] == bundle["metadata"]["currency"]
    assert result["model_version"] == bundle["metadata"]["model_version"]
    assert result["prediction_horizon_days"] == 90


@pytest.mark.parametrize(
    "field,value",
    [
        ("spend_180d", -1),
        ("recency_days", 500),
        ("orders_180d", 1.5),
        ("spend_change_30d", 999),
        ("unexpected_feature", 1),
    ],
)
def test_rejects_invalid_features(
    client,
    headers,
    payload,
    field,
    value,
):
    payload["features"][field] = value

    response = client.post(
        "/predict",
        json=payload,
        headers=headers,
    )

    assert response.status_code == 422


def test_batch_matches_single_predictions(client, headers, payload):
    second = {
        "customer_id": "C00002",
        "features": {
            **payload["features"],
            "recency_days": 20,
        },
    }

    customers = [payload, second]

    response = client.post(
        "/predict/batch",
        json={"customers": customers},
        headers=headers,
    )

    assert response.status_code == 200

    expected = [
        client.post(
            "/predict",
            json=customer,
            headers=headers,
        ).json()
        for customer in customers
    ]

    assert response.json()["predictions"] == expected


@pytest.mark.parametrize("size", [0, 101])
def test_rejects_invalid_batch_size(
    client,
    headers,
    payload,
    size,
):
    response = client.post(
        "/predict/batch",
        json={"customers": [payload] * size},
        headers=headers,
    )

    assert response.status_code == 422
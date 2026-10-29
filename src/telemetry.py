import json
from datetime import datetime, timezone
from uuid import uuid4


def record_predictions(frame, predictions, metadata):
    for features, prediction in zip(
        frame.to_dict(orient="records"),
        predictions,
        strict=True,
    ):
        event = {
            "event": "clv_prediction",
            "event_id": uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_run_id": metadata["mlflow_run_id"],
            "model_version": metadata["model_version"],
            "features": features,
            "prediction": float(prediction),
        }

        print(
            json.dumps(event, allow_nan=False),
            flush=True,
        )
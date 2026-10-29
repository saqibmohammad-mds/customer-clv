import json

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ARTIFACT_DIR,
    DATASET_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.prepare_data import sha256_file


MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"


def load_model_bundle() -> dict:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Trained model is missing. Run python -m src.train"
        )

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    if sha256_file(MODEL_PATH) != metadata["model_file_sha256"]:
        raise ValueError("Model file does not match its metadata.")

    bundle = joblib.load(MODEL_PATH)

    if bundle["metadata"]["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Model features do not match configuration.")

    return bundle


def predict_features(
    bundle: dict,
    features: pd.DataFrame,
) -> np.ndarray:
    columns = bundle["metadata"]["feature_columns"]

    missing = set(columns) - set(features.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    x = features[columns].astype(float)

    if x.empty:
        raise ValueError("At least one customer is required.")

    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Features must contain finite numbers.")

    predictions = np.asarray(
        bundle["model"].predict(x),
        dtype=float,
    )

    if not np.isfinite(predictions).all():
        raise ValueError("Model returned invalid predictions.")

    if (predictions < 0).any():
        raise ValueError("Model returned negative predictions.")

    return predictions


def main() -> None:
    bundle = load_model_bundle()

    if not DATASET_PATH.exists():
        raise FileNotFoundError("Prepared dataset is missing.")

    dataset = pd.read_csv(DATASET_PATH)
    sample = dataset.loc[dataset["split"] == "test"].head(5).copy()

    predictions = predict_features(bundle, sample)

    output = sample[["customer_id", TARGET_COLUMN]].copy()
    output["predicted_clv_90d"] = predictions.round(2)

    print(
        f"Model: {bundle['metadata']['model_name']} | "
        f"Version: {bundle['metadata']['model_version']}"
    )
    print(f"Currency: {bundle['metadata']['currency']}")
    print()
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
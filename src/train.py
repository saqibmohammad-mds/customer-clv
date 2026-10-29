import json
import os
import platform
import subprocess
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor

from src.config import (
    ARTIFACT_DIR,
    CURRENCY,
    DATASET_PATH,
    FEATURE_COLUMNS,
    MANIFEST_PATH,
    PREDICTION_HORIZON_DAYS,
    PROJECT_ROOT,
    RANDOM_SEED,
    REPORT_DIR,
    TARGET_COLUMN,
    ensure_directories,
)
from src.prepare_data import sha256_file


MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
REGISTERED_MODEL_NAME = "customer-clv"


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def configure_mlflow() -> str:
    database_path = (PROJECT_ROOT / "mlflow.db").as_posix()

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{database_path}",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    experiment_name = "customer-clv"

    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=(
                PROJECT_ROOT / "mlartifacts"
            ).as_uri(),
        )

    mlflow.set_experiment(experiment_name)
    return tracking_uri


def calculate_metrics(actual, predicted) -> dict:
    predictions = np.asarray(predicted, dtype=float)

    if not np.isfinite(predictions).all():
        raise ValueError("Model produced invalid predictions.")

    if (predictions < 0).any():
        raise ValueError("Model produced negative CLV predictions.")

    return {
        "mae": float(mean_absolute_error(actual, predictions)),
        "rmse": float(
            np.sqrt(mean_squared_error(actual, predictions))
        ),
        "r2": float(r2_score(actual, predictions)),
    }


def load_dataset():
    if not DATASET_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Prepared data is missing. Run: "
            "python -m src.prepare_data"
        )

    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    if sha256_file(DATASET_PATH) != manifest["dataset_sha256"]:
        raise ValueError(
            "Dataset does not match its manifest. "
            "Run python -m src.prepare_data again."
        )

    if manifest["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Feature configuration has changed.")

    dataset = pd.read_csv(DATASET_PATH)

    required = FEATURE_COLUMNS + [TARGET_COLUMN, "split"]
    missing = set(required) - set(dataset.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    values = dataset[FEATURE_COLUMNS + [TARGET_COLUMN]].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError("Dataset contains missing or invalid values.")

    if dataset[TARGET_COLUMN].lt(0).any():
        raise ValueError("CLV targets cannot be negative.")

    splits = {}

    for split in ("train", "validation", "test"):
        subset = dataset.loc[dataset["split"] == split].copy()

        if len(subset) < 2:
            raise ValueError(f"Insufficient rows in {split}.")

        splits[split] = subset

    return splits, manifest


def build_candidates() -> dict:
    return {
        "baseline_median": DummyRegressor(strategy="median"),

        "random_forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),

        "xgboost": XGBRegressor(
            n_estimators=350,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:tweedie",
            tweedie_variance_power=1.5,
            tree_method="hist",
            random_state=RANDOM_SEED,
            n_jobs=2,
        ),
    }


def main() -> None:
    ensure_directories()

    splits, manifest = load_dataset()
    tracking_uri = configure_mlflow()

    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]

    x_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    x_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[TARGET_COLUMN]

    candidates = build_candidates()
    results = []
    run_ids = {}

    for name, model in candidates.items():
        print(f"\nTraining: {name}")

        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags(
                {
                    "dataset_sha256": manifest["dataset_sha256"],
                    "source_sha256": manifest["source_sha256"],
                    "git_commit": get_git_commit(),
                    "currency": CURRENCY,
                    "target": TARGET_COLUMN,
                }
            )

            mlflow.log_params(
                {
                    "candidate": name,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "horizon_days": PREDICTION_HORIZON_DAYS,
                }
            )

            mlflow.log_params(
                {
                    f"model_{key}": str(value)
                    for key, value in model.get_params().items()
                }
            )

            model.fit(x_train, y_train)

            validation_predictions = model.predict(x_validation)
            metrics = calculate_metrics(
                y_validation,
                validation_predictions,
            )

            mlflow.log_metrics(
                {
                    f"validation_{key}": value
                    for key, value in metrics.items()
                }
            )

            signature = infer_signature(
                x_train,
                model.predict(x_train.head(5)),
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=x_train.head(5),
                pip_requirements=[
                    "numpy==1.26.4",
                    "pandas==2.2.3",
                    "scikit-learn==1.6.1",
                    "xgboost==2.1.4",
                    "mlflow==2.22.0",
                ],
            )

            mlflow.log_artifact(
                str(MANIFEST_PATH),
                artifact_path="data",
            )

            run_ids[name] = run.info.run_id

            results.append(
                {
                    "model": name,
                    **metrics,
                    "run_id": run.info.run_id,
                }
            )

            print(
                f"Validation MAE: {metrics['mae']:.2f} | "
                f"RMSE: {metrics['rmse']:.2f} | "
                f"R²: {metrics['r2']:.3f}"
            )

    leaderboard = (
        pd.DataFrame(results)
        .sort_values(["mae", "model"])
        .reset_index(drop=True)
    )

    leaderboard_path = REPORT_DIR / "model_comparison.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    winner_name = str(leaderboard.iloc[0]["model"])
    winner = candidates[winner_name]
    winner_run_id = run_ids[winner_name]

    # Evaluate only after selection. Do not tune against these results.
    test_predictions = winner.predict(test[FEATURE_COLUMNS])
    test_metrics = calculate_metrics(
        test[TARGET_COLUMN],
        test_predictions,
    )

    test_output = test[
        ["customer_id", "snapshot_date", TARGET_COLUMN]
    ].copy()
    test_output["predicted_clv_90d"] = test_predictions

    predictions_path = REPORT_DIR / "test_predictions.csv"
    test_output.to_csv(predictions_path, index=False)

    with mlflow.start_run(run_id=winner_run_id):
        mlflow.set_tag("selected_by", "lowest_validation_mae")
        mlflow.log_metrics(
            {
                f"test_{key}": value
                for key, value in test_metrics.items()
            }
        )
        mlflow.log_artifact(str(leaderboard_path))
        mlflow.log_artifact(str(predictions_path))

    registered = mlflow.register_model(
        model_uri=f"runs:/{winner_run_id}/model",
        name=REGISTERED_MODEL_NAME,
    )

    metadata = {
        "model_name": winner_name,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "model_version": str(registered.version),
        "mlflow_run_id": winner_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "currency": CURRENCY,
        "prediction_horizon_days": PREDICTION_HORIZON_DAYS,
        "feature_columns": FEATURE_COLUMNS,
        "dataset_sha256": manifest["dataset_sha256"],
        "validation_metrics": {
            key: float(leaderboard.iloc[0][key])
            for key in ("mae", "rmse", "r2")
        },
        "test_metrics": test_metrics,
    }

    # Keep the model and its metadata together for the API.
    bundle = {
        "model": winner,
        "metadata": metadata,
    }

    temporary_path = ARTIFACT_DIR / "model.tmp.joblib"
    joblib.dump(bundle, temporary_path)
    temporary_path.replace(MODEL_PATH)

    metadata["model_file_sha256"] = sha256_file(MODEL_PATH)

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    with mlflow.start_run(run_id=winner_run_id):
        mlflow.log_artifact(str(METADATA_PATH))

    print("\nValidation comparison:")
    print(
        leaderboard[["model", "mae", "rmse", "r2"]]
        .round(3)
        .to_string(index=False)
    )

    print(f"\nSelected model: {winner_name}")
    print(f"Registry version: {registered.version}")
    print(f"Test MAE: {test_metrics['mae']:.2f} {CURRENCY}")
    print(f"Test RMSE: {test_metrics['rmse']:.2f} {CURRENCY}")
    print(f"Test R²: {test_metrics['r2']:.3f}")
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"MLflow tracking: {tracking_uri}")


if __name__ == "__main__":
    main()
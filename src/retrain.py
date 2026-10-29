import json
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from src.config import (
    CURRENCY,
    FEATURE_COLUMNS,
    MANIFEST_PATH,
    PREDICTION_HORIZON_DAYS,
    REPORT_DIR,
    SNAPSHOTS,
    TARGET_COLUMN,
    TRANSACTIONS_PATH,
)
from src.features import build_training_snapshot, validate_transactions
from src.predict import load_model_bundle, predict_features
from src.prepare_data import sha256_file
from src.train import build_candidates, calculate_metrics


def main():
    output_dir = REPORT_DIR / "retraining"
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_path = output_dir / "decision.json"

    # Avoid mistaking files from an earlier local run for a new candidate.
    for filename in ("model.joblib", "model_metadata.json"):
        (output_dir / filename).unlink(missing_ok=True)

    incumbent = load_model_bundle()

    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    source_hash = sha256_file(TRANSACTIONS_PATH)

    if source_hash != manifest["source_sha256"]:
        raise ValueError(
            "Raw data changed without a matching manifest. "
            "Run python -m src.prepare_data first."
        )

    end = pd.Timestamp(manifest["coverage_end_exclusive"])
    horizon = pd.Timedelta(days=PREDICTION_HORIZON_DAYS)
    validation_cutoff = end - horizon

    original_test_end = (
        max(pd.Timestamp(value) for value in SNAPSHOTS["test"])
        + horizon
    )

    last_evaluation_end = pd.Timestamp(
        incumbent["metadata"].get(
            "evaluation_end",
            original_test_end.isoformat(),
        )
    )

    decision = {
        "coverage_end_exclusive": end.isoformat(),
        "validation_cutoff": validation_cutoff.isoformat(),
        "source_sha256": source_hash,
    }

    if validation_cutoff < last_evaluation_end:
        decision["status"] = "skipped"
        decision["reason"] = (
            "No new complete 90-day evaluation window after "
            "the current model's previous evaluation period."
        )
        decision_path.write_text(
            json.dumps(decision, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(decision, indent=2))
        return

    transactions = validate_transactions(
        pd.read_csv(
            TRANSACTIONS_PATH,
            dtype={"customer_id": "string", "order_id": "string"},
        )
    )

    if transactions["order_date"].ge(end).any():
        raise ValueError("Transactions exceed declared coverage.")

    # All training labels finish before validation begins.
    training_cutoffs = [
        validation_cutoff - horizon * offset
        for offset in (3, 2, 1)
    ]

    train = pd.concat(
        [
            build_training_snapshot(
                transactions,
                cutoff.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
            )
            for cutoff in training_cutoffs
        ],
        ignore_index=True,
    )

    validation = build_training_snapshot(
        transactions,
        validation_cutoff.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )

    if len(train) < 100 or len(validation) < 50:
        raise ValueError("Insufficient customers for retraining.")

    model = build_candidates()["xgboost"]

    database = (output_dir / "mlflow.db").resolve().as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{database}")

    experiment_name = "clv-retraining"

    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=(
                output_dir / "mlartifacts"
            ).resolve().as_uri(),
        )

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="xgboost-candidate") as run:
        model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])

        candidate_predictions = model.predict(
            validation[FEATURE_COLUMNS]
        )
        incumbent_predictions = predict_features(
            incumbent,
            validation[FEATURE_COLUMNS],
        )

        candidate_metrics = calculate_metrics(
            validation[TARGET_COLUMN],
            candidate_predictions,
        )
        incumbent_metrics = calculate_metrics(
            validation[TARGET_COLUMN],
            incumbent_predictions,
        )

        passed = (
            candidate_metrics["mae"] < incumbent_metrics["mae"] * 0.95
            and candidate_metrics["rmse"] <= incumbent_metrics["rmse"] * 1.05
        )

        mlflow.log_params({
            "source_sha256": source_hash,
            "validation_cutoff": validation_cutoff.isoformat(),
            "coverage_end": end.isoformat(),
            "train_rows": len(train),
            "validation_rows": len(validation),
        })

        for label, metrics in (
            ("candidate", candidate_metrics),
            ("incumbent", incumbent_metrics),
        ):
            mlflow.log_metrics({
                f"{label}_{key}": value
                for key, value in metrics.items()
            })

        mlflow.set_tag("promotion_gate_passed", str(passed))

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=train[FEATURE_COLUMNS].head(5),
            signature=infer_signature(
                train[FEATURE_COLUMNS].head(5),
                model.predict(train[FEATURE_COLUMNS].head(5)),
            ),
        )

        metadata = {
            "model_name": "xgboost",
            "model_version": f"candidate-{run.info.run_id[:12]}",
            "mlflow_run_id": run.info.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "currency": CURRENCY,
            "prediction_horizon_days": PREDICTION_HORIZON_DAYS,
            "feature_columns": FEATURE_COLUMNS,
            "source_sha256": source_hash,
            "dataset_sha256": manifest["dataset_sha256"],
            "evaluation_end": end.isoformat(),
            "validation_metrics": candidate_metrics,
            "promotion_gate_passed": bool(passed),
        }

        model_path = output_dir / "model.joblib"

        joblib.dump(
            {"model": model, "metadata": metadata},
            model_path,
        )

        metadata["model_file_sha256"] = sha256_file(model_path)

        (output_dir / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        decision.update({
            "status": "eligible_for_review" if passed else "rejected",
            "candidate": candidate_metrics,
            "incumbent": incumbent_metrics,
            "model_version": metadata["model_version"],
        })

        decision_path.write_text(
            json.dumps(decision, indent=2),
            encoding="utf-8",
        )

        mlflow.log_artifact(str(decision_path))

    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
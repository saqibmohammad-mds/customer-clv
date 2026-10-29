import argparse
import json
import os
from datetime import timedelta

import pandas as pd
from azure.identity import AzureCliCredential
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.config import DATASET_PATH, FEATURE_COLUMNS, REPORT_DIR
from src.predict import load_model_bundle, predict_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    output_dir = REPORT_DIR / "monitoring"
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_model_bundle()
    dataset = pd.read_csv(DATASET_PATH)

    reference = dataset.loc[
        dataset["split"] == "validation",
        FEATURE_COLUMNS,
    ].copy()

    reference["prediction"] = predict_features(bundle, reference)

    if args.demo:
        current = dataset.loc[
            dataset["split"] == "test",
            FEATURE_COLUMNS,
        ].copy()
        current["prediction"] = predict_features(bundle, current)
        source = "historical_demo"
    else:
        workspace_id = os.environ["LOGS_WORKSPACE_ID"]
        app_name = os.environ.get(
            "AZURE_CONTAINER_APP",
            "customer-clv-api",
        )

        # Only query this project's app. JSON quoting prevents the
        # app name from changing the structure of the KQL query.
        query = f"""
        ContainerAppConsoleLogs_CL
        | where ContainerAppName_s == {json.dumps(app_name)}
        | where Log_s startswith '{{'
        | extend payload = parse_json(Log_s)
        | where tostring(payload.event) == "clv_prediction"
        | order by TimeGenerated desc
        | take 10000
        | project Log_s
        """

        with AzureCliCredential() as credential:
            with LogsQueryClient(credential) as client:
                response = client.query_workspace(
                    workspace_id,
                    query,
                    timespan=timedelta(days=7),
                )

        if response.status != LogsQueryStatus.SUCCESS:
            raise RuntimeError(
                "Azure returned incomplete monitoring data. "
                "Check workspace access and the query."
            )

        records = []
        seen = set()

        for table in response.tables:
            for row in table.rows:
                event = json.loads(row[0])

                if event["model_run_id"] != bundle["metadata"]["mlflow_run_id"]:
                    continue

                if event["event_id"] in seen:
                    continue

                seen.add(event["event_id"])

                records.append({
                    **event["features"],
                    "prediction": event["prediction"],
                })

        current = pd.DataFrame(records)
        source = "azure_live"

    summary = {
        "source": source,
        "model_run_id": bundle["metadata"]["mlflow_run_id"],
        "reference_rows": len(reference),
        "current_rows": len(current),
    }

    summary_path = output_dir / "summary.json"

    if len(current) < 50:
        summary["status"] = "insufficient_data"
        summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2))
        return

    columns = FEATURE_COLUMNS + ["prediction"]

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference[columns],
        current_data=current[columns],
    )

    report.save_html(str(output_dir / "drift_report.html"))
    report.save_json(str(output_dir / "drift_report.json"))

    drift_result = next(
        metric["result"]
        for metric in report.as_dict()["metrics"]
        if metric["metric"] == "DatasetDriftMetric"
    )

    summary["status"] = (
        "drift_detected"
        if drift_result["dataset_drift"]
        else "no_drift_detected"
    )
    summary["share_of_drifted_columns"] = (
        drift_result["share_of_drifted_columns"]
    )

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))

    # A live drift finding fails the scheduled check so it is visible
    # in GitHub Actions. It does not replace the deployed model.
    if source == "azure_live" and drift_result["dataset_drift"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
"""Minimal local MLflow tracking for the Internet Ads hackathon."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import mlflow

PROJECT_DIR = Path(__file__).resolve().parent
TRACKING_DIR = PROJECT_DIR / "artifacts" / "mlflow"
TRACKING_DB = PROJECT_DIR / "artifacts" / "mlflow.db"
EXPERIMENT_NAME = "Internet Ads Hackathon (final artifacts)"


def configure_tracking() -> str:
    """Use a project-local SQLite backend and artifact directory, without a server."""
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{TRACKING_DB.resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        client.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=TRACKING_DIR.resolve().as_uri(),
        )
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment.experiment_id


def log_run(
    run_name: str,
    *,
    params: Mapping[str, Any],
    metrics: Mapping[str, float],
    tags: Optional[Mapping[str, str]] = None,
    artifacts: Sequence[Path | str] = (),
) -> str:
    """Log one completed run and return its MLflow run ID.

    Non-numeric and non-finite metrics are skipped rather than converted into
    fabricated zeros. Artifact files are logged only if they exist.
    """
    configure_tracking()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({key: str(value)[:500] for key, value in params.items()})
        numeric_metrics = {}
        for key, value in metrics.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number == number and abs(number) != float("inf"):
                numeric_metrics[key] = number
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)
        if tags:
            mlflow.set_tags({key: str(value)[:500] for key, value in tags.items()})
        for artifact in artifacts:
            artifact_path = Path(artifact)
            if artifact_path.is_file():
                mlflow.log_artifact(str(artifact_path))
        return run.info.run_id

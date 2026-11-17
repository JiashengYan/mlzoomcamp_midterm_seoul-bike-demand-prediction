"""Local CSV replay service; the caller supplies an issue time, not weather."""
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from threadpoolctl import threadpool_limits

from .data import ROOT, TARGET, TZ, features_at, load_data


@asynccontextmanager
async def lifespan(app):
    """Load the saved model and source observations before serving requests.

    Inputs:
        app: FastAPI application. Uses models/model.joblib and the fixed CSV.

    Outputs:
        A startup/shutdown context for FastAPI. After loading, the model and
        observations are stored on app.state for request handlers. A missing model
        leaves the service unready (HTTP 503); invalid model/data raises an error.
    """
    artifact = ROOT / "models/model.joblib"
    app.state.bundle = None
    if artifact.exists():
        bundle = joblib.load(artifact)
        app.state.raw = load_data()
        app.state.bundle = bundle
        app.state.issue_start = app.state.raw.index[169] - pd.Timedelta(hours=1)
        app.state.issue_end = app.state.raw.index[-1] - pd.Timedelta(hours=1)
    yield


app = FastAPI(title="Seoul bike demand: observed-history replay", version="0.1.0", lifespan=lifespan)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    issued_at: AwareDatetime = Field(
        description=(
            "Historical replay issue time, with a timezone and zero minutes/seconds. "
            "Any recorded target hour with 168 completed hourly observations is allowed. "
            "GET /health reports the available range and the formal test range. "
            "The target hour starts one hour later; do not enter the current date."
        ),
        examples=["2018-11-16T17:00:00+09:00"],
    )


def ready():
    """Return the loaded model or explain that training is required.

    Inputs:
        No arguments. Reads the model bundle from app.state, initialized by lifespan.

    Outputs:
        The model bundle dictionary. Raises HTTPException with status 503 when no model
        is loaded.
    """
    if app.state.bundle is None:
        raise HTTPException(status_code=503, detail="Model unavailable. Run bike-train first.")
    return app.state.bundle


@app.get("/health")
def health():
    """Report whether the prediction service has a loaded model.

    Inputs:
        No arguments; uses the application state through ready.

    Outputs:
        A dictionary with ready status and model ID, or HTTP 503 if the model is
        unavailable.
    """
    bundle = ready()
    meta = bundle["metadata"]
    return {
        "status": "ready", "model_id": meta["model_id"],
        "issue_start": app.state.issue_start.isoformat(),
        "issue_end": app.state.issue_end.isoformat(),
        "evaluation_issue_start": meta["issue_start"],
        "evaluation_issue_end": meta["issue_end"],
        "timezone": TZ, "issued_at_requires_exact_hour": True,
    }


@app.post("/predict", description=(
    "Replay a historical prediction and compare it with the CSV's recorded target-hour "
    "rentals. Actual values are read only after prediction and are not model inputs. "
    "Percentage error is null when actual rentals are zero. Comparison on shutdown "
    "hours is not an evaluation of the model's normal-operation assumption."
))
def predict(request: PredictionRequest):
    """Predict the target hour starting one hour after the supplied issued_at.

    Inputs:
        request: PredictionRequest containing timezone-aware issued_at. This field
            chooses the historical issue time; it is not the current HTTP arrival time.
            It must be an exact hour within the saved model's accepted range.

    Outputs:
        A dictionary with predicted/actual rentals, errors, target interval, model ID,
        and normal-operation assumption. history_available_through ends completed
        observations; latest_source_hour_start is the start of the newest source record.
        training_label_available_through is the end of the last training-label hour.
        Raises HTTP 422 for invalid time/history or 503 when no model is loaded.
    """
    bundle = ready()
    meta = bundle["metadata"]
    start, end = app.state.issue_start, app.state.issue_end
    issue = pd.Timestamp(request.issued_at).tz_convert(TZ)
    if issue != issue.floor("h"):
        raise HTTPException(status_code=422, detail=(
            "issued_at must be an exact hour (minutes, seconds, and fractions must be "
            "zero). Example: 2018-11-16T17:00:00+09:00. "
            f"Replay issue range: {start.isoformat()} through {end.isoformat()}."
        ))
    if not start <= issue <= end:
        raise HTTPException(status_code=422, detail=(
            f"Replay issue range: {start.isoformat()} through {end.isoformat()}."
        ))
    try:
        row = features_at(app.state.raw, issue)[bundle["columns"]]
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    with threadpool_limits(limits=2):
        count = float(np.maximum(0, bundle["model"].predict(row)[0]))
    target = issue + pd.Timedelta(hours=1)
    actual = int(app.state.raw.loc[target, TARGET])
    absolute_error = abs(count - actual)
    operating = bool(app.state.raw.loc[target, "Functioning Day"] == "Yes")
    used_in_training = (
        target < pd.Timestamp(meta["training_label_available_through"]) and operating
    )
    in_test = (
        pd.Timestamp(meta["issue_start"]) <= issue <= pd.Timestamp(meta["issue_end"])
        and operating
    )
    return {
        "predicted_rentals": count, "unit": "rentals/hour", "issued_at": issue.isoformat(),
        "actual_rentals": actual,
        "absolute_error": absolute_error,
        "absolute_percentage_error": 100 * absolute_error / actual if actual else None,
        "actual_operating": operating,
        "used_in_training": used_in_training,
        "evaluation_scope": "training" if used_in_training else (
            "held_out_test" if in_test else "outside_evaluation"
        ),
        "target_start": target.isoformat(), "target_end": (target + pd.Timedelta(hours=1)).isoformat(),
        # Example: observations END at 17:00; newest record STARTS at 16:00.
        "history_available_through": issue.isoformat(),
        "latest_source_hour_start": (issue - pd.Timedelta(hours=1)).isoformat(),
        "model_id": meta["model_id"],
        "training_label_available_through": meta["training_label_available_through"],
        "simulated_model_available_at": meta["simulated_available_at"],
        "assumption": meta["assumption"],
    }

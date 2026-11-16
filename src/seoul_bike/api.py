"""Local CSV replay service; the caller supplies an issue time, not weather."""
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import AwareDatetime, BaseModel, ConfigDict
from threadpoolctl import threadpool_limits

from .data import ROOT, SOURCE_SHA256, TZ, features_at, load_data


@asynccontextmanager
async def lifespan(app):
    """Load the saved model and source observations before serving requests.

    Inputs:
        app: FastAPI application. Uses models/model.joblib and the fixed CSV.

    Outputs:
        A startup/shutdown context for FastAPI. After loading, the model and
        observations are stored on app.state for request handlers. A missing model
        leaves the service unready (HTTP 503); an invalid model/data version raises an
        error.
    """
    artifact = ROOT / "models/model.joblib"
    app.state.bundle = None
    if artifact.exists():
        bundle = joblib.load(artifact)
        if bundle["metadata"]["source_sha256"] != SOURCE_SHA256:
            raise ValueError("Model and dataset versions do not match.")
        app.state.raw = load_data()
        app.state.bundle = bundle
    yield


app = FastAPI(title="Seoul bike demand: observed-history replay", version="0.1.0", lifespan=lifespan)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    issued_at: AwareDatetime


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
    return {"status": "ready", "model_id": bundle["metadata"]["model_id"]}


@app.post("/predict")
def predict(request: PredictionRequest):
    """Predict the target hour starting one hour after the supplied issued_at.

    Inputs:
        request: PredictionRequest containing timezone-aware issued_at. This field
            chooses the historical issue time; it is not the current HTTP arrival time.
            It must be an exact hour within the saved model's accepted range.

    Outputs:
        A dictionary with estimated rentals, target interval, model ID, and the
        normal-operation assumption. history_available_through is the end of completed
        observations; latest_source_hour_start is the start of the newest source record.
        training_label_available_through is the end of the last training-label hour.
        Raises HTTP 422 for invalid time/history or 503 when no model is loaded.
    """
    bundle = ready()
    meta = bundle["metadata"]
    issue = pd.Timestamp(request.issued_at).tz_convert(TZ)
    if issue != issue.floor("h"):
        raise HTTPException(status_code=422, detail="issued_at must be an exact hour.")
    if not pd.Timestamp(meta["issue_start"]) <= issue <= pd.Timestamp(meta["issue_end"]):
        raise HTTPException(status_code=422, detail=f"Replay issue range: {meta['issue_start']} through {meta['issue_end']}.")
    try:
        row = features_at(app.state.raw, issue)[bundle["columns"]]
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    with threadpool_limits(limits=2):
        count = float(np.maximum(0, bundle["model"].predict(row)[0]))
    target = issue + pd.Timedelta(hours=1)
    return {
        "predicted_rentals": count, "unit": "rentals/hour", "issued_at": issue.isoformat(),
        "target_start": target.isoformat(), "target_end": (target + pd.Timedelta(hours=1)).isoformat(),
        # Example: observations END at 17:00; newest record STARTS at 16:00.
        "history_available_through": issue.isoformat(),
        "latest_source_hour_start": (issue - pd.Timedelta(hours=1)).isoformat(),
        "model_id": meta["model_id"],
        "training_label_available_through": meta["training_label_available_through"],
        "simulated_model_available_at": meta["simulated_available_at"],
        "assumption": meta["assumption"],
    }

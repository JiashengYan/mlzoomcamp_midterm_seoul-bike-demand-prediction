"""Batch retraining and artifact export using the shared development functions."""
import argparse
from datetime import datetime, timezone
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from .data import (
    ROOT, SOURCE_SHA256, TARGET, DEV_END, ISSUE_START, ISSUE_END,
    load_data, build_features, feature_columns, sample_masks, validation_splits, baselines,
)
from .modeling import (
    BASE, make_model, scores, prediction_frame, diagnostics, evaluate,
    tuning_grid, select_candidate,
)


def plot_results(table, validation, test, figures):
    """Save readable model-comparison and final-test prediction charts.

    Inputs:
        table: Candidate results.
        validation: Selected validation predictions.
        test: Fixed-model test predictions.
        figures: Existing output directory.

    Outputs:
        None. Writes validation.png and replay.png to figures and closes the plots;
        existing files with those names are replaced.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    ranked_candidates = table.sort_values("mae")
    best_per_family = ranked_candidates.groupby("family", sort=False).first()
    best = best_per_family.sort_values("mae")
    axes[0].bar(best.index, best.mae, color="#277da8")
    axes[0].axhline(scores(validation.actual, validation.last_week)["mae"],
                    color="#d76b31", linestyle="--", label="Last week baseline")
    axes[0].set(title="Development: best configuration per family", ylabel="MAE (rentals/hour)")
    axes[0].legend()
    absolute_errors = (validation.prediction - validation.actual).abs()
    errors_by_hour = validation.assign(error=absolute_errors)
    by_season = errors_by_hour.groupby("season").error.mean()
    axes[1].bar(by_season.index, by_season, color="#489b81")
    axes[1].set(title="Selected model: validation by season", ylabel="MAE (rentals/hour)")
    fig.savefig(figures / "validation.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    axes[0].plot(test.index, test.actual, label="Actual", linewidth=1)
    axes[0].plot(test.index, test.prediction, label="Prediction", linewidth=1, alpha=0.85)
    axes[0].set(ylabel="Rentals/hour", title="Fixed model, hourly observed-history replay: final test")
    axes[0].legend()
    axes[1].plot(test.index, test.prediction - test.actual, color="#c77036", linewidth=0.8)
    axes[1].axhline(0, color="gray", linewidth=0.7)
    axes[1].set(ylabel="Prediction - actual", xlabel="Target hour (Asia/Seoul)")
    fig.savefig(figures / "replay.png", dpi=150)
    plt.close(fig)


def run():
    """Compare candidates, save one final model, and evaluate the final test period.

    Inputs:
        No arguments. Reads the fixed CSV and split constants through data.py; uses BASE
            and tuning_grid for candidate configurations.

    Outputs:
        None. Writes models/model.joblib, report CSV/JSON files, and figures, replacing
        earlier generated outputs. Prints progress and final metrics. Model selection
        finishes before final-test scoring.
    """
    raw = load_data()
    features, baseline = build_features(raw), baselines(raw)
    dev, test = sample_masks(raw)
    splits = list(validation_splits(raw))
    reports, models = ROOT / "reports", ROOT / "models"
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    models.mkdir(exist_ok=True)
    manifest = pd.DataFrame({"role": np.where(dev, "development",
                           np.where(test, "test", "history_only"))}, index=raw.index)
    for fold, (train, val) in enumerate(splits, 1):
        manifest[f"round_{fold}"] = np.where(train, "train", np.where(val, "validation",
                                              np.where(dev, "excluded", "outside_development")))
    manifest.to_csv(reports / "splits.csv")
    rows, predictions = [], {}
    def compare(family, group, params, phase):
        """Evaluate one configuration and keep its result in the current training run.

        Inputs:
            family: Model type.
            group: Weather feature group.
            params: Hyperparameters.
            phase: initial or tuning. Uses raw, features, baseline, splits, rows, and
                predictions from the enclosing run.

        Outputs:
            None. Appends a metrics row and stores predictions under a new candidate ID
            in the enclosing run; evaluate prints progress.
        """
        identifier = f"c{len(rows) + 1:02d}"
        row, pred = evaluate(raw, features, baseline, splits, family, group, params, phase, identifier)
        rows.append(row)
        predictions[identifier] = pred
    for family, params in BASE.items():
        for group in ("W1", "W2", "W3"):
            compare(family, group, params, "initial")
    initial_results = pd.DataFrame(rows).sort_values("mae")
    best_initial_per_family = initial_results.drop_duplicates("family")
    initial = best_initial_per_family.head(2)
    for _, candidate in initial.iterrows():
        for params in tuning_grid(candidate.family):
            if params != BASE[candidate.family]:
                compare(candidate.family, candidate.features, params, "tuning")
    table = pd.DataFrame(rows)
    selected = select_candidate(table)
    selected_predictions = predictions[selected.id]
    table["selected"] = table.id.eq(selected.id)
    table.to_csv(reports / "validation.csv", index=False)
    selected_predictions.to_csv(reports / "validation_predictions.csv")
    print(f"Selected {selected.id}: {selected.family}/{selected.features}; fitting all {dev.sum()} development labels.", flush=True)
    columns = feature_columns(features, selected.features)
    params = json.loads(selected.params)
    model = make_model(selected.family, columns, params)
    model.fit(features.loc[dev, columns], raw.loc[dev, TARGET])
    metadata = {
        "model_id": f"seoul-bike-v1-{selected.id}", "feature_version": 1,
        "family": selected.family, "weather_features": selected.features, "params": params,
        "source_sha256": SOURCE_SHA256, "training_label_available_through": DEV_END.isoformat(),
        "simulated_available_at": ISSUE_START.isoformat(),
        "issue_start": ISSUE_START.isoformat(), "issue_end": ISSUE_END.isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "assumption": "Normal operation; hourly observations available when their hour ends.",
    }
    # Finish choosing the model before looking at any final-test error.
    joblib.dump({"model": model, "columns": columns, "metadata": metadata}, models / "model.joblib")
    test_prediction = np.maximum(0, model.predict(features.loc[test, columns]))
    test_predictions = prediction_frame(raw, test, test_prediction, baseline,
                                         raw.loc[dev, TARGET].quantile(0.9), 0)
    test_predictions.to_csv(reports / "test_predictions.csv")
    metrics = {
        "metadata": metadata,
        "samples": {"raw": len(raw), "development": int(dev.sum()), "test": int(test.sum()),
                    "rounds": [{"train": int(t.sum()), "validation": int(v.sum()),
                                "excluded": int((dev & ~t & ~v).sum())} for t, v in splits]},
        "selection": {"candidate": selected.id, "candidates": len(table), "rule": "Within 1% of best MAE, minimize worst-season baseline ratio, then bytes and inference time."},
        "validation": diagnostics(selected_predictions), "test": diagnostics(test_predictions),
        "evaluation_scope": (
            "Development validation may train on months later than its held-out weeks. "
            "The final-test model uses only earlier training targets; each prediction "
            "uses observations completed by its supplied issue time."
        ),
    }
    (reports / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_results(table, selected_predictions, test_predictions, figures)
    print(json.dumps({"selected": metadata, "validation": metrics["validation"]["overall"],
                      "test": metrics["test"]["overall"], "test_baselines": metrics["test"]["baselines"]},
                     ensure_ascii=False, indent=2), flush=True)


def main():
    """Run the full model-comparison and final-training command.

    Inputs:
        No function arguments. Reads command-line arguments; only standard help is
            supported.

    Outputs:
        None on success. Runs training with at most two numerical threads. argparse
        exits when help or invalid arguments are supplied.
    """
    argparse.ArgumentParser(description=__doc__).parse_args()
    with threadpool_limits(limits=2):
        run()


if __name__ == "__main__":
    main()

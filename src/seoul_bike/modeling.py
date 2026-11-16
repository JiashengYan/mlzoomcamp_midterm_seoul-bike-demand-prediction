"""Shared model fitting and evaluation for the development notebook and batch job."""
from io import BytesIO
from itertools import product
import json
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import TARGET, feature_columns


BASE = {
    "ridge": {"alpha": 1.0},
    "forest": {"max_depth": 16, "min_samples_leaf": 10},
    "boosting": {"max_leaf_nodes": 15, "l2_regularization": 10.0},
}


def make_model(family, columns, params):
    """Prepare the data-cleaning steps and model to be learned from training rows.

    Inputs:
        family: ridge, forest, or boosting.
        columns: Names of the selected input columns.
        params: Settings for that model, such as Ridge alpha or tree depth.

    Outputs:
        An untrained sklearn Pipeline: a sequence that first fills missing values,
        encodes categories, and, for Ridge, rescales numbers, then trains the regressor.
        Calling fit learns every step only from the supplied training rows. A regressor
        estimates a numeric rental count.
    """
    if family == "ridge":
        categories = ["hour_kind", "weekday"]
        numeric = [c for c in columns if c not in ("hour", "weekend", *categories)]
        numeric_pipe = Pipeline([("fill", SimpleImputer(strategy="median")),
                                 ("scale", StandardScaler())])
        prep = ColumnTransformer([
            ("numeric", numeric_pipe, numeric),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categories),
        ])
        estimator = Ridge(**params)
    else:
        numeric = [c for c in columns if c != "hour_kind"]
        prep = ColumnTransformer([("numeric", SimpleImputer(strategy="median"), numeric)])
        if family == "forest":
            estimator = RandomForestRegressor(n_estimators=200, n_jobs=2, random_state=42, **params)
        else:
            estimator = HistGradientBoostingRegressor(
                max_iter=150, min_samples_leaf=25, learning_rate=0.08,
                early_stopping=False, random_state=42, **params)
    return Pipeline([("prepare", prep), ("regressor", estimator)])


def scores(actual, prediction):
    """Calculate prediction error in the original rental-count units.

    Inputs:
        actual: Observed counts.
        prediction: Predicted counts in the same row order; both arrays must be
            nonempty.

    Outputs:
        A dictionary containing MAE, RMSE, and mean bias. Bias is prediction minus
        actual; positive bias means overprediction.
    """
    error = np.asarray(prediction) - np.asarray(actual)
    return {"mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "bias": float(error.mean())}


def prediction_frame(raw, mask, prediction, baseline, threshold, fold):
    """Put predictions beside actual counts to inspect where errors occur.

    Inputs:
        raw: Complete hourly observations.
        mask: True/False selector for the target hours being scored.
        prediction: Predicted counts in the selected rows' order.
        baseline: Rule-based predictions indexed by target hour.
        threshold: Rental count below which 90% of this training set falls.
        fold: Comparison round number; zero means the final test.

    Outputs:
        A table of actual counts, model/rule predictions, and group labels used to
        inspect errors. Actual target-hour rain/snow only groups already-made
        predictions; those values are never supplied to the model.
    """
    part = baseline.loc[mask].copy()
    part["actual"] = raw.loc[mask, TARGET].to_numpy()
    part["prediction"] = prediction
    part["fold"] = fold
    part["season"] = raw.loc[mask, "Seasons"].to_numpy()
    part["hour"] = part.index.hour
    part["day_type"] = np.where(part.index.day_of_week >= 5, "weekend", "weekday")
    part["holiday"] = raw.loc[mask, "Holiday"].to_numpy()
    # These target weather flags are post-hoc diagnostics, never model inputs.
    part["rain_positive"] = raw.loc[mask, "Rainfall(mm)"].gt(0).to_numpy()
    part["snow_positive"] = raw.loc[mask, "Snowfall (cm)"].gt(0).to_numpy()
    part["high_count"] = part["actual"].ge(threshold)
    return part


def diagnostics(predictions):
    """Summarize overall errors and errors for specific groups of hours.

    Inputs:
        predictions: The nonempty comparison table produced by prediction_frame,
            possibly concatenated across validation rounds.

    Outputs:
        A dictionary of overall and baseline metrics plus breakdowns by round, season,
        hour, day type, holiday, rain, snow, and high-count status, with hour and date
        counts.
    """
    result = {"overall": scores(predictions.actual, predictions.prediction),
              "baselines": {c: scores(predictions.actual, predictions[c])
                            for c in ("yesterday", "last_week", "weekly_mean")}}
    for column in ("fold", "season", "hour", "day_type", "holiday",
                   "rain_positive", "snow_positive", "high_count"):
        result[column] = {
            str(value): {**scores(part.actual, part.prediction),
                         "hours": len(part), "dates": len(set(part.index.date))}
            for value, part in predictions.groupby(column)
        }
    return result


def evaluate(raw, features, baseline, splits, family, group, params, phase, identifier):
    """Train and score one candidate using the same three validation comparisons.

    Inputs:
        raw: Hourly observations indexed by source-hour start.
        features: Input rows indexed by their target-hour start.
        baseline: Historical rule predictions indexed by target hour.
        splits: Pairs of True/False arrays choosing each round's training and validation
            targets.
        family: ridge, forest, or boosting.
        group: W1, W2, or W3 weather-input selection.
        params: Model settings to evaluate.
        phase: initial or tuning, recorded in the comparison report.
        identifier: Unique candidate ID, such as c01.

    Outputs:
        A result dictionary and a table combining the three rounds' predictions. Errors
        are compared with actual counts and last-week predictions by season. Also
        measures saved-model bytes and prediction time per row; prints progress.
    """
    columns = feature_columns(features, group)
    parts, sizes, times = [], [], []
    for fold, (train, val) in enumerate(splits, 1):
        model = make_model(family, columns, params)
        model.fit(features.loc[train, columns], raw.loc[train, TARGET])
        started = perf_counter()
        prediction = np.maximum(0, model.predict(features.loc[val, columns]))
        times.append((perf_counter() - started) / int(val.sum()) * 1000)
        buffer = BytesIO()
        joblib.dump(model, buffer)
        sizes.append(buffer.tell())
        threshold = raw.loc[train, TARGET].quantile(0.9)
        parts.append(prediction_frame(raw, val, prediction, baseline, threshold, fold))
    predictions = pd.concat(parts).sort_index()
    ratios = [scores(p.actual, p.prediction)["mae"] / scores(p.actual, p.last_week)["mae"]
              for _, p in predictions.groupby("season")]
    row = {"id": identifier, "phase": phase, "family": family, "features": group,
           "params": json.dumps(params, sort_keys=True), **scores(predictions.actual, predictions.prediction),
           "worst_season_ratio": max(ratios), "model_bytes": max(sizes),
           "predict_ms_per_row": float(np.mean(times))}
    print(f"{identifier}: {family:8s} {group} MAE={row['mae']:.2f} {params}", flush=True)
    return row, predictions


def tuning_grid(family):
    """List the small predefined set of configurations for one model family.

    Inputs:
        family: ridge, forest, or boosting.

    Outputs:
        A list of parameter dictionaries. The caller skips the already evaluated base
        configuration; no final-test score is used.
    """
    if family == "ridge":
        return [{"alpha": x} for x in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)]
    if family == "forest":
        return [dict(max_depth=d, min_samples_leaf=l) for d, l in product((10, 16), (3, 10, 25))]
    return [dict(max_leaf_nodes=l, l2_regularization=r) for l, r in product((7, 15, 31), (1.0, 10.0, 30.0))]


def select_candidate(table):
    """Select a model by validation error, then stability and running cost.

    Inputs:
        table: Candidate results containing mae, worst_season_ratio, model_bytes, and
            predict_ms_per_row. The seasonal ratio is model error divided by last-week
            rule error for that season.

    Outputs:
        One candidate row (pandas Series). Keep candidates no more than 1% above minimum
        MAE: if minimum is 100, 100.8 qualifies and 102 does not. Then prefer the
        smallest worst seasonal ratio, saved-model size, and prediction time, in that
        order.
    """
    close = table.loc[table.mae <= table.mae.min() * 1.01]
    return close.sort_values(["worst_season_ratio", "model_bytes", "predict_ms_per_row"]).iloc[0]


def feature_group_importance(raw, features, splits, family, group, params, repeats=5):
    """Measure how much validation error rises when one input group is scrambled.

    Inputs:
        raw: Complete observations containing TARGET.
        features: Historical input table aligned with raw.
        splits: Training/validation selectors from validation_splits.
        family: Selected model family.
        group: Selected W1, W2, or W3 weather configuration.
        params: Selected model settings; no settings are changed here.
        repeats: Positive number of shuffles with a fixed random seed.

    Outputs:
        A table of mean MAE increase and shuffle standard deviation per input group,
        weighted by validation-hour counts across rounds. Fits one model per round;
        changes neither inputs nor saved models. Never reads final-test targets.
    """
    columns = feature_columns(features, group)
    groups = {
        "Known target time": [c for c in columns
                              if not c.startswith(("count_", "active_", "w1_", "w2_", "w3_"))],
        "Rental history": [c for c in columns if c.startswith("count_")],
        "Weather history": [c for c in columns if c.startswith(("w1_", "w2_", "w3_"))],
        "Past operating status": [c for c in columns if c.startswith("active_")],
    }
    groups = {name: names for name, names in groups.items() if names}
    increases = {name: np.zeros(repeats) for name in groups}
    rng = np.random.default_rng(42)
    hours = 0
    for train, validation in splits:
        model = make_model(family, columns, params)
        model.fit(features.loc[train, columns], raw.loc[train, TARGET])
        inputs = features.loc[validation, columns]
        actual = raw.loc[validation, TARGET]
        reference = scores(actual, np.maximum(0, model.predict(inputs)))["mae"]
        weeks = inputs.index.tz_localize(None).to_period("W")
        for name, names in groups.items():
            for repeat in range(repeats):
                order = np.arange(len(inputs))
                # Shuffle within each held-out week to retain its seasonal context.
                for week in pd.unique(weeks):
                    positions = np.flatnonzero(weeks == week)
                    order[positions] = rng.permutation(positions)
                changed = inputs.copy()
                # Use one row order for all columns in a group.
                for column in names:
                    changed[column] = inputs[column].to_numpy()[order]
                error = scores(actual, np.maximum(0, model.predict(changed)))["mae"]
                increases[name][repeat] += (error - reference) * len(inputs)
        hours += len(inputs)
    return pd.DataFrame([
        {"group": name, "mae_increase": (values / hours).mean(),
         "shuffle_std": (values / hours).std()}
        for name, values in increases.items()
    ]).sort_values("mae_increase", ascending=False).reset_index(drop=True)

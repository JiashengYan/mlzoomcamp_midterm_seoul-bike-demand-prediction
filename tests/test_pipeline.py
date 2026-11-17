"""Check available-history inputs, target splits, and API predictions."""
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits
from fastapi.testclient import TestClient

from seoul_bike.data import (
    TARGET, WEATHER, ISSUE_START, load_data, build_features, features_at,
    feature_columns, sample_masks, validation_splits, baselines,
)
from seoul_bike.modeling import make_model, select_candidate


@pytest.fixture(scope="module")
def raw():
    """Load the fixed observations once for the tests in this module.

    Inputs:
        No arguments; reads the project's fixed CSV through load_data.

    Outputs:
        A DataFrame shared by tests; tests copy it before making deliberate changes.
    """
    return load_data()


@pytest.fixture(scope="module")
def features(raw):
    """Build the shared feature table once for the tests in this module.

    Inputs:
        raw: The module-scoped observation fixture.

    Outputs:
        A feature DataFrame aligned to raw, including early rows with incomplete
        history.
    """
    return build_features(raw)


def test_split_contract(raw):
    """Check target counts, round counts, coverage, and final-test separation.

    Inputs:
        raw: Complete observations from the fixture.

    Outputs:
        None. Assertion failure if fixed sample counts change, development targets are
        unused in every training round, or development and test targets overlap.
    """
    dev, test = sample_masks(raw)
    assert (dev.sum(), test.sum()) == (7910, 384)
    splits = list(validation_splits(raw))
    assert [(t.sum(), v.sum(), (dev & ~t & ~v).sum()) for t, v in splits] == [
        (6562, 624, 724), (6593, 593, 724), (6613, 624, 673)]
    assert np.logical_or.reduce([t for t, _ in splits]).sum() == dev.sum()
    assert sum(v.sum() for _, v in splits) == 1841
    assert not np.any(dev & test)


def test_feature_alignment(raw, features):
    """Check several historical inputs against their original source records.

    Inputs:
        raw: Complete observations.
        features: Shared feature table.

    Outputs:
        None. Assertion failure for incorrect rental lags/means, weather changes,
        humidity markers, or an unavailable early weekly baseline.
    """
    target = pd.Timestamp("2018-07-15 18:00", tz="Asia/Seoul")
    i = raw.index.get_loc(target)
    row = features.loc[target]
    assert row.count_lag2 == raw[TARGET].iloc[i - 2]
    assert row.count_lag168 == raw[TARGET].iloc[i - 168]
    assert row.count_mean168 == pytest.approx(raw[TARGET].iloc[i - 169:i - 1].mean())
    assert row.w2_temp_delta3 == raw["Temperature(°C)"].iloc[i - 2] - raw["Temperature(°C)"].iloc[i - 5]
    humidity = raw["Humidity(%)"].astype(float).mask(raw["Humidity(%)"].eq(0))
    for lag in (2, 3, 5, 8, 26):
        expected = humidity.shift(lag).isna().astype(int)
        pd.testing.assert_series_equal(features[f"w1_humidity_missing{lag}"], expected, check_names=False)
    # The earliest validation week has fewer than four historical weeks.
    baseline = baselines(raw)
    assert np.isfinite(baseline.loc["2017-12-25", "weekly_mean"]).all()


def test_no_unavailable_observations(raw, features):
    """Check that later unknown observations cannot change a prediction row.

    Inputs:
        raw: Complete observations.
        features: Shared batch feature table.

    Outputs:
        None. Assertion failure if changes to unavailable counts, weather, or operating
        status affect the row, or if single-request and batch features differ.
    """
    issue = ISSUE_START + pd.Timedelta(days=4)
    target = issue + pd.Timedelta(hours=1)
    changed = raw.copy()
    future = changed.index >= issue
    changed.loc[future, TARGET] = 99999
    changed.loc[future, list(WEATHER.values())] = 1234
    changed.loc[future, "Functioning Day"] = "No"
    # Known target calendar holiday is allowed; realized target weather/status/count are
    # not.
    pd.testing.assert_frame_equal(features_at(raw, issue), features_at(changed, issue))
    pd.testing.assert_frame_equal(features.loc[[target]], features_at(raw, issue), check_names=False)


def test_validation_labels_cannot_affect_training(raw, features):
    """Check that validation answers cannot alter training data or fitted preprocessing.

    Inputs:
        raw: Complete observations.
        features: Shared batch feature table.

    Outputs:
        None. Assertion failure if validation-count changes alter any round's training
        matrix/labels, or the checked Ridge transformations and coefficients.
    """
    for train, val in validation_splits(raw):
        changed = raw.copy()
        changed.loc[val, TARGET] += 100000
        changed_features = build_features(changed)
        pd.testing.assert_frame_equal(features.loc[train], changed_features.loc[train])
        np.testing.assert_array_equal(raw.loc[train, TARGET], changed.loc[train, TARGET])
    # Training-only preprocessing and fitted parameters stay identical too.
    train, val = next(validation_splits(raw))
    changed = raw.copy()
    changed.loc[val, TARGET] += 100000
    changed_features = build_features(changed)
    cols = feature_columns(features, "W1")
    left = make_model("ridge", cols, {"alpha": 1})
    right = make_model("ridge", cols, {"alpha": 1})
    left.fit(features.loc[train, cols], raw.loc[train, TARGET])
    right.fit(changed_features.loc[train, cols], changed.loc[train, TARGET])
    np.testing.assert_array_equal(left["regressor"].coef_, right["regressor"].coef_)
    np.testing.assert_array_equal(left["prepare"].transform(features.loc[train, cols]),
                                  right["prepare"].transform(changed_features.loc[train, cols]))


def test_selection_rule():
    """Check the near-tie selection rule using small illustrative results.

    Inputs:
        No arguments. Constructs three candidate rows with known MAE and seasonal
            ratios.

    Outputs:
        None. Assertion failure if the rule selects a worse seasonal ratio among
        near-ties or admits a candidate outside the 1% limit.
    """
    table = pd.DataFrame([
        {"id": "a", "mae": 100, "worst_season_ratio": 0.8, "model_bytes": 100, "predict_ms_per_row": 1},
        {"id": "b", "mae": 100.5, "worst_season_ratio": 0.7, "model_bytes": 200, "predict_ms_per_row": 1},
        {"id": "c", "mae": 102, "worst_season_ratio": 0.1, "model_bytes": 50, "predict_ms_per_row": 1},
    ])
    assert select_candidate(table).id == "b"


def test_api_contract_and_batch_equivalence(raw, features):
    """Check the saved-model API, error responses, and batch agreement.

    Inputs:
        raw: Complete observations.
        features: Shared batch feature table. A previously trained model.joblib is
            required; TestClient starts the API inside the test process.

    Outputs:
        None. Assertion failure for incorrect predictions/status codes or use of target
        operating status to reject requests. Temporarily changes in-memory application
        state only.
    """
    from seoul_bike.api import app
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        bundle = app.state.bundle
        health = client.get("/health").json()
        assert health["evaluation_issue_start"] == bundle["metadata"]["issue_start"]
        assert health["issue_start"] == "2017-12-08T00:00:00+09:00"
        for value in (health["issue_start"], health["issue_end"],
                      "2018-10-16T17:00:00+09:00"):
            trial = client.post("/predict", json={"issued_at": value})
            assert trial.status_code == 200
        october = client.post("/predict", json={
            "issued_at": "2018-10-16T17:00:00+09:00"
        }).json()
        assert october["used_in_training"] is True
        assert october["evaluation_scope"] == "training"
        schema = client.get("/openapi.json").json()
        example = schema["components"]["schemas"]["PredictionRequest"]["properties"]["issued_at"]["examples"][0]
        assert client.post("/predict", json={"issued_at": example}).status_code == 200
        for issue in (ISSUE_START, ISSUE_START + pd.Timedelta(days=4),
                      pd.Timestamp(bundle["metadata"]["issue_end"])):
            response = client.post("/predict", json={"issued_at": issue.isoformat()})
            assert response.status_code == 200
            target = issue + pd.Timedelta(hours=1)
            expected = max(0, bundle["model"].predict(features.loc[[target], bundle["columns"]])[0])
            assert response.json()["predicted_rentals"] == pytest.approx(expected)
            actual = int(raw.loc[target, "Rented Bike Count"])
            assert response.json()["actual_rentals"] == actual
            assert response.json()["absolute_error"] == pytest.approx(abs(expected - actual))
            if actual:
                assert response.json()["absolute_percentage_error"] == pytest.approx(
                    100 * abs(expected - actual) / actual
                )
        for value in ("2018-11-16T12:00:00", "2018-11-16T12:01:00+09:00",
                      "2017-12-07T23:00:00+09:00", "2018-12-01T00:00:00+09:00"):
            assert client.post("/predict", json={"issued_at": value}).status_code == 422
        assert client.post("/predict", json={"issued_at": ISSUE_START.isoformat(), "weather": {}}).status_code == 422
        original = app.state.raw
        altered = original.copy()
        altered.loc[ISSUE_START + pd.Timedelta(hours=1), "Functioning Day"] = "No"
        app.state.raw = altered
        assert client.post("/predict", json={"issued_at": ISSUE_START.isoformat()}).status_code == 200
        altered.loc[ISSUE_START + pd.Timedelta(hours=1), "Rented Bike Count"] = 0
        comparison = client.post("/predict", json={"issued_at": ISSUE_START.isoformat()}).json()
        assert comparison["actual_rentals"] == 0
        assert comparison["absolute_percentage_error"] is None
        assert comparison["actual_operating"] is False
        assert comparison["predicted_rentals"] == pytest.approx(
            max(0, bundle["model"].predict(features.loc[
                [ISSUE_START + pd.Timedelta(hours=1)], bundle["columns"]
            ])[0])
        )
        app.state.raw = original
        app.state.bundle = None
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json={"issued_at": ISSUE_START.isoformat()}).status_code == 503


def test_feature_group_importance():
    """Identify a strong synthetic signal without changing the input table.

    Inputs:
        No arguments. Creates rental-related signal and unrelated weather noise,
            with one training/validation split and three deterministic shuffles.

    Outputs:
        None. Assertion failure if the signal does not outrank noise, results are
        nonfinite, or the diagnostic changes the original input table.
    """
    from seoul_bike.modeling import feature_group_importance

    rng = np.random.default_rng(12)
    index = pd.date_range("2018-01-01", periods=240, freq="h", tz="Asia/Seoul")
    signal = rng.uniform(0, 100, len(index))
    inputs = pd.DataFrame({
        "count_lag2": signal, "w1_temp_lag2": rng.normal(size=len(index)),
    }, index=index)
    original = inputs.copy(deep=True)
    observed = pd.DataFrame({TARGET: 10 * signal + 100}, index=index)
    train = np.arange(len(index)) < 180
    with threadpool_limits(limits=2):
        result = feature_group_importance(
            observed, inputs, [(train, ~train)], "boosting", "W1",
            {"max_leaf_nodes": 7, "l2_regularization": 1.0}, repeats=3,
        )
    pd.testing.assert_frame_equal(inputs, original)
    by_group = result.set_index("group")
    assert np.isfinite(result[["mae_increase", "shuffle_std"]]).all().all()
    assert by_group.loc["Rental history", "mae_increase"] > 100
    assert (by_group.loc["Rental history", "mae_increase"]
            > by_group.loc["Weather history", "mae_increase"])

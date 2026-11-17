"""Build rental-prediction inputs from known dates and completed observations."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data/raw/SeoulBikeData.csv"
SOURCE_SHA256 = "373339b71a8935d69e9af0abf26a70744632119862eeb3919efb389a7b749c60"
TZ = "Asia/Seoul"
TARGET = "Rented Bike Count"
WEATHER = {
    "temp": "Temperature(°C)", "humidity": "Humidity(%)",
    "wind": "Wind speed (m/s)", "visibility": "Visibility (10m)",
    "dew": "Dew point temperature(°C)", "solar": "Solar Radiation (MJ/m2)",
    "rain": "Rainfall(mm)", "snow": "Snowfall (cm)",
}
WEEKS = (
    ("2017-12-25", "2018-03-05", "2018-06-04", "2018-09-17"),
    ("2018-01-08", "2018-04-09", "2018-07-09", "2018-10-01"),
    ("2018-02-19", "2018-05-21", "2018-08-20", "2018-11-05"),
)
DEV_END = pd.Timestamp("2018-11-14 22:00", tz=TZ)
TEST_START = pd.Timestamp("2018-11-15", tz=TZ)
TEST_END = pd.Timestamp("2018-12-01", tz=TZ)
ISSUE_START = TEST_START - pd.Timedelta(hours=1)
ISSUE_END = TEST_END - pd.Timedelta(hours=2)


def load_data(path=RAW_PATH):
    """Read the fixed CSV as a table indexed by the start of each observed hour.

    Inputs:
        path: CSV file path; defaults to RAW_PATH. LF and CRLF are both supported.

    Outputs:
        A table (pandas DataFrame) with 8,760 rows, 13 columns, and a Seoul time index.
        Date and Hour together form the index; Date is then removed as a column. The CSV
        still has 14 columns and is not changed. Raises ValueError for an invalid schema
        or incomplete hourly sequence.
    """
    path = Path(path)
    frame = pd.read_csv(path, encoding="latin-1")
    dates = pd.to_datetime(frame.pop("Date"), format="%d/%m/%Y")
    source_hour_starts = dates + pd.to_timedelta(frame["Hour"], unit="h")
    frame.index = pd.DatetimeIndex(source_hour_starts).tz_localize(TZ)
    frame.index.name = "target_start"
    expected = pd.date_range(
        "2017-12-01", periods=8760, freq="h", tz=TZ, name=frame.index.name
    )
    if not frame.index.equals(expected) or frame.shape[1] != 13:
        raise ValueError("Unexpected hourly axis or CSV schema.")
    return frame


def build_features(frame):
    """Build one row of model inputs for each target hour from known information.

    Inputs:
        frame: Hourly rental, weather, holiday, and operating-status observations. Each
            index timestamp marks the start of the hour recorded in that source row.

    Outputs:
        A new feature table whose row timestamp S is the hour to predict. Rental and
        weather inputs use only source timestamps at or before S minus two hours: target
        18:00 can use the 16:00-17:00 record and earlier records. The target
        date/holiday are known in advance. Early targets may lack history; frame is
        unchanged.
    """
    idx = frame.index
    hour, weekday = idx.hour, idx.day_of_week
    # Dates are known in advance; realized target weather and counts are not.
    values = {
        "hour": hour, "weekday": weekday, "weekend": (weekday >= 5).astype(int),
        "hour_kind": [f"{h:02d}_{int(d >= 5)}" for h, d in zip(hour, weekday)],
        "holiday": frame["Holiday"].eq("Holiday").astype(int),
        "month_sin": np.sin(2 * np.pi * (idx.month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (idx.month - 1) / 12),
    }
    # For target 18:00, the latest usable record covers 16:00-17:00.
    count = frame[TARGET]
    # lag counts hours before the TARGET, not before the latest observation.
    for lag in (2, 3, 4, 24, 48, 168):
        values[f"count_lag{lag}"] = count.shift(lag)
    for window in (3, 24, 168):
        completed_counts = count.shift(2)
        recent_counts = completed_counts.rolling(window, min_periods=window)
        values[f"count_mean{window}"] = recent_counts.mean()
    active = frame["Functioning Day"].eq("Yes").astype(float)
    for lag in (2, 24, 168):
        values[f"active_lag{lag}"] = active.shift(lag)
    for key, column in WEATHER.items():
        source = frame[column].astype(float)
        if key == "humidity":
            source = source.mask(source.eq(0))
        for lag in (2, 3, 5, 8, 26):
            values[f"w1_{key}_lag{lag}"] = source.shift(lag)
            if key == "humidity":
                values[f"w1_{key}_missing{lag}"] = source.shift(lag).isna().astype(int)
        # delta compares past weather with the LATEST usable weather value.
        # For target 18:00: lag3 reads 15:00; delta3 is 16:00 minus 13:00.
        for delta in (1, 3, 6):
            values[f"w2_{key}_delta{delta}"] = source.shift(2) - source.shift(2 + delta)
        # Use consecutive recent hours, not matching hours on previous days.
        for window in (3, 6, 24, 48):
            roll = source.shift(2).rolling(window, min_periods=1)
            values[f"w3_{key}_mean{window}"] = roll.mean()
            values[f"w3_{key}_range{window}"] = roll.max() - roll.min()
            values[f"w3_{key}_valid{window}"] = roll.count()
            if key in ("rain", "snow"):
                positive = source.gt(0).astype(float).mask(source.isna())
                positive_hours = positive.shift(2).rolling(window, min_periods=1)
                values[f"w3_{key}_positive{window}"] = positive_hours.sum()
    return pd.DataFrame(values, index=idx)


def feature_columns(features, group):
    """Select common inputs plus the requested set of historical weather inputs.

    Inputs:
        features: The table produced by build_features.
        group: W1 keeps past weather values; W2 also keeps changes; W3 also keeps
            statistics over consecutive recent hours.

    Outputs:
        A list of input-column names. Every selection includes dates, rental history,
        and past operating status; the weather groups build on one another.
    """
    excluded_prefixes = {"W1": ("w2_", "w3_"), "W2": ("w3_",), "W3": ()}
    return [c for c in features if not c.startswith(excluded_prefixes[group])]


def sample_masks(frame):
    """Choose development and final-test target hours without deleting history.

    Inputs:
        frame: Complete hourly observations, including rows marked as non-operating.

    Outputs:
        Two True/False arrays in source-row order: development, then final test. True
        selects a normal-operation target with enough history in that date range. The
        168-hour rental mean reads S-169 through S-2, so the first 169 targets lack that
        history. False does not remove a source row; final-test membership is given by
        the second array.
    """
    # 168 consecutive records ending at S-2 begin at S-169.
    valid = np.arange(len(frame)) >= 169
    valid &= frame["Functioning Day"].eq("Yes").to_numpy()
    dev = valid & (frame.index < DEV_END)
    test = valid & (frame.index >= TEST_START) & (frame.index < TEST_END)
    return dev, test


def validation_splits(frame):
    """Choose training and validation target hours for each of three comparisons.

    Inputs:
        frame: Complete hourly observations. WEEKS specifies four validation weeks per
            round within the development data.

    Outputs:
        An iterator of (training_mask, validation_mask) pairs, each in source-row order.
        A training row is excluded if its past-count inputs contain a validation answer:
        a validation Sunday 23:00 count would enter Monday 01:00's two-hours-earlier
        input. Also exclude validation targets themselves and the 24 hours before each
        validation week. Final-test targets are outside both masks; other excluded rows
        can also be False in both.
    """
    development_mask, _ = sample_masks(frame)
    for week_starts in WEEKS:
        validation_mask = np.zeros(len(frame), dtype=bool)
        before_validation_week = validation_mask.copy()
        for week_start in week_starts:
            week_start = pd.Timestamp(week_start, tz=TZ)
            week_end = week_start + pd.Timedelta(days=7)
            validation_mask |= (frame.index >= week_start) & (frame.index < week_end)
            guard_start = week_start - pd.Timedelta(hours=24)
            before_validation_week |= (
                (frame.index >= guard_start) & (frame.index < week_start)
            )
        validation_mask &= development_mask

        # A training Monday 01:00 row can read a validation Sunday 23:00 count.
        # Such training rows must be excluded even after the validation week ends.
        validation_count_prefix = np.r_[0, np.cumsum(validation_mask)]
        uses_validation_counts = np.zeros(len(frame), dtype=bool)
        target_positions = np.arange(169, len(frame))
        # Every rental input for row h is drawn from rows h-169 through h-2.
        history_start = target_positions - 169
        history_stop = target_positions - 1  # Exclusive end includes row h-2.
        validation_counts_in_history = (
            validation_count_prefix[history_stop]
            - validation_count_prefix[history_start]
        )
        uses_validation_counts[target_positions] = validation_counts_in_history > 0
        training_mask = (
            development_mask
            & ~validation_mask
            & ~before_validation_week
            & ~uses_validation_counts
        )
        yield training_mask, validation_mask


def baselines(frame):
    """Predict a target using rental counts from previous matching hours.

    Inputs:
        frame: Complete hourly observations containing TARGET; retain shutdown zeros.

    Outputs:
        A table with yesterday's same hour, last week's same hour, and weekly_mean. For
        Monday 18:00, weekly_mean averages available Monday 18:00 records from the
        previous one to four weeks. This differs from averaging all consecutive hours in
        a week. No model is trained.
    """
    # For target 18:00, the latest usable record covers 16:00-17:00.
    count = frame[TARGET]
    weekly = pd.concat([count.shift(168 * n) for n in range(1, 5)], axis=1)
    return pd.DataFrame({
        "yesterday": count.shift(24), "last_week": count.shift(168),
        "weekly_mean": weekly.mean(axis=1),
    }, index=frame.index)


def features_at(frame, issued_at):
    """Build one target's inputs using observations completed by issued_at.

    Inputs:
        frame: Complete hourly observations; target Holiday is known in advance.
        issued_at: A timezone-aware exact historical issue hour, not the HTTP arrival
            time. The target starts one hour later.

    Outputs:
        A one-row input table built from 168 completed hourly records and the target
        date/holiday. For issued_at 17:00, target 18:00 uses source hours through
        16:00-17:00. Raises ValueError for invalid time or incomplete history. The API
        separately checks the accepted date range.
    """
    issue = pd.Timestamp(issued_at)
    if issue.tzinfo is None:
        raise ValueError("issued_at must include a timezone.")
    issue = issue.tz_convert(TZ)
    if issue != issue.floor("h"):
        raise ValueError("issued_at must be an exact hour.")
    target = issue + pd.Timedelta(hours=1)
    history_index = pd.date_range(target - pd.Timedelta(hours=169),
                                  target - pd.Timedelta(hours=2), freq="h")
    if not history_index.isin(frame.index).all():
        raise ValueError("168 consecutive completed hourly records are required.")
    # Add empty rows for issue hour and target hour so the common calculations
    # can produce row S. They contain no future rental/weather observations.
    context_index = pd.date_range(history_index[0], target, freq="h")
    context = frame.loc[history_index].reindex(context_index)
    if target in frame.index:
        target_holiday = frame.loc[target, "Holiday"]
    else:
        target_holiday = "No Holiday"
    context.loc[target, "Holiday"] = target_holiday
    return build_features(context).loc[[target]]

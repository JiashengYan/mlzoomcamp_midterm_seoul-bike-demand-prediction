# Implementation: understand the workflow before the details

**The root development notebook leads analysis, model selection, and evaluation, using shared data.py and modeling.py functions.** This reference defines its rules. train.py is a secondary batch entry for full retraining and saved-artifact export.

Updated: 2026-11-16. Existing experiment results are unchanged by the documentation revision. Start with [README](../README.md) for commands or [Project brief](PROJECT_BRIEF.md) for scope.

## 1. Main sequence and vocabulary

**A sample means one target hour and the information allowed for predicting it.** A feature is one model-input column; a target or label is the actual rental count being predicted. A validation period helps choose a model; the final test is reserved for evaluating the chosen model.

~~~mermaid
flowchart TD
    A["Read every source hour, including shutdowns"] --> B["Build inputs from earlier records"]
    B --> C["Assign target roles"]
    C --> D["Compare three model families"]
    D --> E["Tune the two strongest families"]
    E --> F["Choose configuration using validation"]
    F --> G["Fit all development targets"]
    G --> H["Save model; evaluate later test"]
    H --> I["Serve requests with the same input logic"]
~~~

[model_development.ipynb](../model_development.ipynb) records the main sequence interactively; batch retraining repeats the predefined protocol. All formal model candidates use dates, rental history, operating history, and weather history. Historical rules provide reference scores for judging the model's improvement.

## 2. Time available for one prediction

**Issue at T and predict the hour starting at S=T+1 hour.** Assume source rows start their hour and become available when it ends; both rental and weather delays are fixed to zero in version 1.

| Item | Definition |
| --- | --- |
| Target interval | [S, S+1 hour) |
| Issue time | T=S−1 hour |
| Latest completed observation | [S−2 hours, S−1 hour) |
| Example | Issue 17:00; target 18:00–19:00; latest row 16:00–17:00 |
| Time zone | Asia/Seoul; requests must include an offset/time zone |

These are explicit research assumptions, not confirmed publication semantics. The known target date and holiday may be used. Target actual weather, rental count, and operating status are not prediction inputs.

The history boundary is a time rule: a target beginning at S may use observed records whose starting timestamp is no later than S−2 hours. For target 18:00, the latest record starts at 16:00 and finishes at 17:00; target date and holiday remain known in advance.

The code implements this rule with `shift(2)`: at the 18:00 feature row, read the value originally recorded on the 16:00 source row. The target index stays at 18:00. Because load_data verifies one source row per hour, two rows correspond to two hours. Keep shutdown rows so this relationship remains correct.

## 3. Inputs used by the final model and compared alternatives

**The final model uses target-time information, rental history, past operating status, and W2 weather inputs: previous values plus observed changes.** W1, W2, and W3 were compared on the same validation targets. W3 adds weather means and other recent-hour statistics as an alternative candidate.

| Shared input | Exact implementation |
| --- | --- |
| Target time | Hour, weekday, weekend, holiday, and month sine/cosine |
| Hour/day-type category | `hour_kind`, e.g. `08_0` for weekday 08:00 and `08_1` for weekend 08:00 |
| Past rental values | Target-relative lags 2, 3, 4, 24, 48, 168 hours |
| Recent rental means | Consecutive 3, 24, 168 hours ending at source S−2 |
| Past operating status | Lags 2, 24, 168 hours |

Month sine/cosine uses two numeric coordinates to place each month around a circle, so December and January remain adjacent. The 48-hour weather mean belongs to the W3 candidate; month coordinates are common inputs in every candidate.

### 3.1 W1: values at specified past hours

**W1 supplies eight weather measurements at target-relative lags 2, 3, 5, 8, and 26 hours.** With target 18:00, these rows start at 16:00, 15:00, 13:00, 10:00, and 16:00 the previous day.

The measurements are temperature, humidity, wind, visibility, dew point, solar radiation, rainfall, and Snowfall. Humidity zero is marked missing only in features, and its lagged missing-value indicators are retained. Other zero values are not automatically missing.

### 3.2 W2: add changes between past hours

**W2 keeps W1 and adds latest-known minus earlier weather values, using gaps of 1, 3, and 6 hours.** For a target at 18:00, the three-hour temperature change is the 16:00 value minus the 13:00 value.

Names use two different reference points: `lag3` means three hours before target S, while `delta3` compares the latest usable observation with the observation three hours before that. For target 18:00, lag3 reads 15:00, whereas delta3 subtracts 13:00 from 16:00. This describes an already observed change available at the issue time.

### 3.3 W3: add statistics over consecutive recent hours

**W3 keeps W2 and adds mean, maximum-minus-minimum, and valid-record count over 3, 6, 24, and 48 consecutive hours.** Rainfall and Snowfall also have counts of positive records in each period.

For issue 17:00 and target 18:00, the six-hour mean uses the six source rows starting at 11:00–16:00. For the next issue, it uses 12:00–17:00. This is a rolling-window calculation: the selected block of consecutive records moves forward with time. The observations in each calculation are consecutive completed hours.

Timestamps must be complete. Weather statistics ignore marked missing values, record the number of valid values, and leave mean/range missing when no valid value exists. Rental means require the full number of observations. Rain/snow values are used as recorded numerical measurements. The measurement-interpretation limits are documented in [Data analysis](DATA_ANALYSIS.md).

### 3.4 Missing-value filling and model-specific encoding

**Learn every filling and encoding step from the selected training rows, then apply it unchanged to validation, test, or requests.** For each missing input, learn a replacement median from training rows only. Ridge also learns each numeric column's training mean and scale, and represents categories with separate 0/1 columns (one-hot encoding), including the hour/weekend category. Trees use numeric hour/weekday/weekend and omit the string hour_kind. These encodings express the same available information.

## 4. Select development and final-test targets

**Use 7,910 development targets to choose and fit the model; reserve 384 later targets for final evaluation.** The remaining hours are excluded as targets for the reasons below. All source rows remain on the complete hourly timeline.

| Target category | Hours | Reason or handling |
| --- | ---: | --- |
| Insufficient early history | 169 | Exclude as targets because the full 168-record rental history is unavailable; retain source observations |
| Non-operating hours | 295 | Exclude shutdown zeros from training and evaluation labels; the prediction goal is normal-operation demand |
| Development targets | 7,910 | Choose a configuration and train the final model |
| Training/test boundary | 2 | Enforce training-label availability before the first test issue and reserve one hour for fitting |
| Final-test targets | 384 | Evaluate the selected model on November 15–30, 2018 |

Early history consists of the first 169 hourly records. The boundary hours start at 22:00 and 23:00 on November 14, 2018.

**Exclude shutdown hours as target labels.** Non-operating rows have Functioning Day=No. Their zeros are observed shutdown counts. Training, validation, and test target selectors exclude these hours because the project predicts normal-operation demand. 

**Historical count features currently use the recorded shutdown zeros.** This keeps historical calculations based on recorded transactions, with operating-status inputs at S−2, S−24, and S−168. Shutdown zeros lower the historical means and can therefore affect later normal-operation predictions. Keeping a complete timeline and deciding whether zeros contribute to an average are separate choices. Treating shutdown counts as missing would require a compared alternative; the current experiment has not established which treatment is better.

There are 8,465 operating hours in total and 8,296 with the complete history required for prediction. Development targets start at 2017-12-08 01:00 and are strictly before 2018-11-14 22:00. Final-test targets cover [2018-11-15 00:00, 2018-12-01 00:00).

`sample_masks()` returns two arrays in source-row order: `development_mask` and `test_mask`. True marks membership in the named role; False marks another role. Each array contains 8,760 entries. The saved report's role column uses development, test, and history_only; history_only groups early-history, non-operating, and boundary rows.

### 4.1 Respect label availability at the final-test boundary

**The two boundary hours enforce the order: training labels become available, the model is fitted, then the first test prediction is issued.** The final training label covers November 14 21:00–22:00 and is available at 22:00. The first test prediction is issued at 23:00 for November 15 00:00–01:00, leaving one hour for fitting. The 22:00 and 23:00 source rows are excluded as training targets to respect this schedule.

At the first issue, the 22:00–23:00 record is completed history; the 23:00–00:00 record becomes available at 00:00. Each prediction's observation cutoff separately prevents unavailable rental or weather values from entering its inputs.

## 5. Choose models using three seasonal validation rounds

**Three rounds sample all 12 observed months while limiting each candidate to three validation fits.** Each round holds out four weeks, one per season; the rounds use each season's first, middle, and last month respectively. With four seasonal weeks per round, three rounds suffice to cover every month.

The purpose is representative validation distributed across the months of the observed year. Rotating weeks reduces reliance on one set of held-out weeks while retaining most development targets for training in each round. Three is a coverage/computation tradeoff: additional rounds require additional fits. These are repeated evaluation fits of the same candidate design; the final model is fitted once on all development targets.

For example, a model validating a June week may train on July–October records. That result measures a model trained across the observed year's different periods.

Weeks were originally chosen using calendar information and seed 42; runtime uses the fixed dates below and never resamples based on outcomes. Each week is Monday 00:00 through the following Monday 00:00, stays within its month, and is disjoint from other validation weeks. Rounds use the first/middle/last month of each season respectively.

| Round | Winter | Spring | Summer | Autumn |
| --- | --- | --- | --- | --- |
| 1 | 2017-12-25 | 2018-03-05 | 2018-06-04 | 2018-09-17 |
| 2 | 2018-01-08 | 2018-04-09 | 2018-07-09 | 2018-10-01 |
| 3 | 2018-02-19 | 2018-05-21 | 2018-08-20 | 2018-11-05 |

| Round | Training targets | Validation targets | Temporarily excluded |
| --- | ---: | ---: | ---: |
| 1 | 6,562 | 624 | 724 |
| 2 | 6,593 | 593 | 724 |
| 3 | 6,613 | 624 | 673 |

**Validation reduces the training targets available within each comparison fit, while final fitting uses all 7,910 development targets.** A round uses 6,562/6,593/6,613 training targets after holding out its validation weeks and applying the exclusion rules. Across the three rounds, validation contains 1,841 targets from 12 selected weeks spread across 12 months. A target held out in one round can train in other rounds, and every development target trains in at least one. The 384 final-test targets remain reserved for evaluation. The actual target-role list is [splits.csv](../reports/splits.csv).

### 5.1 Assign training, validation, and excluded roles

**Each of the 7,910 development targets receives one of three roles for that round.** `validation_splits()` constructs the training and validation arrays from these roles.

| Per-round role | Selection rule | training_mask | validation_mask |
| --- | --- | --- | --- |
| Training | Eligible development targets remaining after the exclusions below | True | False |
| Validation | Operating development targets in the four selected weeks | False | True |
| Temporarily excluded | Development targets outside validation weeks whose history contains validation counts, or that fall in the preceding 24 hours | False | False |

Early-history, non-operating, boundary, and final-test rows retain their roles throughout the three comparison rounds. Every source row remains available to provide observations once its hour has completed. The per-sample time cutoff and per-round training exclusions determine which values each prediction or fit may read.

### 5.2 Keep validation rental answers out of training

**Exclude a training target if its rental-history inputs contain any of this round's validation rental answers.** Exclusion assigns a target to the temporarily-excluded role for that round. Its source observation remains available as history under the time-availability rules.

Select normal-operation validation targets; exclude those targets from fitting; exclude training rows reading their counts; and exclude the 24 hours before each validation week. Fit missing-value filling, scaling, and model parameters only on the remaining training rows.

Example: a validation week ends on Sunday at midnight. Its Sunday 23:00 actual count would become the two-hours-earlier count for a Monday 01:00 training target. Exclude that Monday target even though it lies outside the validation week.

Implementation detail: h is an integer row position, while S elsewhere is a timestamp. Every rental input for h reads rows h−169 through h−2. Count validation answers in that interval using cumulative counts; a positive count excludes h. This cumulative-count calculation efficiently applies the preceding selection rule.

**The extra one-day gap is a conservative engineering choice.** Before each validation week, exclude 24 hours of immediately preceding training targets. One full day gives a simple, reproducible gap in hourly data; this exclusion is separate from removing training inputs that contain validation answers. The experiment did not compare gap lengths.

Earlier validation counts become known history for later validation predictions after their hours finish. Training uses the selected training rows, and source weather remains available under each sample's time cutoff. Adjacent observations can remain statistically related; generalization is assessed separately on the later final test.

## 6. Compare and choose models

**Start with three algorithms and three weather groups on identical rounds, then tune only the two strongest families.** Each candidate's three validation predictions are combined into one comparison result.

| Family | Base configuration |
| --- | --- |
| Ridge | alpha=1 |
| Random forest | 200 trees, maximum depth 16, minimum leaf size 10 |
| HistGradientBoostingRegressor | 150 iterations, maximum 15 leaves, minimum leaf size 25, learning rate 0.08, L2=10; internal random early stopping disabled |

Seed 42 is used where applicable; numerical threads and forest jobs are limited to two. After the 9 initial candidates, tune at the best weather group for each of the top two families, skipping its existing base configuration. At most 8 additional settings per family are tried.

The fixed choices are Ridge alpha {0.01,0.1,1,10,100,1000}; forest depth {10,16} × minimum leaf {3,10,25}; boosting leaves {7,15,31} × L2 {1,10,30}. The actual run compared 22 configurations, with 66 validation fits and one final refit. The [executable notebook](../model_development.ipynb) calls evaluate() for the initial comparison and tuning, select_candidate() to freeze the choice, and make_model() for final fitting. It computes its tables and charts from those in-memory results.

Notebook section 6 compares each tuned family's initial best MAE with its best evaluated MAE and plots the evaluated parameter combinations. The search keeps that family's best initial weather group fixed, so the parameter comparison uses the same inputs and validation targets. This is a limited predefined search, with tuning effort concentrated on the two strongest initial families.

Three rules provide baselines: yesterday's same hour, last week's same hour, and the mean of available matching hours from 1–4 previous weeks. At least one matching historical value suffices; shutdown zeros remain genuine records.

### 6.1 Scores and selection

**Minimize combined validation MAE, the average absolute error in rentals/hour.** Each of the 1,841 validation hours has one contribution; months with fewer operating hours are not upweighted.

Also report RMSE (which emphasizes larger errors), mean prediction-minus-actual bias, per-round/season/hour/day-type/holiday/rain/snow results, and hours/date coverage. The high-count diagnostic groups target hours exceeding that training set's 90th-percentile count.

Candidates within 1% of the minimum MAE are engineering near-ties. Among them, minimize the worst seasonal ratio of model MAE to last-week baseline MAE, then saved-model bytes, then measured batch prediction time per row. For example, if minimum MAE is 100, an MAE of 100.8 qualifies while 102 does not. A seasonal ratio of 0.7 means model error is 70% of the last-week rule's error in that season; choose the candidate with the best worst-season ratio. The 1% band is a fixed engineering selection rule. Statistical equivalence and operational tolerances require their own criteria.

Target actual rain/snow is used only to group errors after prediction. Final-test results never choose features or hyperparameters.

### 6.2 Inspect which input groups the selected model relies on

**Notebook section 7.1 measures the validation error increase after scrambling one group of inputs.** The four groups are known target time, rental history, weather history, and past operating status. Each group's related columns use the same shuffled row order within each validation week, retaining that week's seasonal context.

The measured MAE increases are about 288.57 rentals/hour for rental history, 75.68 for weather history, 59.54 for known target time, and 0.00 for past operating status. The last value indicates no measurable reliance in this diagnostic; it does not establish that operating status is irrelevant in other data.

The helper feature_group_importance() refits the already selected settings on the three existing training subsets. It compares unchanged validation predictions with five seeded shuffles per group and averages error increases in proportion to validation-hour counts. Error bars show shuffle variability, not confidence intervals. The final test and saved service model are untouched.

A larger increase indicates stronger reliance by this fitted model. Correlated groups can substitute for one another, and scrambled combinations can be unusual, so this diagnostic does not establish causal effects or the value of removing inputs and retraining. It is explanatory; feature selection and tuning remain frozen. See the [permutation method and correlation limits](https://scikit-learn.org/stable/modules/permutation_importance.html). The notebook saves [feature_importance.png](../reports/figures/feature_importance.png).

## 7. Read the measured results

**Gradient boosting + W2 achieved final-test MAE 85.2 rentals/hour, about 33.5% below the matching-hour weekly-mean baseline.** This result covers the 384 target hours on November 15–30, 2018. Development validation selected the configuration before this final test was evaluated.

| Family's best evaluated setting | Validation MAE |
| --- | ---: |
| Ridge | 170.5 |
| Random forest | 135.2 |
| Gradient boosting | 121.3 |

With W2 fixed, parameter tuning reduced gradient-boosting validation MAE from 126.26 to 121.32 (3.91%) and random-forest MAE from 139.23 to 135.20 (2.89%). Notebook section 6 shows both the [before/after comparison](../reports/figures/tuning_comparison.png) and the [evaluated parameter combinations](../reports/figures/tuning_parameters.png).

The selected model uses 150 iterations, 31 leaves, minimum leaf 25, learning rate 0.08, and L2=30. W3 did not improve the initial comparison. Final-test RMSE is 129.7 and bias −20.1 rentals/hour. The matching-hour weekly-mean baseline has MAE 128.2. See [all candidates](../reports/validation.csv) and [detailed scores](../reports/metrics.json). Validation and final-test scores cover different hours; compare models or baselines within the same evaluation period. Assessing weather causality or next-year reliability requires additional evidence.

### 7.1 Relative errors and reference value

**The model improves on simple matching-hour reuse for most normal-operation hours in this test.** It has lower absolute error than the matching-hour weekly-mean rule on 239/384 hours (62.2%); 77.6% of predictions are within 20% of actual rentals, versus 61.5% for the rule. Mean absolute count error falls from 128.19 to 85.25 rentals/hour, a 33.5% reduction.

The model jointly uses known dates, recent rental activity, and completed weather history. The measured baselines reuse yesterday's count, last week's count, or a mean of the previous one to four matching weekday/hour counts. A weather-only matching comparison requires its own specified rule and evaluation under the same observation cutoff.

For each positive actual count, calculate absolute error / actual count. Averaging those percentages gives MAPE: 21.33% for the model and 35.78% for the weekly-mean rule. The share within 20% counts hours whose ratio is at most 0.2. These summaries use all 384 test hours, each with a positive actual count; 20% is a descriptive band, while model selection continues to use validation MAE.

**Volume groups reveal where the reference is weaker.** These are post-prediction diagnostics grouped by actual rentals, separate from model inputs or output classes.

| Actual rentals/hour | Hours | Mean actual | Model MAE | Weekly-mean MAE | Model closer to actual |
| --- | ---: | ---: | ---: | ---: | ---: |
| Below 100 | 13 | 73.62 | 95.43 | 140.38 | 30.8% |
| 100–499 | 121 | 284.07 | 49.86 | 94.51 | 61.2% |
| 500–999 | 186 | 747.28 | 76.19 | 129.21 | 65.6% |
| 1,000 and above | 64 | 1,325.73 | 176.39 | 186.43 | 60.9% |

The low-volume group has a mean absolute error exceeding its mean actual count, and the model is closer on only four of those thirteen hours. MAPE is sensitive to small actual counts, as described in the [metric documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html). The stronger overall reference value and this limitation are both part of the result.

Notebook section 8.1 computes the comparison table, hour-by-hour wins, volume groups, and [error-coverage curve](../reports/figures/performance.png) directly from its final-test predictions. The curve shows thresholds up to 100%, with every test hour retained in the denominator. Evidence covers the 16-day November test period; operational benefits and performance in other periods require separate evaluation.

### 7.2 Reports for detailed checks

**The notebook presents the reviewable experiment; batch training exports the standard evidence files below.** Use these files to inspect exact settings, selected target hours, and individual predictions.

| Supporting file | What to inspect |
| --- | --- |
| [metrics.json](../reports/metrics.json) | Model settings, sample counts, overall and grouped errors |
| [validation.csv](../reports/validation.csv) | All compared configurations and the selected candidate |
| [splits.csv](../reports/splits.csv) | Each target's development/test and per-round role |
| [validation_predictions.csv](../reports/validation_predictions.csv) | Selected candidate's validation predictions |
| [test_predictions.csv](../reports/test_predictions.csv) | Final-test predictions, observations, and rule predictions |
| [Prediction plot](../reports/figures/replay.png) | Actual rentals, predicted rentals, and their difference |

## 8. Serve historical predictions with a fixed model

**The API reads completed history for each supplied issue time and applies the same saved model throughout the replay period.**

The final model trains on all development targets and stays fixed during test/service replay. Earlier test observations become historical inputs for later predictions after their hours finish. Every test issue therefore uses its own completed history. Metadata distinguishes last-label availability, simulated model availability, and actual artifact creation time.

The API accepts exact timezone-aware issue hours from 2018-11-14 23:00 through 2018-11-30 22:00 Seoul time. It returns count, units, target interval, model ID, cutoffs, and the normal-operation assumption. Invalid requests/history yield 422, missing model 503. All responses use the normal-operation assumption. Commands and Docker instructions are in [README](../README.md).

### 8.1 Request, response, and errors

**POST /predict accepts a historical issue hour and returns the next target interval with its prediction and timing metadata.** The service obtains historical weather and rental counts from the fixed local CSV.

~~~json
{"issued_at": "2018-11-16T17:00:00+09:00"}
~~~

| Response time field | Meaning for an issue at 17:00 |
| --- | --- |
| `history_available_through` | Observations completed through 17:00 |
| `latest_source_hour_start` | Latest record starts at 16:00 and covers 16:00–17:00 |
| `training_label_available_through` | End of the final hour whose actual count was used to train the saved model |
| `simulated_model_available_at` | Earliest historical issue allowed for the saved model; distinct from its actual file creation time |

| Endpoint or condition | Behavior |
| --- | --- |
| `GET /health` | Returns readiness and model ID |
| `/docs` | Interactive API reference |
| Accepted issue times | 2018-11-14 23:00 through 2018-11-30 22:00, Asia/Seoul |
| Invalid time, extra fields, or insufficient history | HTTP 422 with a reason |
| No saved model | HTTP 503; train before prediction |

The example client defaults to localhost port 8000. If a Docker container uses host port 8001, map 127.0.0.1:8001:8000 and call the example with `--url http://127.0.0.1:8001`.

## 9. Implementation reference

**Use this section to locate the modules called by the core notebook.** Data and model calculations are shared with the secondary batch entry; feature tables are computed in memory.

### 9.1 Function responsibilities

**The notebook owns the visible development sequence; data.py supplies inputs, modeling.py supplies fitting and scoring, and train.py runs the batch workflow.** api.py serves the saved artifact.

| Function/file | Purpose | Main input → output |
| --- | --- | --- |
| `load_data` in data.py | Check CSV and hourly time axis | CSV path → hourly observation table |
| `build_features` | Build inputs from dates and past records | Hourly observations → feature table |
| `feature_columns` | Select W1/W2/W3 inputs | Feature table and group → column names |
| `sample_masks` | Select development/test targets | Observations → Boolean row selectors |
| `validation_splits` | Define three comparison rounds | Observations → train/validation selectors |
| `baselines` | Form simple historical predictions | Counts → three rule-prediction columns |
| `features_at` | Form one request's feature row | Observations and issue time → one row |
| modeling.py | Reusable fitting, evaluation, parameter grids, and selection | Features and settings → models, predictions, and scores |
| `feature_group_importance` | Measure reliance on four input groups using validation | Selected settings and splits → MAE increases |
| train.py | Secondary full batch rerun and export | Observations → saved model and standard reports |
| api.py | Validate requests and apply saved model | Issue time → prediction and metadata |

A Boolean selector, also called a mask, is a list of True/False values indicating which rows to use; it does not remove rows from the source file.

### 9.2 Build a single request with the shared feature function

**The API first isolates available observations and then calls the same build_features function.** The function receives completed observations and the target's known schedule values.

~~~mermaid
flowchart LR
    A["Check timezone and exact issue hour"] --> B["Set S = issue + 1 hour"]
    B --> C["Take rows S−169 through S−2"]
    C --> D["Add blank rows through S; fill known target holiday"]
    D --> E["build_features; keep only row S"]
~~~

The longest rental mean needs 168 consecutive records. With the newest at S−2, the oldest is S−169. The first 169 source rows provide the initial history for subsequent eligible targets.

For target 18:00, the actual history ends with the row starting at 16:00. Add empty 17:00 and 18:00 rows only so the shared function can calculate the input row labelled 18:00. Only the known target holiday is filled in these added rows; rental and weather cells remain empty. Within the accepted replay range, the CSV supplies the known target holiday. Extension beyond that range requires a defined holiday-data source.

### 9.3 Checks and conditions for revisiting choices

**Tests verify actual feature values, selected target hours, and fitted training inputs.** Changing unavailable observations must leave a request's inputs unchanged; changing validation rental answers must leave training matrices/labels unchanged. Ridge transformations and coefficients are also checked. Batch and API outputs, request rejection, fixed counts, and the near-tie rule are tested.

Locked package versions are in uv.lock.

A new data version requires coordinated source-check and time-split updates before retraining. The current loader validates the documented fixed CSV.

| Current choice | Reason | Reconsider when |
| --- | --- | --- |
| Three seasonal whole-week rounds | About 83% of development targets train per round; all train across rounds | Validation instability warrants more rounds without changing final test |
| W3 candidate with at most 48 hours of weather statistics | Test recent-background statistics; the selected W2 model uses past values and changes | Validation supports longer history, after checking dependencies again |
| sklearn boosting | Shares preprocessing/testing with the other algorithms and limits dependencies | A specific capability/performance need justifies XGBoost |
| FastAPI and static plots | Clear request contract and reviewable first version | A concrete need for interactive UI or another framework arises |

An observed year, few rain/snow events, and correlated adjacent hours limit conclusions. The final test measures 16 days of November; future-season evaluation requires further periods. The optional cloud procedure is described in [Deployment](DEPLOYMENT.md).

References retained for implementation context: [course requirements](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/projects), [lagged-feature example](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html), [cyclical-time example](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html), [time-series evaluation research](https://arxiv.org/abs/1905.11744).

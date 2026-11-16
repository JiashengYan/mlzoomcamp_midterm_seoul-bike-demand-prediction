# Seoul Bike Demand Prediction

**Predict system-wide bicycle rental counts one hour ahead from known dates and completed rental/weather observations.** This project compares models, tunes the strongest candidates, evaluates one final choice, and exposes it through a local web service.

## 1. Problem and data

**The prediction supports analysts studying hourly bicycle activity across the Seoul system.** At 17:00, it estimates rentals during 18:00–19:00 using observations completed through 17:00; the latest source row covers 16:00–17:00.

The Seoul Bike Sharing Demand dataset contains 8,760 hourly records and 14 fields, covering December 2017 through November 2018. It supplies rental counts, eight weather measurements, date/holiday information, and operating status. The [data guide](data/README.md) records the UCI source, Kaggle reference, attribution, and download verification.

The output estimates rental events under normal operation. The current service replays historical requests against this fixed dataset. Target dates and holidays are known; weather inputs come from completed observations. The project assumes records describe hour-start intervals and become available when their hour ends.

## 2. Approach and main results

**Across 384 normal-operation test hours, 77.6% of model predictions were within 20% of actual rentals, compared with 61.5% for the matching-hour weekly-mean rule.**

~~~mermaid
flowchart LR
    A["Inspect data and build historical inputs"] --> B["Compare Ridge, forest, and boosting"]
    B --> C["Tune the two strongest families"]
    C --> D["Choose on development validation"]
    D --> E["Fit once; evaluate later test hours"]
~~~

The experiment compared 22 configurations using the same three validation rounds. Validation weeks span all 12 observed months; the final model is fitted on all 7,910 development targets. The selected gradient boosting model combines known target times, recent rentals, and past weather values/changes to adapt its estimates to the observed conditions. Notebook section 7.1 measures reliance on these input groups using development validation.

### 2.1 Performance as an operational reference

**The model was closer to actual rentals than the weekly-mean rule in 239 of 384 hours (62.2%).** This supports its reference value for most operating hours in this test, beyond directly reusing matching historical counts.

| Method | Hours within 20% of actual | Mean hourly percentage error |
| --- | ---: | ---: |
| Selected gradient boosting | 77.6% | 21.3% |
| Mean of matching weekday/hour counts from the previous 1–4 weeks | 61.5% | 35.8% |

Percentage error is the absolute prediction error divided by actual rentals for that hour. The 20% band is a descriptive reporting choice; model selection uses validation MAE.

Actual rentals averaged about 675/hour. Mean absolute error was 85.25/hour for the model and 128.19/hour for the rule, a 33.5% reduction. This gives the count error its volume context.

**Low-volume hours need closer attention:** the 13 hours below 100 actual rentals averaged about 74 rentals/hour and a mean absolute prediction error of 95 rentals/hour. The overall comparison therefore includes a weakness at low volumes. Final testing covers November 15–30, 2018; the [detailed performance analysis](docs/IMPLEMENTATION_DISCUSSION.md#71-relative-errors-and-reference-value) explains the volume groups and comparison scope.

Development validation can train on months later than a held-out week; final testing uses a model fitted only on earlier targets. These results measure the stated periods.

![Share of test hours within each relative-error level](reports/figures/performance.png)

## 3. Review the existing results

**Open [model_development.ipynb](model_development.ipynb) to inspect the executed development workflow and saved outputs.** The notebook records the calls that generate its tables and charts.

| What to review | Notebook sections |
| --- | --- |
| Data checks, patterns, inputs, and validation design | 1–4 |
| Initial models, parameter-search effects, final selection, and input-group importance | 5–7.1 |
| Final-test scores, predictions, and error groups | 8–9 |
| Every initial and tuned candidate | 10 |

For individual scores and settings, inspect [metrics.json](reports/metrics.json) and [validation.csv](reports/validation.csv). The [actual/predicted timeline](reports/figures/replay.png) shows when errors occurred. The detailed [report index](docs/IMPLEMENTATION_DISCUSSION.md#72-reports-for-detailed-checks) explains the other generated files.

## 4. Reproduce the experiment

**Prepare the environment once, then choose the notebook route to replay development or the batch route to generate a saved service model.** Both routes call the same project functions and use the same fixed protocol.

Use Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/). From the project root, run:

~~~powershell
python scripts/download_data.py
uv sync --locked --python 3.11
~~~

The first command downloads or verifies the fixed CSV. The second installs the locked environment in .venv. Each `uv run` command activates that environment for its process, so a separate shell-activation step is unnecessary. To activate it manually in PowerShell, use `.\.venv\Scripts\Activate.ps1`; in bash, use `source .venv/bin/activate`.

### 4.1 Replay development in the notebook

**Open the English notebook, choose the project's .venv Python kernel, and run all cells from the beginning.** Select .venv/Scripts/python.exe on Windows or .venv/bin/python on Linux/macOS.

This generates data analysis, model comparisons, tuning charts, input-group importance, and final-test scores directly from the CSV. The fitted model remains in the notebook kernel; tuning, feature-importance, and performance figures are saved under reports/figures. Prior model/report files are not required. Allow several minutes; runtime varies by machine.

### 4.2 Run batch training and export

**Use this route to generate models/model.joblib and the standard CSV/JSON reports.** It reruns model comparison, tuning, final fitting, and evaluation without a notebook UI.

~~~powershell
uv run --locked bike-train
uv run --locked pytest -q
~~~

The expected selected configuration is gradient boosting with W2, with validation MAE around 121.32 and test MAE around 85.25. Batch training replaces generated model/report files. Run the tests after export: the API test requires the saved model.

## 5. Test the web service

**The service loads models/model.joblib; generate it with section 4.2 when starting from a fresh checkout.** Choose either local startup or Docker, then send the example request from another terminal.

### 5.1 Start locally

**Launch the API on localhost using the saved model and fixed CSV.**

~~~powershell
uv run --locked uvicorn seoul_bike.api:app --host 127.0.0.1 --port 8000
~~~

### 5.2 Start with Docker

**With Docker Engine running, build and start the image containing the code, fixed CSV, and saved model.**

~~~powershell
docker build -t seoul-bike-demand:v1 .
docker run --rm -p 127.0.0.1:8000:8000 seoul-bike-demand:v1
~~~

### 5.3 Send the example request

**For either startup route, call the service in a second terminal.**

~~~powershell
uv run --locked python scripts/predict_example.py
~~~

The example sends `POST /predict` with:

~~~json
{"issued_at": "2018-11-16T17:00:00+09:00"}
~~~

A local HTTP check on November 16, 2026 returned HTTP 200 for both health and prediction. The prediction response included:

~~~json
{
  "predicted_rentals": 1554.2845177480015,
  "target_start": "2018-11-16T18:00:00+09:00",
  "model_id": "seoul-bike-v1-c17"
}
~~~

This excerpt shows the prediction, target hour, and saved model used; the full response also includes the interval end, observation cutoff, and normal-operation assumption. `GET /health` reports readiness; [localhost /docs](http://127.0.0.1:8000/docs) provides the interactive API reference. Exact accepted times, response fields, and errors are described in the [API contract](docs/IMPLEMENTATION_DISCUSSION.md#81-request-response-and-errors).

For an optional hosted service, follow the [Cloud Run recipe](docs/DEPLOYMENT.md) after account and budget approval. It reuses this Docker image. Cloud execution and a hosted test URL are pending.

## 6. Code and detailed documentation

**The notebook presents the experiment; the shared package and independent tools support execution.** src/seoul_bike contains data preparation, modeling, batch retraining, and the API. scripts contains standard-library data-download and HTTP-call tools. Dependencies are in pyproject.toml/uv.lock, and Dockerfile builds the service image.

| Detail needed | Reference |
| --- | --- |
| Problem scope, intended use, and evidence limits | [Project brief](docs/PROJECT_BRIEF.md) |
| Source fields, examples, distributions, and observed patterns | [Data analysis](docs/DATA_ANALYSIS.md) |
| Exact features, splits, model settings, API contract, and function locations | [Implementation](docs/IMPLEMENTATION_DISCUSSION.md) |
| Dataset source, attribution, and download checks | [Data guide](data/README.md) |
| Optional cloud deployment, authenticated calls, and cleanup | [Deployment recipe](docs/DEPLOYMENT.md) |

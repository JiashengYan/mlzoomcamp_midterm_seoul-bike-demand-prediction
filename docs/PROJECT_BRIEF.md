# Project brief: one-hour-ahead Seoul bicycle rentals

**Build one model that estimates hourly bicycle rental events from known dates and completed observations.** The intended audience is an analyst or researcher studying system-wide activity and prediction errors.

Updated: 2026-11-16. Scope: Machine Learning Zoomcamp Modules 1–6. Version 1 training, evaluation, a local API, and Docker are implemented. The root [development notebook](../model_development.ipynb) is the main project entry; shared modules support it and batch retraining.

## 1. Problem and practical use

**Estimate how many rental events will occur in a specified hour under normal operation.** At 17:00, the target is 18:00–19:00. A system-level estimate helps examine hourly patterns and unusual activity; its use is system-wide activity analysis.

`Rented Bike Count` counts rental events. The dataset's system-wide hourly resolution defines the project's prediction scope. The model preserves continuous counts and genuine peaks.

## 2. Data available

**The fixed Seoul Bike Sharing Demand CSV covers one year with one row per hour.** It has 8,760 rows and 14 columns from 2017-12-01 through 2018-11-30; 8,465 target hours are marked as operating.

| Information | Use |
| --- | --- |
| Date, hour, holiday, and season labels | Understand daily, weekly, and within-year patterns |
| Historical rental counts | Represent recent demand and matching past hours |
| Eight weather measurements | Represent observed conditions and their recent changes |
| Operating status | Exclude shutdown hours from training and evaluation target labels; retain the complete hourly history |

The [Kaggle entry](https://www.kaggle.com/datasets/saurabhshahane/seoul-bike-sharing-demand-prediction) corresponds to the [UCI source](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand). Download and attribution are in [Data source](../data/README.md); real examples and descriptive evidence are in [Data analysis](DATA_ANALYSIS.md).

## 3. Information boundary

**Each prediction uses only records completed by its issue time, plus dates and holidays known in advance.** With issue time T and target start S=T+1 hour, the latest complete source row starts at S−2 hours.

~~~mermaid
flowchart LR
    A["16:00–17:00 observation"] --> B["17:00 issue"]
    B --> C["18:00–19:00 target"]
~~~

Version 1 assumes Hour is the interval start and both rental and weather records are available immediately after that hour ends. These are research assumptions requiring source confirmation; configurable publication delays are not implemented.

The service reads local history; callers provide only an issue time. The model directly predicts rentals from completed observations and known schedule values. Predictions assume normal operation. Non-operating hours are excluded from training, validation, and test target labels. Their source rows remain in the time axis, and recorded zeros still contribute to historical count inputs, alongside selected past operating-status inputs.

## 4. How the model uses time and weather

**One shared model combines calendar position with actual recent conditions.** It learns from all observed seasons and allows time and weather inputs to interact.

Hour, weekday, weekend, holiday, and two month-cycle values describe time. Historical counts and past weather describe recent conditions. Weather comparisons start with values at selected previous hours, add differences between previous hours, and then test averages and other statistics over consecutive recent hours. The final comparison selected the middle option, W2.

The final W2 model combines month coordinates with past weather values and changes, so it can use annual position alongside the recent observed conditions. The 48-hour weather average was part of the W3 alternative tested during comparison. Rain/snow measurements are used numerically; their physical interpretation is documented in [Data analysis](DATA_ANALYSIS.md).

## 5. Evaluation and delivery

**Select the configuration on development data, then fit one final model and evaluate a later untouched period.** Each validation round includes one week per season; three rounds rotate through each season's first, middle, and last month, covering all 12 months with three fits per candidate. This balances coverage and computation. Validation temporarily reduces each round's training set; a held-out target can train in another round, and final fitting uses all 7,910 development targets. The later final-test targets remain reserved for evaluation.

There are 7,910 development targets and 384 final-test targets on November 15–30, 2018. For example, validation on a June week can use a model trained on July–October as well as earlier records; that score describes a model trained across the observed year's different periods. The final test is after final training. Exact dates and exclusion rules belong to [Implementation](IMPLEMENTATION_DISCUSSION.md).

Compare Ridge, random forest, and gradient boosting. MAE is the average absolute count error; it is accompanied by large-error, bias, seasonal, and weather-group diagnostics. Version 1 delivers a reproducible training command, saved preprocessing/model, English analysis notebook, API, dependency lock, Docker image recipe, and prediction/error plots. Prediction intervals, category probabilities, and a separate frontend are outside the initial scope.

## 6. Evidence and limits

**The selected gradient boosting + W2 model has validation MAE 121.3 and final-test MAE 85.2 rentals/hour.** The rule averaging the same weekday and hour over the previous one to four weeks has test MAE 128.2; the comparison records every evaluated configuration.

The evidence covers associations within the observed year and prediction performance during the 16-day final test. Next-year reliability, operational benefit, and causal weather effects require additional data or dedicated evaluation. Dataset assumptions remain visible in the documents and model metadata.

Run instructions are in [README](../README.md), with actual metrics in [reports](../reports/metrics.json). Course context: [official project requirements](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/projects).

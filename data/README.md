# Dataset source and handling

**Use the fixed original CSV unchanged; compute features separately.** This file covers source, attribution, download, and units. Analysis is in [Data analysis](../docs/DATA_ANALYSIS.md).

Updated: 2026-11-16.

## 1. Source and attribution

**Download the UCI original without requiring a Kaggle login.** The selected Kaggle copy was previously checked against this fixed CSV.

- [Kaggle entry](https://www.kaggle.com/datasets/saurabhshahane/seoul-bike-sharing-demand-prediction)
- [UCI dataset](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)
- [Original ZIP](https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip)
- Citation: Seoul Bike Sharing Demand [Dataset]. (2020). UCI Machine Learning Repository. [DOI:10.24432/C5F62R](https://doi.org/10.24432/C5F62R)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as listed by UCI; retain attribution.

## 2. Download or verify

**The standard-library script verifies an existing file and downloads only when it is absent.** Run from the project root:

~~~powershell
uv run --locked python scripts/download_data.py
~~~

Prepare the environment using [README section 4](../README.md#4-reproduce-the-experiment) first. These commands work on Windows and Ubuntu.

Destination: `data/raw/SeoulBikeData.csv`. The downloader and runtime loader accept
Windows CRLF and Unix LF line endings without SHA-256 verification or rewriting the
file. The downloader checks 8,760-by-14 dimensions; the runtime loader checks the CSV
schema and hourly timeline. Equivalent CSV formatting does not block startup.
`.gitattributes` also disables Git line-ending conversion for tracked raw CSV files.
Existing models remain usable without retraining. Historical model metadata may still
include `source_sha256` for reference; it is not used to gate loading.

## 3. Read the rows and units

**One row records system-wide rental events and background conditions for an hour.** The CSV covers 2017-12-01 through 2018-11-30, with 8,760 rows and 14 columns; 8,465 hours are operating and 295 are not.

Read using `latin-1` and parse Date as `%d/%m/%Y`. The loader combines Date and Hour into the index, then removes Date as a column, so its table has 13 ordinary columns plus timestamps while the source CSV retains 14 columns. Rental counts measure the number of recorded rental events. Temperature/dew point use °C; humidity %; wind m/s; visibility 10m; radiation MJ/m²; Rainfall mm; Snowfall cm.

The project treats Hour as the interval start and observations as available at hour end, both explicit assumptions. Rain/snow recording semantics and publication delays still need source confirmation. A positive Snowfall value records that field's measured value; active snowfall and road-state interpretation require confirmed source semantics.

## 4. Keep source rows separate from target selection

**Select normal-operation target labels while retaining every source hour on the timeline.** The current historical count features include recorded shutdown zeros; humidity zero is marked missing when constructing weather inputs. The handling and its implications are explained in [Implementation](../docs/IMPLEMENTATION_DISCUSSION.md). The raw CSV remains unchanged.

`raw/` holds the source; features are computed directly in memory. Target roles are recorded in `reports/splits.csv`. Downloaded data and generated models are ignored by Git; scripts and English descriptions support reproduction. Exact split rules are in [Implementation](../docs/IMPLEMENTATION_DISCUSSION.md).

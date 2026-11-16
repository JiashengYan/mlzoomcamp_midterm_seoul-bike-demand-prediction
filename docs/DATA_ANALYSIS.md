# Data analysis: read the rows before interpreting the model

**The data supports system-wide hourly rental prediction: the hourly sequence is complete, weekday/weekend patterns differ, and counts vary with season and recorded weather.** Of 8,760 rows, 295 shutdown hours are excluded as target labels; 17 suspicious humidity zeros are handled in features. The sections below explain these findings and their evidence. Model accuracy is evaluated separately in the [analysis notebook](../model_development.ipynb).

Updated: 2026-11-16. Descriptive statistics were computed on 2026-11-13; split counts were checked on 2026-11-15. The source retains its fixed file fingerprint.

## 1. Start with one row

**A row contains system-wide totals and background conditions for one hour.** Date and Hour identify the record; rental count is the target, with weather, holiday, season, and operating context alongside it.

| Item | Description |
| --- | --- |
| Format | Comma-separated CSV; header first; no combined timestamp column |
| File size | 604,166 bytes, approximately 590.0 KiB |
| Dimensions | 8,760 data rows, 14 columns; header excluded |
| Encoding | Read as latin-1; source bytes preserved |
| Date format | Day/month/year; 01/12/2017 means December 1, 2017 |
| Time coverage | 2017-12-01 00:00 through 2018-11-30 23:00, interpreted as Seoul time |
| Row meaning | System-wide rentals and weather/holiday/operating context for one date and hour |

### 1.1 Original header and first three records

**These lines are copied directly from the source, without cleaning or reordering.** Read 01/12/2017 as December 1, 2017.

~~~csv
Date,Rented Bike Count,Hour,Temperature(°C),Humidity(%),Wind speed (m/s),Visibility (10m),Dew point temperature(°C),Solar Radiation (MJ/m2),Rainfall(mm),Snowfall (cm),Seasons,Holiday,Functioning Day
01/12/2017,254,0,-5.2,37,2.2,2000,-17.6,0,0,0,Winter,No Holiday,Yes
01/12/2017,204,1,-5.5,38,0.8,2000,-17.6,0,0,0,Winter,No Holiday,Yes
01/12/2017,173,2,-6,39,1,2000,-17.7,0,0,0,Winter,No Holiday,Yes
~~~

### 1.2 Ordinary, peak, and non-operating examples

**The examples show ordinary activity (254), peak activity (3,556), and a non-operating hour (0).** Operating status identifies which target hours are evaluated, and all recorded counts remain available as history.

| Date | Hour | Rentals | Temperature °C | Humidity % | Rainfall mm | Operating |
| --- | --- | --- | --- | --- | --- | --- |
| 01/12/2017 | 0 | 254 | -5.2 | 37 | 0.0 | Yes |
| 01/12/2017 | 1 | 204 | -5.5 | 38 | 0.0 | Yes |
| 19/06/2018 | 18 | 3,556 | 24.1 | 57 | 0.0 | Yes |
| 11/04/2018 | 0 | 0 | 14.4 | 82 | 0.0 | No |

Under the project's explicit assumption, Hour=0 covers 00:00–01:00. The interval convention still needs confirmation from the source. After loading, Date is combined with Hour to form the time index, and Date is removed from the ordinary columns: the in-memory table has 13 columns plus that index, while the CSV still has 14 columns.

## 2. Understand the 14 fields

**The CSV has one target column, two date/hour columns, eight weather measurements, and three category/status columns.** The evaluation compares the predictive contribution of these partly related measurements. Ranges below use all 8,760 rows.

### 2.1 Time and target

**Date and Hour locate an hour; Rented Bike Count counts rental events during it.** Each recorded rental event contributes to that hour's count.

| Original field | Content and type | Observed example/range | Interpretation |
| --- | --- | --- | --- |
| `Date` | Date text | `01/12/2017`; 365 dates | Parse Date and combine it with Hour for chronological ordering |
| `Hour` | Integer hour | 0—23 | Combine with Date to identify an hour |
| `Rented Bike Count` | Integer count | All: 0—3,556; operating: 2—3,556 | Hourly rental events, used as the numeric prediction target |

### 2.2 Weather measurements and units

**Keep the observed numeric values and distinguish their units from an interpretation of behavior.** Interpreting accumulated rain or snow requires confirmation of how each source measurement is recorded.

| Original field | Meaning | Unit | Observed range | Notes |
| --- | --- | --- | --- | --- |
| `Temperature(°C)` | Temperature | °C | -17.8 to 39.4 | Weather context; associations may be nonlinear |
| `Humidity(%)` | Relative humidity | % | 0 to 98 | 17 zeros need verification |
| `Wind speed (m/s)` | Wind speed | m/s | 0.0 to 7.4 | Retain recorded zero wind values |
| `Visibility (10m)` | Visibility | 10 m | 27 to 2000 | 2,000 in units of 10 m represents 20,000 m |
| `Dew point temperature(°C)` | Dew point | °C | -30.6 to 27.2 | Strongly correlated with temperature; potentially redundant |
| `Solar Radiation (MJ/m2)` | Solar radiation | MJ/m² | 0.0 to 3.52 | Zero can be valid; associated with time of day |
| `Rainfall(mm)` | Rainfall record | mm | 0.0 to 35.0 | Use recorded values; accumulation semantics require confirmation |
| `Snowfall (cm)` | Snowfall record | cm | 0.0 to 8.8 | Recorded values and differences are used as inputs; cumulative physical interpretation requires confirmed recording semantics |

Visibility 2,000 means 20 km. Snowfall inputs describe recorded values and recent changes. Interpreting active snowfall, road conditions, or travel behavior requires confirmed measurement semantics and relevant additional observations.

### 2.3 Season, holiday, and operating status

**Holiday is separate from weekend, and Functioning Day must be interpreted per hourly row.** The season label locates dates within this observed year.

| Original field | Values and all-hour counts | Use and limits |
| --- | --- | --- |
| `Seasons` | Winter 2,160；Spring 2,208；Summer 2,208；Autumn 2,184 | Season categories for the observed year |
| `Holiday` | Holiday 432；No Holiday 8,328 | Source holiday marker; separate from weekend |
| `Functioning Day` | Yes 8,465；No 295 | Despite its name, inspect hourly status; some dates have partial shutdowns |

The model's output follows the dataset's system-wide hourly resolution, using one shared model across the observed seasons.

## 3. Data quality and feature handling

**Every expected timestamp and recorded cell is present. Feature handling also accounts for the specific values described below.** Keep the original CSV unchanged; feature-layer handling is described separately.

### 3.1 Completeness

**Every expected hour appears once and all parsed cells contain recorded values.** This supplies the complete hourly sequence needed for historical calculations. The following checks address individual value semantics.

| Check | Result | Interpretation |
| --- | --- | --- |
| Duplicate raw rows | 0 | No duplicates found |
| Duplicate date/hour pairs | 0 | One row per hour |
| Missing hourly timestamps | 0 | Covers 365×24 hours |
| Nulls after parsing | 0 | All fields contain values |
| Blank/common missing markers | 0 | Checked empty strings, na, n/a, null, none, ? |
| File version | SHA-256 matches the downloader's fixed value | Analysis uses the downloaded version |

Common missing markers checked were empty strings, na, n/a, null, none, and ?. The file fingerprint establishes version consistency; measurement quality is examined through the field-specific checks below.

### 3.2 Non-operating rows

**Exclude all 295 shutdown hours from training, validation, and test target labels: their zero rentals reflect shutdown, while the project predicts normal-operation demand.** Operating hours range from 2 upward. There are 12 fully non-operating dates and one with 7 non-operating hours.

| Non-operating date | Non-operating hours |
| --- | --- |
| 2018-04-11 | 24 |
| 2018-05-10 | 24 |
| 2018-09-18 | 24 |
| 2018-09-19 | 24 |
| 2018-09-28 | 24 |
| 2018-09-30 | 24 |
| 2018-10-02 | 24 |
| 2018-10-04 | 24 |
| 2018-10-06 | 7 |
| 2018-10-09 | 24 |
| 2018-11-03 | 24 |
| 2018-11-06 | 24 |
| 2018-11-09 | 24 |

There are 352 fully operating dates and one additional date with 17 operating hours. Shutdowns are 3.4% of all rows. Shutdown causes require additional source information. Build historical inputs on the complete hourly sequence, then use operating-status selectors to choose target hours. This preserves the one-row-per-hour correspondence. The current implementation includes shutdown zeros in historical count values and means, together with selected past operating-status inputs. Excluding shutdown target labels and retaining shutdown history are separate processing choices.

### 3.3 Humidity zero

**Treat the 17 suspicious humidity zeros as missing in model inputs, retain missing-value indicators, and learn filling values from training rows only.** For example, at 2018-05-19 07:00 temperature is 11.4°C and dew point 4.5°C, yet humidity is zero. The cause remains unconfirmed. These records occur on five dates in May, all during operation, about 0.2% of all rows.

| Source date | Zero-humidity hours |
| --- | --- |
| 21/05/2018 | 6 |
| 22/05/2018 | 4 |
| 28/05/2018 | 4 |
| 27/05/2018 | 2 |
| 19/05/2018 | 1 |

The original CSV retains these measurements unchanged; the missing-value treatment applies when building model inputs.

### 3.4 Other zeros and repeated maximums

**Feature handling follows each measurement's meaning.** Solar radiation is zero in 4,300 hours; visibility 2,000 occurs 2,245 times. A recording cap would need source evidence. Separate suspicious measurements from plausible rental peaks.

## 4. See how rental counts are distributed

**Operating mean demand is 729.2 rentals/hour, median 542, and maximum 3,556.** The model retains the observed peaks and their contribution to prediction error.

### 4.1 Percentiles and totals

**A percentile describes where observations fall in the distribution.** Operating P90≈1,696.6 means about 90% of operating hourly counts are at or below that value; it describes the observed count distribution.

| Statistic | All 8,760 hours | Operating 8,465 hours |
| --- | --- | --- |
| Hours | 8,760 | 8,465 |
| Mean | 704.6 | 729.2 |
| Median P50 | 504.5 | 542.0 |
| Minimum | 0.0 | 2.0 |
| 25th percentile P25 | 191.0 | 214.0 |
| 75th percentile P75 | 1,065.2 | 1,084.0 |
| 90th percentile P90 | 1,671.1 | 1,696.6 |
| 95th percentile P95 | 2,043.0 | 2,058.8 |
| 99th percentile P99 | 2,526.2 | 2,556.4 |
| Maximum | 3,556.0 | 3,556.0 |

Percentiles use linear interpolation, so decimal percentile cutoffs fall between the observed integer counts. Total rentals over the year are 6,172,314 rental events.

### 4.2 Descriptive count ranges

**These ranges describe the observed distribution.** The model keeps continuous counts.

| Rental range | Operating hours | Share of operating hours |
| --- | --- | --- |
| Below 100 | 916 | 10.8% |
| 100—499 | 3,150 | 37.2% |
| 500—999 | 1,979 | 23.4% |
| 1,000—1,499 | 1,286 | 15.2% |
| 1,500—1,999 | 650 | 7.7% |
| 2,000—2,499 | 387 | 4.6% |
| 2,500—2,999 | 66 | 0.8% |
| 3,000 and above | 31 | 0.4% |

Even 100–999 spans substantially different activity levels; tenfold magnitude buckets would hide useful distinctions.

### 4.3 Largest observed counts

**The five highest records all start at 18:00 in June and are compatible with a high-activity period.** These observations are retained for training and evaluation as recorded high-activity hours.

| Date | Target hour | Actual rentals | Temperature °C | Rainfall mm |
| --- | --- | --- | --- | --- |
| 19/06/2018 | 18 | 3,556 | 24.1 | 0.0 |
| 21/06/2018 | 18 | 3,418 | 27.8 | 0.0 |
| 12/06/2018 | 18 | 3,404 | 24.9 | 0.0 |
| 20/06/2018 | 18 | 3,384 | 27.0 | 0.0 |
| 04/06/2018 | 18 | 3,380 | 24.4 | 0.0 |

Further source issues can be investigated, while the current model retains these recorded peak values.

## 5. Observe daily, weekly, and monthly patterns

**Hour, type of day, and position within the observed year are all useful context.** Each profile summarizes observed counts for the stated group of hours.

### 5.1 Weekdays versus weekends

**Weekdays rise sharply at 08:00 and peak at 18:00; weekends have a broader afternoon peak at 17:00.** Use 335 complete operating, non-holiday dates: 238 Monday–Friday and 97 Saturday–Sunday dates.

| Hour start | Weekday mean rentals | Weekend mean rentals |
| --- | --- | --- |
| 00:00 | 515.9 | 685.1 |
| 01:00 | 395.6 | 560.0 |
| 02:00 | 268.8 | 419.2 |
| 03:00 | 177.8 | 289.4 |
| 04:00 | 121.0 | 178.2 |
| 05:00 | 154.1 | 128.7 |
| 06:00 | 362.4 | 168.2 |
| 07:00 | 805.3 | 264.7 |
| 08:00 | 1,353.5 | 436.2 |
| 09:00 | 746.8 | 527.9 |
| 10:00 | 543.2 | 573.8 |
| 11:00 | 612.1 | 664.3 |
| 12:00 | 696.8 | 805.5 |
| 13:00 | 697.7 | 911.3 |
| 14:00 | 712.4 | 961.6 |
| 15:00 | 785.1 | 1,042.2 |
| 16:00 | 908.9 | 1,109.1 |
| 17:00 | 1,194.8 | 1,179.1 |
| 18:00 | 1,766.3 | 1,152.2 |
| 19:00 | 1,330.7 | 1,080.0 |
| 20:00 | 1,156.5 | 1,036.2 |
| 21:00 | 1,119.0 | 991.8 |
| 22:00 | 1,008.2 | 872.8 |
| 23:00 | 715.4 | 674.3 |

'Weekday' here means Monday–Friday. The shape is consistent with a commuting interpretation; confirming trip purpose requires purpose-specific records.

### 5.2 Similarity within day types

**Normalized Monday–Friday profiles are highly similar; weekday/weekend profiles differ more.** Divide each hourly count by that date's complete 24-hour mean, then average by day of week.

| Compared profiles | Profile correlation |
| --- | --- |
| Any pair of Monday–Friday average profiles | 0.983—0.996 |
| Saturday versus Sunday | 0.989 |
| Pooled weekday versus pooled weekend profiles | 0.616 |

The full-date mean is only for retrospective analysis; prediction inputs use observations completed by their issue time. Profile correlation measures similarity of the observed curve shapes.

### 5.3 Holidays

**The file has 18 holiday dates; the operating subset has 17.** Source Holiday records total 432 hours, of which 408 are operating. Operating holiday mean is 529.2 versus 739.3 on non-holidays. Month/weather composition also differs, so the group difference describes their combined observed contexts.

### 5.4 Month-level differences

**Winter activity is lower and June has the highest mean; hotter July/August do not continue increasing.** Each month appears once, so recurrence next year is untested.

| Month | Operating hours | Dates with operating hours | Mean rentals/hour | Mean temperature °C |
| --- | --- | --- | --- | --- |
| 2017-12 | 744 | 31 | 249.1 | -1.9 |
| 2018-01 | 744 | 31 | 201.6 | -3.9 |
| 2018-02 | 672 | 28 | 225.9 | -1.7 |
| 2018-03 | 744 | 31 | 511.6 | 8.0 |
| 2018-04 | 696 | 29 | 753.2 | 13.0 |
| 2018-05 | 720 | 30 | 982.1 | 18.2 |
| 2018-06 | 720 | 30 | 1,245.7 | 23.1 |
| 2018-07 | 744 | 31 | 987.2 | 27.8 |
| 2018-08 | 744 | 31 | 876.2 | 28.7 |
| 2018-09 | 624 | 26 | 1,079.5 | 21.7 |
| 2018-10 | 665 | 28 | 978.5 | 12.7 |
| 2018-11 | 648 | 27 | 718.7 | 7.4 |

Date counts include any date with at least one operating hour; October includes a partial shutdown date. The differences also combine weather, holidays, and unobserved changes. Month-cycle inputs and recent history are learned in the same model, with one shared predictor.

## 6. Examine weather associations carefully

**The following groups describe how recorded weather and rental counts vary together.** Their predictive contribution must be checked using allowed historical inputs.

### 6.1 Recorded rainfall

**Rainfall>0 covers 516 operating hours, about 6.1%, and its mean rentals are lower than the zero-rainfall group's.** Rainfall and Snowfall retain their separate recorded values; rainfall zero identifies a group by that source field alone.

| Rainfall record | Operating hours | Mean rentals/hour | Median |
| --- | --- | --- | --- |
| Rainfall = 0 | 7,949 | 765.6 | 591.0 |
| Rainfall > 0 | 516 | 167.3 | 71.0 |

The following table additionally groups by hour range. Each group retains its observed temperature, month, and day-type composition.

| Hour-start range | Rainfall=0 hours | Mean rentals in that group | Rainfall>0 hours | Mean rentals in that group |
| --- | --- | --- | --- | --- |
| 00-05 | 1,971 | 315.6 | 141 | 103.9 |
| 06-09 | 1,302 | 703.0 | 109 | 159.7 |
| 10-15 | 2,009 | 743.7 | 109 | 188.3 |
| 16-19 | 1,334 | 1,288.3 | 78 | 274.3 |
| 20-23 | 1,333 | 1,002.3 | 79 | 156.0 |

The mean difference measures the association between the recorded groups. Estimating a causal rainfall effect requires a separate study design.

### 6.2 Temperature

**The 20–30°C group has a higher mean, while the above-30°C group does not keep rising.** The groups also differ in timing and other conditions.

| Temperature range | Operating hours | Mean rentals/hour | Median |
| --- | --- | --- | --- |
| Below 0°C | 1,433 | 198.0 | 169.0 |
| 0°C to below 10°C | 2,118 | 426.3 | 333.0 |
| 10°C to below 20°C | 2,050 | 837.2 | 790.0 |
| 20°C to below 30°C | 2,340 | 1,148.1 | 1,075.0 |
| 30°C and above | 524 | 1,112.2 | 916.0 |

These temperature ranges organize the descriptive analysis. Forecasting uses numeric historical weather values together with the other model inputs.

### 6.3 Rain and snow coverage

**Positive Snowfall records mostly occur in winter; detailed combinations have much less support.** Splitting further by weekday, hour, and temperature can leave few events.

| Source season | Operating hours | Rainfall>0 hours | Snowfall>0 hours |
| --- | --- | --- | --- |
| Winter | 2,160 | 41 | 392 |
| Spring | 2,160 | 191 | 0 |
| Summer | 2,208 | 197 | 0 |
| Autumn | 1,937 | 87 | 51 |

There are 443 operating hours with Snowfall>0, with 28 also Rainfall>0. These are frequencies of positive source records, interpreted under the measurement limits in section 2.2.

### 6.4 Correlations

**Temperature is positively associated with rentals, and weather fields can repeat information.** Nonlinear predictive contribution is evaluated through the model comparison.

| Weather field | Pearson correlation with rentals | Spearman correlation with rentals |
| --- | --- | --- |
| Temperature | 0.563 | 0.612 |
| Relative humidity | -0.207 | -0.229 |
| Wind speed | 0.125 | 0.155 |
| Visibility | 0.212 | 0.199 |
| Dew point | 0.400 | 0.411 |
| Solar radiation | 0.274 | 0.410 |
| Rainfall record | -0.129 | -0.306 |
| Snowfall record | -0.152 | -0.247 |

Use operating hours; omit the 17 humidity zeros for humidity correlation, keep other weather values. Pearson describes linear association; Spearman describes rank association. Temperature and dew point have Pearson correlation about 0.914. These coefficients describe the observations; fitted predictive contribution is assessed in the model evaluation.

### 6.5 Recent weather statistics

**Use actual previous values, past changes, and means over consecutive hours to describe known weather history.** For example, a six-hour temperature mean averages six neighboring completed hourly records, using one value for each completed hour in that consecutive period.

W1 and W2 describe past values and changes; W3 additionally tests recent consecutive-hour statistics. The selected model uses W2. Definitions belong to [Implementation](IMPLEMENTATION_DISCUSSION.md). Annual month position and recent weather measurements describe different aspects of conditions.

## 7. Relate past records to a future target

**Each target uses its known date/holiday and observations completed by the issue time.** Source observations provide historical features; target-role selection determines which actual counts participate in fitting or scoring.

### 7.1 Historical count associations

**Counts two hours earlier, yesterday, and last week are correlated with operating targets, partly because they share time patterns and background.** This motivates testing history inputs, and then measuring their contribution in validation.

| Hours before target | Comparable target hours | Correlation with target rentals |
| --- | --- | --- |
| 2 hours | 8,463 | 0.759 |
| 24 hours | 8,441 | 0.730 |
| 48 hours | 8,417 | 0.682 |
| 168 hours | 8,297 | 0.701 |

For each target, find its historical count by the earlier timestamp on the complete hourly sequence, then select operating targets with enough history. Historical shutdown zeros remain. These coefficients quantify historical count associations.

### 7.2 Prediction inputs for a 17:00 issue

**The target is 18:00–19:00 and the latest completed source row covers 16:00–17:00.** The local service reads that history for the request.

| Information | Status at 17:00 | Allowed input? |
| --- | --- | --- |
| Rentals and weather for 16:00–17:00 | Assumed available at hour end | Yes; each publication delay needs confirmation |
| Final rentals for 17:00–18:00 | Not complete | No |
| Date/time and holiday for 18:00–19:00 | Known schedule | Yes |
| Actual weather for 18:00–19:00 | Not yet observed | Never a prediction input; post-prediction analysis only |
| Actual rentals for 18:00–19:00 | Prediction target | No |

The historical record with the latest allowed start time is S−2 hours. For target 18:00 it describes 16:00–17:00, completed by the 17:00 issue time. The code reads that past value at the target feature row using shift(2); the output row keeps the target timestamp. Actual future outcomes are for scoring or later error grouping only.

## 8. Summary of findings and target roles

**The evidence supports comparing models that jointly use dates, historical counts, and historical weather.** Keep genuine peaks, preserve shutdown rows for history, and avoid interpreting associations as business actions.

| Finding | Evidence | Implication |
| --- | --- | --- |
| Complete hourly series | 365 days, 8,760 consecutive hours; no duplicate or missing timestamps | Construct histories on actual hourly timestamps |
| Zero rentals coincide with non-operation | All 295 zero-count rows have Functioning Day=No | Exclude shutdown hours as training, validation, and test targets |
| Wide count range | Operating mean 729.2, median 542.0, maximum 3,556 rentals/hour | Keep continuous counts and peaks; avoid premature coarse classes |
| Weekday/weekend profiles differ | On complete non-holiday dates, weekdays peak at 18 and weekends at 17 | Model hour together with day type |
| Monthly levels vary | Operating means: month 1 has 201.6, month 6 has 1,245.7 | Use recent background and annual position; cross-year recurrence requires additional years |
| Weather is worth investigating | The 516 Rainfall>0 hours average 167.3 rentals; Rainfall=0 averages 765.6 | Test historical weather and changes; group means describe the observed weather associations |
| Historical counts are associated | Operating targets correlate about 0.759 with counts 2 hours earlier | Evaluate history-based inputs; measure predictive gain using the defined validation protocol |

**The 8,296 operating hours with complete history divide into 7,910 development targets, 384 final-test targets, and two excluded boundary targets.** All target hours are scanned hourly, while the complete source sequence supplies history.

| Part of the 8,296 eligible hours | Hours |
| --- | ---: |
| Development: compare and fit models | 7,910 |
| Final test: evaluate the selected model | 384 |
| Boundary: exclude as labels for the training/issue schedule | 2 |

The three rounds have training counts 6,562/6,593/6,613 and validation counts 624/593/624. Their 1,841 distinct validation targets cover 12 months; every development target trains in at least one round. Exact dates and training-answer exclusions are only specified in [Implementation](IMPLEMENTATION_DISCUSSION.md).

## 9. Reproduction and remaining limits

**The original data fingerprint and scope of each statistic are explicit; model results are separate.** One year, few rain/snow events, and adjacent-hour dependence limit generalization. Hour intervals, publication delays, humidity zeros, and rain/snow semantics still need source clarification.

Unless stated otherwise, field ranges use all rows; rental distributions, months, weather, and lag correlations use operating targets; daily profiles use complete non-holiday dates. Adjacent hours can share conditions; reported coverage includes distinct dates. The original descriptive environment used pandas 3.0.6 and NumPy 2.4.6; the training environment is recorded separately.

CSV SHA-256:

~~~text
373339b71a8935d69e9af0abf26a70744632119862eeb3919efb389a7b749c60
~~~

A minimal read-only calculation:

~~~python
import pandas as pd

df = pd.read_csv("data/raw/SeoulBikeData.csv", encoding="latin-1")
dates = pd.to_datetime(df["Date"], format="%d/%m/%Y")
timestamps = dates + pd.to_timedelta(df["Hour"], unit="h")
operating = df["Functioning Day"].eq("Yes")
summary = df.loc[operating, "Rented Bike Count"].describe()
previous = df["Rented Bike Count"].shift(24)
valid = operating & previous.notna()
correlation = df.loc[valid, "Rented Bike Count"].corr(previous.loc[valid])
~~~

The [English notebook](../model_development.ipynb) runs data analysis, model comparison, parameter tuning, and final evaluation, generating its tables and charts from those computations. [Data source](../data/README.md) records attribution/license; [UCI](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand) provides field context.

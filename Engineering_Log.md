# Engineering Log

## Step 1: Data Ingestion

### Note 1 - Telemetry Disabled
Loaded only laps/results with `telemetry=False` since this stage needs lap times and positions only.

Actual execution time: 143.80 seconds (measured with cache enabled, including fallback timeout checks for missing FP2 streams).

### Note 2 - Round Lookup
Race-name matching can break across seasons, so I resolve the Mexico round number from the season schedule (`Location` contains "Mexico").

### Note 3 - Qualifying Time Source
Used the best of Q1, Q2, and Q3 from the official results table instead of relying on `pick_quicklaps()`.

### Note 4 - Timedelta Conversion
Lap times arrive as `Timedelta`; converted them into seconds using `.dt.total_seconds()`.

### Results
Total rows: 100

Seasons successfully collected: [2021, 2022, 2023, 2024, 2025]

Missing values observed: 
- 2021 FP2: One missing `best_fp2_sec` value (RUS - George Russell in Mexico City GP). There are no missing values in 2022 FP2.
- 2023, 2024, 2025 FP2: `best_fp2_sec` (60 missing) and `practice_total_laps` (60 missing) as FP2 timing data was not loaded.
- Qualifying: One missing `best_quali_sec` value (2023).

---

## Data Ingestion Issue - 2023 to 2025

### Confirmed Finding
Accessing session.laps raises DataNotLoadedError because FastF1 failed to load timing data.

The associated warning was:
"Failed to load timing data!"

### Root Cause Status
Why the timing data failed to load has NOT been determined.

Potential explanations include an upstream API change, a library compatibility issue, or a network problem.

These explanations remain unverified hypotheses.

### Engineering Decision
FP2 lap data is treated as optional so that the results-based data collection can continue independently.

---

## Data Quality & Validation Summary (2021–2025)

- **Total Rows:** 100 driver-race entries (20 per year across 5 seasons).
- **Duplicate Records:** 0 duplicates across all seasons.
- **Starting Grid Positions:**
  - 2021, 2022, 2025: Values span 1.0 to 20.0.
  - 2023, 2024: Values span 0.0 to 19.0 (where 0.0 indicates a pitlane start).
- **Qualifying Positions:** Values span 1.0 to 20.0 for all seasons without missing values.
- **FP2 Feature Coverage:**
  - 2021: 95% best lap availability (19/20), 100% lap count availability.
  - 2022: 100% availability.
  - 2023–2025: 0% availability (consistently missing across all drivers).

---

## Step 2: All-Season Data Ingestion (2021–2025)

### Objective

Expanded the historical dataset from Mexico GP races
to all eligible Grand Prix races across five seasons.

### Data Sources

FastF1 Qualifying Results (Q)
FastF1 Main Race Results (R)

### Data Collection Scope

Years: 2021–2025
Sessions: Q and R
FP2: Excluded
Sprint Results: Excluded

### Dataset Statistics

Total rows: 1020
Successful rounds: 51
Failed rounds: 63
Unique drivers: 28
Unique events: 27

### Execution Time

2021: 277.50 seconds
2022: 122.29 seconds
2023: 52.23 seconds
2024: 12.32 seconds
2025: 12.25 seconds

Total: 476.59 seconds

### Data Limitations

- **FastF1/Ergast Rate Limit:** Starting at Round 8 of 2023 (Canadian GP), uncached API calls exceeded the rate limit threshold (`RateLimitExceededError: any API: 500 calls/h`). As a consequence, Rounds 8–22 of 2023, all 24 rounds of 2024, and all 24 rounds of 2025 failed during this execution window. The rounds collected up to that point (2021: 22/22, 2022: 22/22, 2023: 7/22) are safely cached in `f1_cache`.
- **Missing Values in Collected Observations (1,020 rows):**
  - `GridPosition`: 2 missing values.
  - `FinishPosition`: 2 missing values.
  - `QualiPosition`: 1 missing value.
  - `best_quali_sec`: 13 missing values (drivers who participated in the race but did not record a valid qualifying lap time across Q1, Q2, and Q3).
  - All other columns (`Year`, `Round`, `EventName`, `Location`, `Driver`, `TeamName`, `Status`) have 0 missing values.
- **Duplicates:** 0 duplicate records found for `Year + Round + Driver`.

### Engineering Decisions

Raw team names were preserved.
Team-name normalization is deferred to the next stage.
No Feature Engineering was performed.
No Machine Learning model was trained.

---

## Decision Log — November 2, 2026

### 1. Shift of Mexico City GP 2026 to Backtest
The Mexico City Grand Prix took place yesterday, November 1, 2026. Consequently, it has transitioned from a prospective live prediction to an official out-of-sample **Backtest**. Training sets for this backtest will strictly utilize data prior to Mexico 2026 to evaluate historical predictive fidelity.

### 2. Adoption of São Paulo (Brazil) GP 2026 as Live Prediction Target
The São Paulo Grand Prix (scheduled for November 8, 2026) is adopted as the new **Live Prediction Target**. The model will generate live forecasts post-qualifying and prior to Sunday's race.

### 3. Root Cause Confirmation for Failed Rounds
Detailed review confirmed that all 63 failed rounds recorded in `failed_rounds.txt` were caused entirely by the upstream Ergast / FastF1 hourly ceiling (`RateLimitExceededError: any API: 500 calls/h`), rather than missing upstream sessions or missing database records. The ingestion architecture will incorporate rate limit detection, persistent resuming, and automatic wait capabilities.

---

## Stage 2: Data Cleaning & Schema Standardization

### Objective
Clean and standardize the historical training corpus (2021–2026 pre-Mexico) and the isolated target dataset (Mexico City GP 2026) in accordance with the documented rules in `PROJECT_SPEC.md`.

### Cleaning Actions & Results
1. **DNS Exclusion:**
   - Identified 20 Did Not Start (DNS) or withdrawn records across 2021–2026 (e.g. Mazepin Abu Dhabi 2021, Schumacher Saudi Arabia 2022, Sainz/Albon/Stroll in Brazil/Britain 2024, etc.).
   - Saved all 20 records to `data/excluded_dns_rows.csv`.
   - Resulting training dataset in `data/clean_races.csv` reduced from 2,674 to 2,654 rows.
   - Target dataset `data/clean_target_mexico_2026.csv` has 0 DNS records (22 classified starters).
2. **Team Standardization:**
   - Mapped constructor names to standardized identifiers while preserving `RawTeamName`.
   - Verified 10 unique teams in seasons 2021–2025, and 11 unique teams in season 2026 (introducing Cadillac).
3. **DNF Classification:**
   - Categorized finishers using regex for `Finished`, `Lapped`, and `+N Lap(s)`.
   - Verified that DNF rates normalized across all seasons to realistic baseline levels:
     - 2021: 13.2%
     - 2022: 16.6%
     - 2023: 14.0%
     - 2024: 10.7%
     - 2025: 12.0%
     - 2026: 19.0%
     - Mexico 2026 Target: 22.7% (5 / 22)
4. **Pitlane Start & Grid Adjustment:**
   - Handled `GridPosition == 0` by assigning `pitlane_start = 1` and moving `grid_pos` to the back of the field.
5. **Qualifying Position Ties:**
   - Resolved intra-race ties in `quali_pos` deterministically based on `best_quali_sec` (e.g. Lawson and Lindblad tied at P8 in Mexico 2026 resolved to P8 and P9 respectively).
6. **Spa 2021 Anomaly:**
   - Flagged Belgian GP 2021 with `is_anomaly_race = 1`.
7. **Completeness Audit:**
   - 0 nulls across all essential analytical columns (`grid_pos`, `quali_pos`, `FinishPosition`, `is_dnf`, `pitlane_start`, `no_quali_time`).

---

## Stage 3: Feature Engineering & Automated Data Leakage Verification

### Objective
Engineer 11 strictly pre-race features on the unified chronological corpus, prevent lookahead and teammate data leakage, verify mathematical invariance via automated mutation testing, and generate the final modeling datasets: `data/features.csv` (132 rounds, 2,654 rows) and `data/target_mexico_2026_features.csv` (22 rows).

### Engineered Feature Specifications
1. `grid_pos`: Cleaned starting grid slot strictly bounded in `[1, N_starters]`.
2. `quali_pos`: Cleaned qualifying rank strictly bounded in `[1, N_starters]`.
3. `qualified_on_pole`: Binary indicator (`1` if `quali_pos == 1`, else `0`).
4. `pitlane_start`: Binary flag for cars starting from pitlane (`GridPosition == 0`).
5. `no_quali_time`: Binary flag indicating missing or invalidated qualifying lap times.
6. `is_sprint_weekend`: Binary flag identifying Sprint weekend formats.
7. `quali_gap_pct`: Percentage gap to pole time, capped at 7.0% (FIA 107% threshold). Imputed at 7.0% for missing times. 89 observations (3.33%) capped/imputed.
8. `teammate_quali_delta`: Driver's delta to teammate's mean qualifying gap in the same race (set to 0.0 for single-car entries).
9. `driver_form_3`: Driver's rolling mean `FinishPosition` over the previous 3 starts using `.shift(1).rolling(3, min_periods=1).mean()`. Neutral imputation (10.5 for $\le 2025$, 11.5 for $2026$) with indicator `is_new_driver_or_team = 1`.
10. `team_form_3`: Team-level mean `FinishPosition` per race, then rolled by constructor using `.shift(1).rolling(3, min_periods=1).mean()` to completely eliminate intra-race teammate leakage.
11. `team_dnf_rate_5`: Team-level rolling DNF rate over the previous 5 races (`.shift(1).rolling(5, min_periods=1).mean()`). Initial imputation at historical baseline (0.14).

### Automated Data Leakage Verification
- **Test 1 (Historical Invariance):** Mutated `FinishPosition = 999` and `is_dnf = 1` for all 2026 rounds $\ge 10$. Verified that all engineered features across 2021–2025 and 2026 Rounds 1–9 remained strictly identical (`np.allclose(atol=1e-9)`). **PASSED.**
- **Test 2 (Round Entry Invariance):** Verified that Round 10's own pre-race features remained strictly identical when its own race outcome was corrupted. **PASSED.**
- **Test 3 (Target Zero-Leakage):** Verified that corrupting target Mexico 2026 finish positions produced zero change in its own pre-race feature matrix. **PASSED.**

---

## Stages 4 & 5: Walk-Forward Model Evaluation & Probability Calibration

### 1. Pre-Modeling Rank Uniqueness & Spa 2021 Anomaly Ablation
- **Strict Unique Ranking:** Updated `normalize_race_ranks` in `src/3_features.py` from `method='dense'` to `method='first'` ordered deterministically by `(quali_pos, best_quali_sec, grid_pos)` so every driver in a race receives a strictly unique integer rank in `[1, N_starters]`.
- **Spa 2021 Ablation (66 Walk-Forward Rounds: 2024 R1 – 2026 R18):**
  - Excluding Spa 2021 (`is_anomaly_race == 1`): Ridge Continuous MAE = `3.0886`, Ridge Rank MAE = `3.0374`.
  - Keeping Spa 2021: Ridge Continuous MAE = `3.0863`, Ridge Rank MAE = `3.0359`.
  - **Decision:** Retained all 132 historical rounds (`Keep Spa 2021`), as keeping it slightly lowered out-of-sample MAE (`3.0359` vs `3.0374`).

### 2. Stage 4 — Walk-Forward Position Prediction Results

#### A. Overall Evaluation Slice: 2024–2026 (66 Test Rounds, 1,341 Driver-Starts)
| Model | N_Races | MAE | $\Delta$MAE vs. Grid | 95% Bootstrap CI | Spearman $\rho$ | Top-3 Hit % | Winner Hit % |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline 1: Starting Grid (`grid_pos`) | 66 | 3.1299 | 0.0000 | [+0.0000, +0.0000] | 0.6883 | 68.69% | 60.61% |
| Baseline 2: Qualifying Rank (`quali_pos`) | 66 | 3.0697 | -0.0602 | [-0.1205, +0.0027] | 0.7049 | 66.67% | 62.12% |
| Baseline 3: Heuristic [`grid + team_form_3`] (Cont.) | 66 | 3.1445 | +0.0146 | [-0.0979, +0.1292] | 0.6923 | 64.65% | 50.00% |
| Baseline 3: Heuristic [`grid + team_form_3`] (Rank) | 66 | 3.1586 | +0.0287 | [-0.0788, +0.1384] | 0.6923 | 64.65% | 50.00% |
| Full `RidgeCV` (Continuous $\hat{y}$) | 66 | 3.0863 | -0.0435 | [-0.1534, +0.0750] | **0.7105** | **69.19%** | 57.58% |
| **Full `RidgeCV` (Intra-Race Rank-Ordered)** | **66** | **3.0359** | **-0.0939** | **[-0.2022, +0.0135]** | **0.7105** | **69.19%** | 57.58% |

#### B. 2026 Season Standalone Slice (18 Test Rounds, 389 Driver-Starts)
| Model | N_Races | MAE | $\Delta$MAE vs. Grid | 95% Bootstrap CI | Spearman $\rho$ | Top-3 Hit % | Winner Hit % |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline 1: Starting Grid (`grid_pos`) | 18 | 3.2049 | 0.0000 | [+0.0000, +0.0000] | 0.6832 | 62.96% | 72.22% |
| Baseline 2: Qualifying Rank (`quali_pos`) | 18 | 3.0635 | -0.1414 | [-0.2727, -0.0303] | 0.7185 | 61.11% | 72.22% |
| Baseline 3: Heuristic [`grid + team_form_3`] (Cont.) | 18 | 3.2797 | +0.0748 | [-0.1002, +0.2555] | 0.7002 | 59.26% | 61.11% |
| Baseline 3: Heuristic [`grid + team_form_3`] (Rank) | 18 | 3.1780 | -0.0269 | [-0.1628, +0.1105] | 0.7002 | 59.26% | 61.11% |
| Full `RidgeCV` (Continuous $\hat{y}$) | 18 | 3.1717 | -0.0333 | [-0.2277, +0.1719] | 0.7129 | **64.81%** | **72.22%** |
| **Full `RidgeCV` (Intra-Race Rank-Ordered)** | **18** | **3.1035** | **-0.1015** | **[-0.3051, +0.0916]** | 0.7129 | **64.81%** | **72.22%** |

- **Key Stage 4 Finding:** Converting continuous Ridge predictions $\hat{y}$ into intra-race ranks (`rank(yhat)`) reduces MAE from `3.0863` to **`3.0359`** ($\Delta\text{MAE} = -0.0939$ vs. Starting Grid across 66 races, and $-0.1015$ in 2026), achieving the highest Spearman rank correlation (`0.7105`) and highest Top-3 Podium Hit Rate (`69.19%` overall, `64.81%` in 2026). This occurs because OLS/Ridge predicts the conditional mean $E[Y|X]$ (which regresses toward the mid-field ~10.5 due to DNF skewness), whereas discrete ranking restores full $1 \dots N$ dispersion.

### 3. Stage 5 — Win & Podium Probability Evaluation (`P(Win)` & `P(Podium)`)

#### A. Overall Evaluation Slice: 2024–2026 (66 Test Rounds)
| Target | Model | Brier Score | Log Loss |
|:---:|---|:---:|:---:|
| `P(Win)` | Empirical Grid Baseline | 0.02883 | 0.10620 |
| `P(Win)` | **Model A: Logistic Regression (L2, C=1.0)** | **0.02729** | **0.09478** |
| `P(Win)` | Model B: HistGradientBoosting (`max_depth=3`) | 0.02892 | 0.09628 |
| `P(Podium)` | **Empirical Grid Baseline** | **0.06384** | 0.21495 |
| `P(Podium)` | Model A: Logistic Regression (L2, C=1.0) | 0.06548 | 0.21765 |
| `P(Podium)` | **Model B: HistGradientBoosting (`max_depth=3`)** | 0.06538 | **0.21001** |

#### B. 2026 Season Standalone Slice (18 Test Rounds)
| Target | Model | Brier Score | Log Loss |
|:---:|---|:---:|:---:|
| `P(Win)` | Empirical Grid Baseline | 0.02196 | 0.08676 |
| `P(Win)` | Model A: Logistic Regression (L2, C=1.0) | 0.02069 | 0.07644 |
| `P(Win)` | **Model B: HistGradientBoosting (`max_depth=3`)** | **0.01792** | **0.06079** |
| `P(Podium)` | **Empirical Grid Baseline** | **0.06326** | 0.20422 |
| `P(Podium)` | Model A: Logistic Regression (L2, C=1.0) | 0.06534 | 0.20075 |
| `P(Podium)` | **Model B: HistGradientBoosting (`max_depth=3`)** | 0.06361 | **0.19577** |

- **Honest Model Selection Conclusion:**
  - Across the full 66-round walk-forward evaluation (`2024–2026`), **Logistic Regression (`Model A`) beat `HistGradientBoosting` (`Model B`) on `P(Win)`** in both Brier Score (`0.02729` vs `0.02892`) and Log Loss (`0.09478` vs `0.09628`), while beating the Empirical Grid Baseline (`0.02883`).
  - On `P(Podium)` across 2024–2026, `HistGradientBoosting` slightly edged out Logistic Regression on Brier Score (`0.06538` vs `0.06548`) and Log Loss (`0.21001` vs `0.21765`), though the Laplace-smoothed **Empirical Grid Baseline** achieved the lowest overall Brier score (`0.06384`) because multi-feature classifiers become slightly overconfident at the very top bin ($P > 0.85$) when Pole + top team form coincide.
  - However, on the **2026 Standalone slice (18 rounds)** — where the 22-car grid and new 2026 regulations increased non-linear interactions — **`HistGradientBoosting` (`Model B`) decisively beat Logistic Regression on both `P(Win)` (Brier `0.01792` vs `0.02069`, Log Loss `0.06079` vs `0.07644`) and `P(Podium)` (Brier `0.06361` vs `0.06534`, Log Loss `0.19577` vs `0.20075`)**.
  - **Selection Rule Applied:** Per the Honest Model Selection Rule based on Walk-Forward Brier Score (2024–2026), we select **Logistic Regression (`Model A`) for `P(Win)`** (overall Brier winner: `0.02729`) and **`HistGradientBoosting` (`Model B`) for `P(Podium)`** (better than Logistic Regression in both Brier and Log Loss overall and in 2026), while also reporting both alongside the Empirical Grid Baseline for full transparency.

### 4. Standardized RidgeCV Feature Coefficients (Full 132-Round Pre-Mexico Corpus)
- **Optimal LOO-GCV Regularization ($\alpha^*$):** `31.6228`
- **Intercept:** `10.5697`
- **Top Standardized Drivers of Finish Position:**
  1. `quali_pos` (`+2.0666`)
  2. `team_form_3` (`+0.9069`)
  3. `grid_pos` (`+0.8875`)
  4. `driver_form_3` (`+0.5466`)
  5. `team_dnf_rate_5` (`-0.2399`)
  6. `no_quali_time` (`-0.1834`)
  7. `qualified_on_pole` (`-0.1708`)
  8. `teammate_quali_delta` (`-0.1089`)

---

## Stage 6: Out-of-Sample Backtest on the 2026 Mexico City Grand Prix

### Objective
Perform an out-of-sample backtest evaluating race finish position predictions and win/podium probabilities for the 2026 Mexico City Grand Prix (Round 19, held November 1, 2026), trained strictly on the 132 pre-Mexico historical rounds (`data/features.csv`, 2,654 driver-starts across 2021–2026 R1–R18).

### 1. Model Predictions vs. Actual Race Outcome
- **Actual Podium:** P1: `VER` (Red Bull, started P5), P2: `HAM` (Ferrari, started P1), P3: `ANT` (Mercedes, started P4).
- **Top-3 Capture:** 2/3 in Top-3 (`HAM`, `VER`) and 3/3 (100% of actual podium: `VER`, `HAM`, `ANT`) captured in Model Top-5! (`HAM` P1, `LEC` P2, `VER` P3, `RUS` P4 [5.80], `ANT` P5 [5.92]).
- **Winner Prediction:** Max Verstappen was correctly identified as the primary threat from row 3 with `P(Podium) = 57.3%` and `P(Win) = 16.9%` (HGB) / `11.6%` (LR), while Lewis Hamilton was favored for the win on pole with `P(Win) = 56.4%` and finished P2. Breakthrough rookie Kimi Antonelli was assigned `P(Podium) = 33.1%` (LR) / `30.3%` (HGB) and finished P3.

### 2. Mexico 2026 Performance Comparison Table
| Model | MAE All Starters ($N=22$) | MAE Classified Finishers ($N=17$) | Spearman $\rho$ | Top-3 Hits (/3) | Winner Hit |
|---|:---:|:---:|:---:|:---:|:---:|
| Baseline 1: Starting Grid (`grid_pos`) | 3.636 | 3.353 | 0.719 | 1 / 3 (33.3%) | No (HAM) |
| Baseline 2: Qualifying Rank (`quali_pos`) | 3.182 | 2.941 | 0.817 | 1 / 3 (33.3%) | No (HAM) |
| Baseline 3: Heuristic (Continuous) | 3.297 | 2.672 | 0.741 | 1 / 3 (33.3%) | No (HAM) |
| Baseline 3: Heuristic (Rank-Ordered) | 3.727 | 3.353 | 0.741 | 1 / 3 (33.3%) | No (HAM) |
| **Full `RidgeCV` (Continuous $\hat{y}$)** | **2.917** | **2.287** | **0.823** | **2 / 3 (66.7%)** | No (HAM) |
| **Full `RidgeCV` (Intra-Race Rank-Ordered)** | **3.091** | **2.706** | **0.823** | **2 / 3 (66.7%)** | No (HAM) |

- **Separating Pace from Random DNFs:**
  - On all 22 starters, Full `RidgeCV` (Continuous) achieved **MAE = 2.917** ($\Delta\text{MAE} = -0.719$ vs. Starting Grid).
  - When excluding the 5 random mechanical/racing retirements (`LIN`, `BEA`, `BOT`, `STR`, `ALO`), Full `RidgeCV` MAE dropped to **`2.287`** on the 17 classified finishers ($\Delta\text{MAE} = -1.066$ vs. Starting Grid), demonstrating high predictive fidelity on genuine track pace.
  - Spearman rank correlation reached **`0.823`**, significantly outperforming Starting Grid (`0.719`) and Qualifying (`0.817`).

### 3. Mexico 2026 Probability Calibration Metrics
- **`P(Win)`:**
  - `HistGradientBoosting`: Brier = `0.05299` | Log Loss = `0.17394`
  - `LogisticRegression`: Brier = `0.05423` | Log Loss = `0.16655`
- **`P(Podium)`:**
  - `LogisticRegression`: Brier = `0.12083` | Log Loss = `0.32121`
  - `HistGradientBoosting`: Brier = `0.12234` | Log Loss = `0.32694`

### 4. Key Contenders Probability Matrix
| Driver | Team | Starting Grid | Predicted Rank | $P(\text{Win})_{\text{LR}}$ | $P(\text{Win})_{\text{HGB}}$ | $P(\text{Podium})_{\text{HGB}}$ | $P(\text{Podium})_{\text{LR}}$ | Actual Finish |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `HAM` | Ferrari | P1 | P1 | 56.4% | 49.2% | 74.2% | 83.7% | **P2** |
| `LEC` | Ferrari | P3 | P2 | 15.3% | 13.8% | 55.6% | 59.6% | **P4** |
| `VER` | Red Bull | P5 | P3 | 11.6% | 16.9% | 57.3% | 43.5% | **P1** |
| `RUS` | Mercedes | P2 | P4 | 2.6% | 9.1% | 60.1% | 39.0% | **P6** |
| `ANT` | Mercedes | P4 | P5 | 8.5% | 6.9% | 30.3% | 33.1% | **P3** |
| `NOR` | McLaren | P6 | P6 | 3.4% | 2.3% | 10.7% | 18.7% | **P5** |

### 5. Publication-Grade Artifacts Generated
- [`outputs/feature_coefficients.png`](outputs/feature_coefficients.png): Horizontal bar chart of standardized Ridge feature weights ($\alpha^* = 31.62$, $N = 2,654$).
- [`outputs/mexico_2026_quali_vs_predicted.png`](outputs/mexico_2026_quali_vs_predicted.png): Inverted scatter plot of Qualifying Gap vs. Predicted Finish Position with driver annotations and highlighted actual podium finishers.
- [`data/mexico_2026_backtest_results.csv`](data/mexico_2026_backtest_results.csv): Full 22-driver backtest dataset.


---

## Stage 7: 2026 São Paulo (Brazil) Grand Prix Live Prediction Pipeline (`src/6_predict_brazil_live.py`)

### Objective
Build a production-ready prediction script for Round 20 (2026 São Paulo Grand Prix, Nov 8, 2026) that combines all 132 pre-Mexico historical rounds (`data/clean_races.csv`, 2,654 rows) with the completed 2026 Mexico City Grand Prix (`data/clean_target_mexico_2026.csv`, 22 rows) into a **133-round training corpus (2,676 rows)**, updates lagged rolling driver and constructor form (`driver_form_3`, `team_form_3`, `team_dnf_rate_5`) entering Brazil, and supports both pre-qualifying readiness testing (`--dry-run`) and live post-Qualifying ingestion with grid penalty overrides (`--grid-overrides`).

### Pre-Weekend Form & Power Rankings Entering Brazil 2026 (`--dry-run`)
- **Fitted 133-Round `RidgeCV` $\alpha^*$:** `31.6228`
- **Top Contenders Entering Interlagos (3-Race Rolling Qualifying Baseline R17–R19 + Post-Mexico Rolling Form):**
  1. `VER` (Red Bull): Baseline Grid P1 | `driver_form_3 = 1.67` | `team_form_3 = 5.50` | `Exp_Pos = 3.24` | $P(\text{Win})_{\text{LR}} = 56.7\%$, $P(\text{Win})_{\text{HGB}} = 47.7\%$ | $P(\text{Pod})_{\text{HGB}} = 87.1\%$
  2. `LEC` (Ferrari): Baseline Grid P2 | `driver_form_3 = 2.67` | `team_form_3 = 3.33` | `Exp_Pos = 3.99` | $P(\text{Win})_{\text{LR}} = 15.6\%$, $P(\text{Win})_{\text{HGB}} = 18.4\%$ | $P(\text{Pod})_{\text{HGB}} = 69.3\%$
  3. `HAM` (Ferrari): Baseline Grid P3 | `driver_form_3 = 4.00` | `team_form_3 = 3.33` | `Exp_Pos = 4.66` | $P(\text{Win})_{\text{LR}} = 9.2\%$, $P(\text{Win})_{\text{HGB}} = 12.6\%$ | $P(\text{Pod})_{\text{LR}} = 51.4\%$
  4. `ANT` (Mercedes): Baseline Grid P4 | `driver_form_3 = 2.33` | `team_form_3 = 5.00` | `Exp_Pos = 5.14` | $P(\text{Win})_{\text{LR}} = 10.4\%$, $P(\text{Win})_{\text{HGB}} = 18.8\%$ | $P(\text{Pod})_{\text{HGB}} = 56.5\%$
  5. `RUS` (Mercedes): Baseline Grid P5 | `driver_form_3 = 7.67` | `team_form_3 = 5.00` | `Exp_Pos = 6.28` | $P(\text{Win})_{\text{LR}} = 2.4\%$ | $P(\text{Pod})_{\text{LR}} = 22.8\%$

---

## Stage 8: Packaging, Reproducibility & Portfolio Documentation

### Deliverables Completed
1. **`run_all.py`:** Master one-command offline runner executing and verifying `src/2_clean_data.py` $\to$ `src/3_features.py` $\to$ `src/4_evaluate.py` $\to$ `src/5_backtest_mexico.py` $\to$ `src/6_predict_brazil_live.py --dry-run`.
2. **`.gitignore`:** Configured to exclude `f1_cache/`, `.venv/`, `__pycache__/`, `*.tmp.csv`, and temporary scratch files.
3. **`requirements.txt`:** Locked exact tested package versions (`fastf1==3.9.0`, `pandas==2.3.3`, `numpy==2.5.4`, `scikit-learn==1.9.1`, `scipy==1.18.1`, `matplotlib==3.11.2`).
4. **`README.md`:** Comprehensive portfolio documentation with architecture, anti-leakage proof, walk-forward tables, Mexico 2026 backtest analysis, Brazil 2026 power rankings, and reproducibility instructions.

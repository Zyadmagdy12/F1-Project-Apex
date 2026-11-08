# PROJECT APEX — IMPLEMENTATION LOG

All implementation steps, architectural decisions, bug discoveries, and verification checkpoints are logged chronologically below.

---

## Checkpoint 0 & 1 — Repository Audit & Data Provenance
- **Files Inspected:** `src/2_clean_data.py`, `src/3_features.py`, `src/4_train_and_eval.py`, `src/6_predict_brazil_live.py`, `data/raw_all_races_data.csv`, `data/clean_races.csv`, `data/clean_target_mexico_2026.csv`, `data/excluded_dns_rows.csv`, `data/brazil_2026_actual_quali.csv`, `data/brazil_2026_live_predictions.csv`.
- **Findings:**
  - `data/clean_races.csv` contains 132 historical Grand Prix rounds (`2021 R1 Bahrain` through `2026 R18 Singapore`, `2,654` starter rows).
  - `data/clean_target_mexico_2026.csv` contains Round 19 (`2026 Mexico City GP`, `22` starter rows).
  - `data/brazil_2026_actual_quali.csv` contains Round 20 (`2026 São Paulo GP`, `22` official qualifying and penalty-adjusted starting grid rows).
  - `data/excluded_dns_rows.csv` contains `20` Did Not Start / Withdrawn entries excluded prior to race start.
  - Across the 133 completed rounds (`2,676` rows), `2,297` are classified finishers (`Finished`, `Lapped`, `+N Laps`), `11` are post-race disqualifications (`Disqualified`), and `368` are on-track retirements (`is_dnf = 1`), consisting of `64` attributed mechanical failures (2021–2023), `69` attributed collision/spun-off incidents (2021–2023), and `235` generic `'Retired'` rows (2023–2026).
- **Preservation Rule:** All legacy scripts and `data/*.csv` files were left completely untouched.

## Checkpoint 2 — Leak-Free Feature Engineering (`src/apex_features.py`)
- **Created:** `src/apex_features.py` implementing `engineer_apex_features()` and `get_feature_dictionary()`.
- **Features Engineered:**
  - Position/Ranking feature vector (`APEX_POSITION_FEATURES`, 10 features): `Qualifying_Rank`, `Starting_Grid_Position`, `Delta_Q_to_G`, `Teammate_Qualifying_Delta`, `Driver_Form`, `Team_Form`, `Driver_Track_History`, `Driver_Track_Missing`, `Circuit_Overtake_Difficulty_Index`, `Circuit_Attrition_Index`.
  - DNF Hazard feature vector (`APEX_DNF_FEATURES`, 10 features): `Starting_Grid_Position`, `Qualifying_Rank`, `Delta_Q_to_G`, `Constructor_DNF_Rate_Smoothed`, `Constructor_DNF_EWMA`, `Constructor_Attributed_Mech_Rate`, `Driver_DNF_Rate_Smoothed`, `Driver_DNF_EWMA`, `Driver_Attributed_Incident_Rate`, `Circuit_Attrition_Index`.
- **Design Decision on `Teammate_Qualifying_Delta`:** Instead of silently filling the 20 solo-starter rows (where a teammate had a DNS) with `0.0` (which falsely implies equal qualifying pace), `engineer_apex_features` leaves those 20 rows as `NaN`, sets `Teammate_Missing_Flag = 1`, and imputes via fold-fitted `SimpleImputer(strategy='median')` inside each model pipeline.

## Checkpoint 3 & 4 — Monotonic HGB, DNF Hazard Classifier, Plackett-Luce & Conformal Ridge (`src/apex_models.py`)
- **Created:** `src/apex_models.py`.
- **`MonotonicHGBClassifier`:** Dynamically builds `monotonic_cst` from feature names (`Qualifying_Rank: -1`, `Starting_Grid_Position: -1`, all others `0`), verifies `1 in model.classes_`, and supports automated sweep testing via `run_monotonicity_sweep_test`.
- **`CalibratedDNFHazardModel`:** Fits an L2-regularized Logistic + shallow HGB hazard blend and compares Platt (Sigmoid) vs. Isotonic calibration strictly on past expanding-window out-of-fold predictions.
- **`PlackettLuceRankingModel`:** Fits L2-regularized Plackett-Luce listwise MLE (`scipy.optimize.minimize(..., method='L-BFGS-B', jac=True)`) on classified historical finishing orders (`is_classified_finisher == 1`), placing unclassified retirements only in the initial risk pool without inventing relative finishing orderings among DNFs. Integrates calibrated `P_DNF` via $S = 25,000$ Gumbel-Max Monte Carlo rollouts (`seed=42`) over stratified Bernoulli survival realizations.
- **`LeakageSafeRidgeConformalModel`:** Wraps `SimpleImputer` + `StandardScaler` + `RidgeCV` and computes finite-sample 90% split/walk-forward conformal prediction intervals using strictly prior out-of-sample residuals (`q_level = ceil((n + 1) * 0.90) / n`, `method='higher'`).
- **`compute_exact_interventional_shap`:** Enumerates all $2^{10} = 1,024$ coalitions over $K = 80$ fixed historical pre-race background rows (`seed=42`) in a single vectorized batch call per driver-model pair, achieving exact Shapley additivity ($\max |\text{error}| = 8.88 \times 10^{-16}$).

## Checkpoint 5, 6, 7 & 8 — Master Execution Pipeline (`src/apex_runner.py`)
- **Created & Executed:** `src/apex_runner.py` (completed in `283.80s`).
- **Refinement During Verification:**
  - Initially, `src/apex_runner.py` attempted to reproduce the legacy 12-feature `40.24%` Verstappen artifact using the Phase 2 `df_feat` dataframe (where solo-starter `teammate_quali_delta` rows were `NaN` rather than `0.0`). Updated `src/apex_runner.py` to invoke the legacy `3_features.compute_features_pipeline(hist_133, braz_raw)` for the legacy reproduction comparison, reproducing Verstappen's legacy P21 probability (`0.360629` raw / `0.402358` calibrated) to 6 decimal places.
  - Replaced iterative column insertion of dynamic position columns (`P_Uncond_Pos_1..22`, `P_Cond_Pos_1..22`, `P_Cond_Given_Finish_Pos_1..22`) with `pd.concat` to prevent DataFrame fragmentation warnings.

## Checkpoint 9 & 10 — Export Verification & Automated Acceptance Tests (`tests/test_project_apex.py`)
- **Created & Executed:** `tests/test_project_apex.py`.
- **Result:** All 11 automated test suites (`test_01_temporal_leakage` through `test_11_export_integrity`) passed with `0` errors (`11/11 PASSED`).

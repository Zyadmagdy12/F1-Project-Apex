# PROJECT APEX — MASTER EXECUTION TODO & VERIFICATION TRACKER

Status Legend:
- `[ ]` Pending
- `[-]` In Progress
- `[x]` COMPLETED AND VERIFIED (Backed by executed code, saved outputs, and passing automated tests in `EVIDENCE_LOG.md`)

---

## Phase -1 & Phase 0: Project Audit, Data Provenance & Governance
- [x] **0.1** Inspect repository structure, scripts (`src/2_clean_data.py`–`src/6_predict_brazil_live.py`), and source datasets (`data/*.csv`).
- [x] **0.2** Verify data provenance, season coverage (132 historical rounds + R19 Mexico + R20 São Paulo), DNS exclusions (20 rows), DSQ separations (11 rows), and status taxonomy (`Finished`, `Lapped`, attributed 2021–2023 retirements vs. generic 2023–2026 `Retired`).
- [x] **0.3** Create and maintain `TODO.md`, `EXECUTION_PLAN.md`, `IMPLEMENTATION_LOG.md`, and `EVIDENCE_LOG.md`.

## Phase 1: Fix Non-Monotonic HistGradientBoosting Artifact
- [x] **1.1** Refactor `HistGradientBoostingClassifier` in `src/apex_models.py` (`MonotonicHGBClassifier`) with dynamic column-name-mapped `monotonic_cst` (`Qualifying_Rank: -1`, `Starting_Grid_Position: -1`, all other features: `0`).
- [x] **1.2** Verify positive-class (`1`) probability indexing and race-level probability calibration/normalization.
- [x] **1.3** Implement and run automated monotonicity sweep verification (`run_monotonicity_sweep_test`) across positions $1\dots22$ for both `Qualifying_Rank` and `Starting_Grid_Position` (`0` violations across `924` transitions per race).
- [x] **1.4** Reproduce and document the Verstappen P21 `40.24%` (`0.402358` calibrated / `0.360629` raw) podium artifact in the unconstrained legacy model vs. the monotonic model (`0.319676` legacy 12-feature monotonic / `0.269764` Apex 10-feature monotonic / `0.111480` Plackett-Luce / `0.009013` LogisticRegression).

## Phase 2: Expanded Leak-Free Feature Engineering (`src/apex_features.py`)
- [x] **2.1** Preserve and validate core predictors (`Qualifying_Rank`, `Starting_Grid_Position`, `Team_Form`, `Driver_Form`, plus 3-race rolling references).
- [x] **2.2** Engineer `Teammate_Qualifying_Delta` (`%` gap to primary teammate in same session, clipped to `[-5.0%, +5.0%]`, with explicit `Teammate_Missing_Flag` and fold-safe median imputation instead of silent `0.0` replacement).
- [x] **2.3** Engineer `Circuit_Overtake_Difficulty_Index` strictly from prior races at each circuit (excluding DNFs/DSQs, blended with global prior when prior circuit history $<3$ races).
- [x] **2.4** Engineer `Circuit_Attrition_Index` strictly from prior races at each circuit (smoothed with global empirical DNF prior).
- [x] **2.5** Engineer `Driver_Track_History` (recency-weighted, field-size-normalized prior finishing percentile at the circuit, shrunk toward `0.50` with `Driver_Track_Starts` and `Driver_Track_Missing` indicators).
- [x] **2.6** Engineer `Delta_Q_to_G = Starting_Grid_Position - Qualifying_Rank` to isolate grid penalties and promotions from raw qualifying speed.

## Phase 3: Dedicated Calibrated DNF Hazard Classifier (`src/apex_models.py`)
- [x] **3.1** Separate `is_dsq` (`Disqualified`, 11 rows) from on-track `is_dnf` retirements (`368` rows) and document the API status limitation (`235` generic `'Retired'` rows in 2023–2026 vs. `64` attributed mechanical and `69` attributed incident rows in 2021–2023).
- [x] **3.2** Engineer constructor, driver, and circuit attrition features (`Constructor_DNF_Rate_Smoothed`, `Constructor_DNF_EWMA`, `Constructor_Attributed_Mech_Rate`, `Driver_DNF_Rate_Smoothed`, `Driver_DNF_EWMA`, `Driver_Attributed_Incident_Rate`, `Circuit_Attrition_Index`).
- [x] **3.3** Implement `CalibratedDNFHazardModel` comparing Platt (Sigmoid) vs. Isotonic calibration strictly on past out-of-fold predictions (`Brier Platt = 0.110873`, `Brier Isotonic = 0.110790` across 66 walk-forward races).

## Phase 4: Listwise Plackett-Luce Ranking Model (`src/apex_models.py`)
- [x] **4.1** Implement L2-regularized Plackett-Luce MLE (`PlackettLuceRankingModel`) trained on classified historical finishing orders without inventing orderings among unclassified DNFs.
- [x] **4.2** Integrate calibrated `P_DNF` with Plackett-Luce via $S = 25,000$ Gumbel-Max Monte Carlo rollouts (`seed=42`) over stratified survival realizations.
- [x] **4.3** Export and verify both **Unconditional** (`P_DNF + sum(P_Uncond_Pos_1..N) = 1.0`) and **Conditional** (`sum_drivers(P_Cond_Pos_k) = 1.0` and `sum_pos(P_Cond_Given_Finish_Pos_k) = 1.0`) position probability matrices, plus `P_Win_PL` and `P_Podium_PL`.

## Phase 5: RidgeCV Refactor & Walk-Forward Conformal Intervals (`src/apex_models.py`)
- [x] **5.1** Wrap `SimpleImputer`, `StandardScaler`, and `RidgeCV` inside `LeakageSafeRidgeConformalModel`.
- [x] **5.2** Compute 90% finite-sample split/walk-forward conformal intervals strictly from prior out-of-sample walk-forward residuals (`1,776` residuals before Mexico 2026; `1,798` residuals before São Paulo 2026).
- [x] **5.3** Validate empirical coverage across all 66 walk-forward races (`90.16%` continuous / `92.54%` integer rank bounds), by season (`2024`, `2025`, `2026`), and by starting-grid bucket.

## Phase 6: Exact Interventional SHAP Explanations (`src/apex_models.py`)
- [x] **6.1** Implement `compute_exact_interventional_shap` via full $2^{10} = 1,024$ coalition enumeration over $K = 80$ pre-race historical background rows (`seed=42`).
- [x] **6.2** Compute exact Shapley values for all 22 drivers in both São Paulo 2026 and Mexico 2026 across all 7 models (`308` driver-model explanations, max additivity error $8.88 \times 10^{-16}$).
- [x] **6.3** Export dedicated Max Verstappen P21 case study (`outputs/project_apex/verstappen_shap_case.json`).

## Phase 7: 66-Race Walk-Forward Validation & Bootstrap CIs (`src/apex_runner.py`)
- [x] **7.1** Execute expanding-window chronological walk-forward validation across all 66 eligible 2024–2026 Grand Prix races (`2024 R1` to `2026 R18`, $N = 1,341$ starts).
- [x] **7.2** Compute race-level Spearman $\rho$, Continuous MAE, Rank MAE, Top-3 Hit Rate, Brier Score, Log Loss, and 5-bin calibration reliability curves against the Starting Grid baseline (including a calibrated Grid-only probability model).
- [x] **7.3** Compute 10,000-replicate paired race-cluster bootstrap 95% confidence intervals (`seed=42`) for all headline metric differentials.

## Phase 8, 9 & 10: Event Forecasts, Structured Exports & Automated Acceptance Suite
- [x] **8.1 & 9.1–9.7** Export all 10 required CSV and JSON files into `outputs/project_apex/`:
  - `sao_paulo_predictions.csv` & `sao_paulo_predictions.json`
  - `mexico_backtest.csv` & `mexico_backtest.json`
  - `walk_forward_metrics.json`
  - `verstappen_shap_case.json`
  - `model_manifest.json`, `feature_dictionary.json`, `data_quality_report.json`, `run_report.json`
- [x] **10.1** Execute `tests/test_project_apex.py` and verify all 11 acceptance tests pass (`11/11 PASSED`).


---

## Phase 11: Downstream 4K Visualization Suite (`VISUALIZATION_TODO.md`)
- [x] **11.0** Pre-rendering data audit & visualization execution plan (`VISUALIZATION_AUDIT.md`, `VISUALIZATION_PLAN.md`, `VISUALIZATION_TODO.md`, `VISUALIZATION_RUN_LOG.md`).
- [x] **11.1** Panel 1: São Paulo Prediction Center (`outputs/project_apex/panels/panel_01_sao_paulo_prediction_center.png`).
- [x] **11.2** Panel 2: Multi-Model Probability Comparison (`outputs/project_apex/panels/panel_02_multi_model_probability_comparison.png`).
- [x] **11.3** Panel 3: Mexico GP Backtest (`outputs/project_apex/panels/panel_03_mexico_gp_backtest.png`).
- [x] **11.4** Panel 4: 66-Race Model Evaluation (`outputs/project_apex/panels/panel_04_66_race_model_evaluation.png`).
- [x] **11.5** Panel 5: Verstappen Recovery and SHAP Audit (`outputs/project_apex/panels/panel_05_verstappen_recovery_and_shap_audit.png`).
- [x] **11.6** Panel 6: Full Position Probability and DNF Heatmap (`outputs/project_apex/panels/panel_06_full_position_probability_and_dnf_heatmap.png`).
- [x] **11.7** Automated post-render verification (`tests/test_apex_visualizations.py` — `6/6 PASSED`).

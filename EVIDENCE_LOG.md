# PROJECT APEX — EVIDENCE & VERIFICATION LOG

Every task marked `[x]` in `TODO.md` is backed by the executed evidence entries below.

---

## `[HGB-01]` Monotonic HistGradientBoosting & Verstappen P21 Artifact Audit
- **Source Code:** `src/apex_models.py` (`MonotonicHGBClassifier`, `build_monotonic_constraint_vector`, `run_monotonicity_sweep_test`), `tests/test_project_apex.py` (`test_03_monotonic_hgb`, `test_04_verstappen_artifact`).
- **Exported Evidence:** `outputs/project_apex/verstappen_shap_case.json`, `outputs/project_apex/sao_paulo_predictions.json`.
- **Executed Test Output:**
  - `Monotonic_HGB_Sweep_Verification`:
    - `sao_paulo_win_violations`: `0` / `924` transitions
    - `sao_paulo_podium_violations`: `0` / `924` transitions
    - `mexico_win_violations`: `0` / `924` transitions
    - `mexico_podium_violations`: `0` / `924` transitions
    - Diagnostic Unconstrained HGB Podium violations on São Paulo 2026: `16` violations (`max_positive_jump = +0.054354`).
  - `Verstappen_P21_Artifact_Audit`:
    - Legacy 12-feature Unconstrained HGB (São Paulo P21): `0.360629` raw / `0.402358` (`40.24%`) calibrated (`legacy_40_24_pct_reproduced = true`).
    - Legacy 12-feature Monotonic HGB (São Paulo P21): `0.270396` raw / `0.319676` (`31.97%`) calibrated.
    - Project Apex 10-feature Monotonic HGB (São Paulo P21): `0.292741` raw / `0.269764` (`26.98%`) calibrated (`max_grid_sweep_positive_jump = 0.0`).
    - Project Apex Integrated Plackett-Luce (São Paulo P21): `P_Podium_PL = 0.111480` (`11.15%`), `P_Win_PL = 0.034280` (`3.43%`).
    - Project Apex Calibrated LogisticRegression (São Paulo P21): `P_Podium_LR = 0.009013` (`0.90%`), `P_Win_LR = 0.000085` (`0.01%`).
- **Status:** `PASS`

---

## `[FEAT-01]` Leak-Free Feature Engineering & Temporal Mutation Verification
- **Source Code:** `src/apex_features.py` (`engineer_apex_features`), `tests/test_project_apex.py` (`test_01_temporal_leakage`, `test_02_race_grouping`).
- **Exported Evidence:** `outputs/project_apex/feature_dictionary.json`, `outputs/project_apex/data_quality_report.json`.
- **Executed Test Output:**
  - `Temporal_Leakage_Mutation_Test`: Mutating all 2026 race outcomes (`FinishPosition = 23 - FinishPosition`, `Status = 'Retired'`) produced `max_abs_diff_on_prior_races = 0.0` across all 2025 races and 2026 Round 1 pre-race features.
  - `Race_Grouping_Integrity`: `66` unique chronological race folds verified (`n_entrants >= 18` in every fold; zero intra-race splits).
- **Status:** `PASS`

---

## `[DNF-01]` Out-of-Sample Calibrated DNF Hazard Classifier
- **Source Code:** `src/apex_models.py` (`CalibratedDNFHazardModel`), `tests/test_project_apex.py` (`test_05_dnf_calibration`).
- **Exported Evidence:** `outputs/project_apex/walk_forward_metrics.json` (`brier_scores_all_1341_starts.dnf_probability`, `calibration_curve_bins.Calibrated_DNF_Platt`, `dnf_performance_by_constructor`).
- **Executed Test Output:**
  - 66-race out-of-sample Brier Score (`1,341` starts):
    - Raw Uncalibrated DNF Model: `0.110552`
    - Platt (Sigmoid) Calibrated DNF: `0.110873` (`LogLoss = 0.378022`)
    - Isotonic Calibrated DNF: `0.110790` (`LogLoss = 0.382028`)
    - Calibrated Grid-Only Baseline: `0.110194`
  - Pre-São Paulo inner OOF selector (`n_oof = 1,750`): Platt Brier `0.130495` (`LogLoss = 0.427259`) vs. Isotonic Brier `0.130956` (`LogLoss = 0.448009`); `platt_sigmoid` selected as primary smooth calibrator.
- **Status:** `PASS`

---

## `[PL-01]` Listwise Plackett-Luce Ranking & Mathematical Coherence Verification
- **Source Code:** `src/apex_models.py` (`PlackettLuceRankingModel`), `tests/test_project_apex.py` (`test_06_plackett_luce_coherence`).
- **Exported Evidence:** `outputs/project_apex/sao_paulo_predictions.csv`, `outputs/project_apex/mexico_backtest.csv`, `outputs/project_apex/sao_paulo_predictions.json` (`plackett_luce_coherence_verification`).
- **Executed Test Output ($S = 25,000$ simulations, `seed=42`, max MC SE $= 0.002193$):**
  - Unconditional row sum identity (`max |P_DNF_i + sum_k P_Uncond_Pos_k - 1.0|`): `0.0` (exact within CSV rounding `< 1e-6`).
  - Unconditional column sum identity (`max |sum_i P_Uncond_Pos_k - P(N_finishers >= k)|`): `0.0`.
  - Conditional position column sum identity (`max |sum_i P_Cond_Pos_k - 1.0|`): `< 1e-6`.
  - Conditional finisher row sum identity (`max |sum_k P_Cond_Given_Finish_Pos_k - 1.0|`): `< 1e-6`.
- **Status:** `PASS`

---

## `[CONF-01]` Walk-Forward 90% Conformal Prediction Intervals
- **Source Code:** `src/apex_models.py` (`LeakageSafeRidgeConformalModel`), `tests/test_project_apex.py` (`test_07_conformal_intervals`).
- **Exported Evidence:** `outputs/project_apex/walk_forward_metrics.json` (`conformal_interval_coverage_validation`), `outputs/project_apex/sao_paulo_predictions.csv`, `outputs/project_apex/mexico_backtest.csv`.
- **Executed Test Output:**
  - Strictly prior out-of-sample calibration pool sizes: `1,776` prior driver-starts for Mexico 2026 (`q_hat = 6.2700`), `1,798` prior driver-starts for São Paulo 2026 (`q_hat = 6.2729`).
  - 66-race empirical coverage (`1,341` starts, nominal `0.90`):
    - Continuous interval `[Lower_Bound_Cont, Upper_Bound_Cont]`: `90.16%` (`0.9016`), mean width `11.54` positions.
    - Integer rank interval `[Lower_Bound_Rank, Upper_Bound_Rank]`: `92.54%` (`0.9254`), mean width `12.25` positions.
    - Recent 10 races integer rank coverage: `92.73%` (`0.9273`).
    - Mexico 2026 backtest coverage: `90.91%` continuous (`20/22`), `95.45%` rank bounds (`21/22`).
- **Status:** `PASS`

---

## `[SHAP-01]` Exact Interventional Shapley Explanations ($2^{10} = 1,024$ Coalitions)
- **Source Code:** `src/apex_models.py` (`compute_exact_interventional_shap`), `tests/test_project_apex.py` (`test_08_shap_additivity`).
- **Exported Evidence:** `outputs/project_apex/verstappen_shap_case.json`, `outputs/project_apex/sao_paulo_predictions.json`, `outputs/project_apex/mexico_backtest.json`.
- **Executed Test Output:**
  - Verified `308` driver-model explanations (`22` drivers $\times$ `7` models $\times$ `2` events).
  - Maximum additivity error $\max |\text{Baseline} + \sum_j \phi_j - f(\mathbf{x})| = 8.881784 \times 10^{-16}$ (`< 1e-6` threshold).
- **Status:** `PASS`

---

## `[WF-01]` 66-Race Walk-Forward Evaluation & 10,000-Replicate Cluster Bootstrap CIs
- **Source Code:** `src/apex_runner.py` (`run_walk_forward_validation`, `compute_paired_cluster_bootstrap_ci`), `tests/test_project_apex.py` (`test_09_walk_forward_validation`, `test_10_bootstrap_ci`).
- **Exported Evidence:** `outputs/project_apex/walk_forward_metrics.json`.
- **Executed Test Output (`66` races, `1,341` starts, `B = 10,000` cluster bootstrap replicates, `seed=42`):**
  - Spearman $\rho$ vs. Starting Grid (`0.6883`):
    - `RidgeCV`: `0.7102` ($\Delta = +0.0219$, 95% CI `[+0.0053, +0.0392]`, significant)
    - `Integrated DNF + Plackett-Luce`: `0.7117` ($\Delta = +0.0233$, 95% CI `[+0.0087, +0.0390]`, significant)
  - MAE vs. Starting Grid (`3.1622`):
    - `RidgeCV` Continuous MAE: `3.0369` ($\Delta = +0.1253$, 95% CI `[+0.0043, +0.2401]`, significant)
    - `RidgeCV` Rank MAE: `3.0481` ($\Delta = +0.1141$, 95% CI `[+0.0022, +0.2263]`, significant)
    - `Integrated DNF + Plackett-Luce` Rank MAE: `3.0277` ($\Delta = +0.1345$, 95% CI `[+0.0297, +0.2381]`, significant)
  - Win Brier Score vs. Calibrated Grid-Only Baseline (`0.031881` race-mean / `0.031710` pooled):
    - `Monotonic_HGB`: `0.028094` race-mean / `0.027827` pooled ($\Delta = +0.003788$, 95% CI `[+0.000556, +0.006898]`, significant)
- **Status:** `PASS`

---

## `[EXP-01]` Export Schema & JSON Serialization Integrity
- **Source Code:** `tests/test_project_apex.py` (`test_11_export_integrity`).
- **Exported Evidence:** All 10 files in `outputs/project_apex/`.
- **Executed Test Output:**
  - Verified all 10 files exist, are non-empty, parse cleanly with standard `json.loads` / `pd.read_csv`, contain zero `NaN`/`Infinity` tokens, have exactly `22` unique drivers in both `sao_paulo_predictions.csv` and `mexico_backtest.csv`, and satisfy all probability range and sum identities.
- **Status:** `PASS`

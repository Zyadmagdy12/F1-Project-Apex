# PROJECT APEX v2.0 — VISUALIZATION TODO & ACCEPTANCE TRACKER (`VISUALIZATION_TODO.md`)

Status Legend:
- `[ ]` Pending
- `[-]` In Progress
- `[x]` COMPLETED AND VERIFIED

---

## Stage V0: Pre-Rendering Data Audit & Planning
- [x] **V0.1** Review `TODO.md`, `EXECUTION_PLAN.md`, `IMPLEMENTATION_LOG.md`, `EVIDENCE_LOG.md`, and all 10 files in `outputs/project_apex/`.
- [x] **V0.2** Audit all 22 drivers, 66 walk-forward races, feature columns, exact SHAP additivity (`8.88e-16`), cross-model probability independence, and DNF/Plackett-Luce probability semantics (`VISUALIZATION_AUDIT.md`).
- [x] **V0.3** Create `VISUALIZATION_PLAN.md`, `VISUALIZATION_TODO.md`, and `VISUALIZATION_RUN_LOG.md`, and extend `TODO.md`.

## Stage V1: Implement & Render the Six 4K (3840x2160, 16:9) Panels (`src/apex_visualize.py`)
- [x] **V1.1 — Panel 1 (`panel_01_sao_paulo_prediction_center.png`):** São Paulo GP 2026 Prediction Center (All 22 drivers, `Quali -> Grid`, $\Delta_{Q\to G}$, `RidgeCV` `Exp_Finish`, 90% Conformal Rank & Continuous intervals, `P_Win_HGB`, `P_Podium_HGB`, `P_Win_PL`, `P_Podium_PL`, `P_DNF`, and Grid-to-Finish interval chart). **[COMPLETED AND VERIFIED]**
- [x] **V1.2 — Panel 2 (`panel_02_multi_model_probability_comparison.png`):** Multi-Model Probability Comparison (`Monotonic_HGB` vs. `LogisticRegression` vs. `Plackett-Luce` Win & Podium probabilities, raw vs. calibrated markers, `0/924` Monotonicity Sweep Audit vs. `16` unconstrained violations, and cross-model spread table). **[COMPLETED AND VERIFIED]**
- [x] **V1.3 — Panel 3 (`panel_03_mexico_gp_backtest.png`):** Mexico City GP 2026 Historical Backtest (All 22 drivers, Predicted vs. Actual Finish, 90% Conformal Coverage `95.45%` rank / `90.91%` cont, Spearman $\rho = 0.8430$ vs. Grid `0.7188`, Continuous MAE `2.8386`, and `2/3` Podium Overlap reconciliation). **[COMPLETED AND VERIFIED]**
- [x] **V1.4 — Panel 4 (`panel_04_66_race_model_evaluation.png`):** 66-Race Walk-Forward Model Evaluation (`1,341` starts, `10,000`-replicate paired race-cluster bootstrap 95% CIs, 5-bin calibration reliability curves, conformal coverage by season/grid bucket, and constructor DNF Brier performance). **[COMPLETED AND VERIFIED]**
- [x] **V1.5 — Panel 5 (`panel_05_verstappen_recovery_and_shap_audit.png`):** Verstappen P21 Recovery & Exact SHAP Audit (`2^10 = 1,024` coalitions exact Interventional SHAP waterfalls for `Monotonic_HGB` Podium and `RidgeCV` Expected Finish, legacy `40.24%` artifact reproduction vs. monotonic sweep curve, and P1–P22 Plackett-Luce unconditional vs. conditional survival distribution). **[COMPLETED AND VERIFIED]**
- [x] **V1.6 — Panel 6 (`panel_06_full_position_probability_and_dnf_heatmap.png`):** Full $22 \times 23$ Position Probability & DNF Heatmap (`P_Uncond_Pos_1..22` + `P_DNF = 100.0%` row identity, column survival probability $P(N_{\text{finishers}} \ge k)$, summary expected rank/win/podium columns, and explicit FIA vs. simulation DNF semantics). **[COMPLETED AND VERIFIED]**

## Stage V2: Automated Post-Render Verification (`tests/test_apex_visualizations.py`)
- [x] **V2.1** Verify all 6 PNG files exist, are non-empty, and have exact 4K 16:9 resolution (`3840 x 2160` pixels). **[COMPLETED AND VERIFIED]**
- [x] **V2.2** Verify every displayed number, driver ordering, probability definition, SHAP additivity, and heatmap row/column sum against `outputs/project_apex/`. **[COMPLETED AND VERIFIED]**
- [x] **V2.3** Update `VISUALIZATION_TODO.md`, `TODO.md`, and `VISUALIZATION_RUN_LOG.md` to `[x] COMPLETED AND VERIFIED`. **[COMPLETED AND VERIFIED]**

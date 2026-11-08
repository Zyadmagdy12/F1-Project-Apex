# PROJECT APEX v2.0 — VISUALIZATION RUN LOG (`VISUALIZATION_RUN_LOG.md`)

## Checkpoint V0 — Data Audit & Plan Initialization (`2026-11-08T05:41:00+02:00`)
- Inspected `TODO.md`, `EXECUTION_PLAN.md`, `IMPLEMENTATION_LOG.md`, `EVIDENCE_LOG.md`, and all 10 files in `outputs/project_apex/`.
- Verified:
  - `sao_paulo_predictions.csv` (`22` rows, `122` columns) and `sao_paulo_predictions.json`
  - `mexico_backtest.csv` (`22` rows, `132` columns) and `mexico_backtest.json`
  - `walk_forward_metrics.json` (`66` chronological races, `1,341` starts, `10,000`-replicate paired cluster bootstrap CIs)
  - `verstappen_shap_case.json` (exact $2^{10}=1,024$ coalition Interventional SHAP explanations, additivity error `0.0` / `8.88e-16`)
- Created `VISUALIZATION_AUDIT.md`, `VISUALIZATION_PLAN.md`, and `VISUALIZATION_TODO.md`.
- Next step: Implement `src/apex_visualize.py` and render all 6 4K (`3840x2160`, 16:9) panels into `outputs/project_apex/panels/`.


## Checkpoint V1 — 4K Panel Rendering & Typography Refinement (`2026-11-08T06:08:57+02:00`)
- Executed `src/apex_visualize.py` (`3840x2160` px, `16:9` aspect ratio, `200 DPI` on `19.2" x 10.8"` canvas).
- Visually inspected all 6 rendered PNG panels via `read_file` and refined column coordinates, bar label offsets, horizontal bar left margins (`ax_boot`, `ax_sh1`, `ax_sh2`), and footer card typography so zero text clipping or overlap exists on any panel.
- Rendered panels written to `outputs/project_apex/panels/` and mirrored in `outputs/project_apex/`:
  1. `panel_01_sao_paulo_prediction_center.png` (`3840x2160` px, `898,090` bytes, `std=0.1437`)
  2. `panel_02_multi_model_probability_comparison.png` (`3840x2160` px, `688,673` bytes, `std=0.1777`)
  3. `panel_03_mexico_gp_backtest.png` (`3840x2160` px, `797,343` bytes, `std=0.1506`)
  4. `panel_04_66_race_model_evaluation.png` (`3840x2160` px, `705,435` bytes, `std=0.1891`)
  5. `panel_05_verstappen_recovery_and_shap_audit.png` (`3840x2160` px, `588,805` bytes, `std=0.1637`)
  6. `panel_06_full_position_probability_and_dnf_heatmap.png` (`3840x2160` px, `744,436` bytes, `std=0.1678`)

## Checkpoint V2 — Automated Verification Suite (`tests/test_apex_visualizations.py`)
- Executed `.\.venv\Scripts\python.exe tests/test_apex_visualizations.py` -> **All 6/6 panels PASSED** all dimensional, pixel-contrast, data-integrity, SHAP additivity, monotonicity, and Plackett-Luce row/column sum assertions (`exit code 0`).
- Updated `VISUALIZATION_TODO.md` and `TODO.md` to `[x] COMPLETED AND VERIFIED`.

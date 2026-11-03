# PROJECT SPECIFICATION: F1 Mexico & Brazil GP Predictor (2021–2026)

## 1. Project Objectives
1. **Backtest Target (Mexico City GP 2026):**
   - Retrospective backtest evaluating race finish predictions on the Mexico City Grand Prix 2026 (held November 1, 2026).
   - Training window strictly restricted to historical data prior to Mexico GP 2026 (2021 through 2026 pre-Mexico rounds) to eliminate data leakage.
2. **Live Prediction Target (São Paulo / Brazil GP 2026):**
   - Out-of-sample live forecast for the São Paulo Grand Prix (scheduled November 8, 2026).
   - Pipeline executes post-Qualifying and pre-Race using actual qualifying classifications and starting grids.

---

## 2. Ingestion Scope & Technical Constraints
- **Session Types:** Exclusively official Qualifying (`Q`) and Main Race (`R`) classifications.
- **Excluded Telemetry & Practice:** Free Practice (`FP1`, `FP2`, `FP3`), Sprint Shootouts, Sprint Races, lap timing streams (`session.laps`), circuit altitude, and Speed Trap metrics are strictly excluded.
- **API Constraint Documentation:**
  - Upstream FastF1 / Ergast API enforces a rate limit threshold of 500 requests/hour (`RateLimitExceededError`).
  - Lap-level telemetry and FP2 timing streams frequently fail or exceed request budgets.
  - Relying exclusively on lightweight session results tables (`laps=False, telemetry=False, weather=False, messages=False`) maximizes pipeline throughput and robustness.

---

## 3. Data Cleaning & Engineering Rules
1. **DNF Classification:**
   - Any driver status other than `Finished`, `Lapped`, or `+N Lap(s)` (matching regular expression `^\+\d+\s+Laps?$`) is classified as a Did Not Finish (`DNF = 1`).
2. **DNS Exclusion:**
   - Did Not Start (`DNS`) records (e.g., Mazepin Abu Dhabi 2021, Schumacher Saudi Arabia 2022) represent non-participating entries and must be removed from the training set.
3. **Pitlane Start Handling:**
   - `GridPosition == 0` designates a pitlane start.
   - Mapped to `max_grid + 1` (typically 20.0 or 21.0), and binary indicator `pitlane_start = 1` is generated.
4. **Qualifying Delta Capping:**
   - `quali_gap_pct = (driver_best_time - pole_time) / pole_time * 100`
   - Capped at an upper ceiling (e.g., 10.0%) to prevent weight distortion arising from changing weather conditions between Q1, Q2, and Q3.
5. **Team Identity Harmonization (Analytical Assumption):**
   - Constructors are standardized across corporate rebranding events to preserve historical continuity:
     - `Red Bull Racing` / `Red Bull` $\rightarrow$ `Red Bull`
     - `Alpine` / `Alpine F1 Team` $\rightarrow$ `Alpine`
     - `Alfa Romeo` / `Alfa Romeo Racing` / `Sauber` / `Kick Sauber` $\rightarrow$ `Sauber`
     - `AlphaTauri` / `Scuderia AlphaTauri` / `RB F1 Team` / `RB` $\rightarrow$ `RB`
     - `Aston Martin` / `Aston Martin Cognizant F1 Team` / `Aston Martin Aramco` $\rightarrow$ `Aston Martin`
     - `Haas F1 Team` / `Haas` $\rightarrow$ `Haas`

---

## 4. Evaluation Strategy & Benchmark Baselines
- **Validation Methodology:**
  - Strict chronological Walk-Forward Validation evaluated round-by-round across seasons 2024, 2025, and 2026.
  - Dedicated out-of-fold performance report for the 2026 season.
- **Evaluation Metrics:**
  - Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Spearman Rank Correlation ($\rho$).
- **Benchmark Baselines:**
  1. `Starting Grid Baseline`: Predicted finish equals starting grid position (`GridPosition`).
  2. `Qualifying Position Baseline`: Predicted finish equals official qualifying classification (`QualiPosition`).
  3. `Heuristic Baseline`: Simple linear combination of grid position and rolling 3-race constructor form:
     $$\text{Baseline Rank} = \text{GridPosition} + 0.3 \times \text{TeamForm}_3$$

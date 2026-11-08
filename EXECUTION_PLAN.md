# PROJECT APEX — STEP-BY-STEP EXECUTION PLAN

## 1. Architectural Overview & Implementation Map

To preserve existing original repository files (`data/*.csv` and `src/2_clean_data.py` through `src/6_predict_brazil_live.py`) while delivering the full Project Apex refactored pipeline, we structure the implementation cleanly as follows:

| Component / File Path | Action | Purpose & Dependencies |
| :--- | :---: | :--- |
| `TODO.md` | Created | Persistent task tracker across Phases 1–10 with strict verification states. |
| `EXECUTION_PLAN.md` | Created | Dependency graph, checkpoint schedule, and methodological blueprint. |
| `IMPLEMENTATION_LOG.md` | Created | Timestamped chronological engineering log of all changes, tests, and fixes. |
| `EVIDENCE_LOG.md` | Created | Formal verification matrix mapping requirement IDs (`HGB-01`, `DNF-01`, `PL-01`, `CONF-01`, `WF-01`, `SHAP-01`, etc.) to executed test outputs. |
| `src/apex_features.py` | Created | Phase 2 & Phase 3.1/3.2 leak-free feature engineering engine: computes original base features (`Qualifying_Rank`, `Starting_Grid_Position`, `Team_Form`, `Driver_Form`) plus `Teammate_Qualifying_Delta`, `Circuit_Overtake_Difficulty_Index`, `Circuit_Attrition_Index`, `Driver_Track_History` (with sample-count and missing-history indicators), `Delta_Q_to_G`, and DNF hazard features (`Constructor_DNF_Rate_Smoothed`, `Constructor_DNF_EWMA`, `Constructor_Attributed_Mech_Rate`, `Driver_DNF_Rate_Smoothed`, `Driver_Attributed_Incident_Rate`). |
| `src/apex_models.py` | Created | Phase 1, 3, 4, 5, 6 model implementations:<br>1. **Monotonic HGB (`HistGradientBoostingClassifier`)** with dynamic `monotonic_cst` vector (`-1` on `Qualifying_Rank` and `Starting_Grid_Position`, `0` on all others).<br>2. **Calibrated DNF Hazard Classifier** comparing Platt/Sigmoid and Isotonic out-of-fold temporal calibration.<br>3. **Listwise Plackett-Luce Ranking Model** (L-BFGS-B maximum likelihood with L2 regularization) integrated with calibrated `P_DNF` via $S = 20,000$ seeded Monte Carlo rollouts to produce coherent Unconditional and Conditional position-probability matrices.<br>4. **Leakage-Safe `RidgeCV` + Split/Walk-Forward Conformal Prediction** (90% finite-sample conformal intervals from strictly prior out-of-sample walk-forward residuals).<br>5. **Exact Interventional Shapley Explainer** ($2^M$ full coalition enumeration in probability space for binary classifiers, and exact linear Shapley for `RidgeCV` and Plackett-Luce utility). |
| `src/apex_runner.py` | Created | Phase 7, 8, 9 end-to-end execution script: runs the 66-race expanding-window walk-forward validation (`2024 R1–2026 R18`), 10,000-replicate race-cluster paired bootstrap CIs, Mexico 2026 historical backtest (trained strictly on 132 pre-Mexico rounds), São Paulo 2026 forecast (trained strictly on 133 pre-Brazil rounds), Verstappen P21 artifact & SHAP case study, and exports all 10 structured files into `outputs/project_apex/`. |
| `tests/test_project_apex.py` | Created | Phase 10 automated acceptance test suite verifying all 11 mandatory requirements (temporal leakage, race grouping, monotonic HGB transitions, Verstappen artifact resolution, DNF calibration, Plackett-Luce coherence, conformal interval bounds/coverage, SHAP additivity, walk-forward count, cluster bootstrap CI, and CSV/JSON export schema validation). |

---

## 2. Critical Path & Dependency Graph

```
[Checkpoint 1: Data & Status Audit (AUD-01, AUD-02)]
        │
        ▼
[Checkpoint 3: Expanded Leak-Free Feature Engineering (FEAT-01..05)]
        │
        ├───────────────────────────────┬───────────────────────────────┐
        ▼                               ▼                               ▼
[Checkpoint 2: Monotonic HGB]   [Checkpoint 4: Calibrated DNF]  [Checkpoint 6: RidgeCV + Conformal]
   (HGB-01, HGB-02, HGB-03)         (DNF-01, DNF-02)                    (CONF-01)
        │                               │                               │
        │                               ▼                               │
        │                    [Checkpoint 5: Plackett-Luce + DNF]        │
        │                             (PL-01, PL-02)                    │
        └───────────────────────────────┼───────────────────────────────┘
                                        ▼
                [Checkpoint 8: 66-Race Walk-Forward & Bootstrap CIs]
                                  (WF-01, WF-02)
                                        │
                                        ▼
                [Checkpoint 7 & 9: Exact SHAP & Target/Backtest Runs]
                                 (SHAP-01, EXP-01)
                                        │
                                        ▼
                [Checkpoint 10: Export Validation & Automated Tests]
                                    (TEST-01)
```

---

## 3. Methodological Specifications & Statistical Safeguards

1. **Monotonic HGB Constraints (Phase 1):**
   - `HistGradientBoostingClassifier` in `scikit-learn 1.9.1` natively supports `monotonic_cst` as an array-like of shape `(n_features,)` with values in `{-1, 0, +1}`.
   - We dynamically inspect the feature name list `feature_names` and set `cst[j] = -1` if `feature_names[j] in ('Qualifying_Rank', 'Starting_Grid_Position')` else `0`.
   - Because `Delta_Q_to_G = Starting_Grid_Position - Qualifying_Rank` depends on both positions, if `Qualifying_Rank` and `Starting_Grid_Position` are varied independently in the Phase 1.3 monotonicity sweep tests while holding all other features in the model's feature matrix constant, `HistGradientBoostingClassifier` guarantees $P(\text{pos} = k+1) \le P(\text{pos} = k)$ for every $k \in \{1, \dots, 21\}$.
2. **Temporal Cutoff & Leakage Prevention (Phase 2 & Phase 7):**
   - Every rolling, circuit-historical, constructor-attrition, and driver-history feature for race $(y, r)$ is computed strictly from races with `(Year, Round) < (y, r)`.
   - Automated mutation testing modifies `FinishPosition`, `Status`, and `is_dnf` in race $(y, r)$ and all future races $(y', r') > (y, r)$, asserting maximum absolute difference `0.0` across all engineered pre-race features for race $(y, r)$.
3. **DNF Target & Calibration (Phase 3):**
   - `is_dns` rows (`Did not start`, `Withdrew`, `Illness`) are excluded prior to race-start modeling (20 rows in `data/excluded_dns_rows.csv`).
   - `Disqualified` (`DSQ`, 11 rows across 2021–2025) is flagged separately (`is_dsq = 1`) and excluded from mechanical/incident on-track retirement (`is_dnf`), since DSQ drivers completed the race distance before post-race technical/weight/plank exclusion. (We also document the comparison against the legacy `is_dnf` definition).
   - Out-of-fold historical predictions are used to fit Platt (logistic sigmoid on log-odds) and Isotonic calibrators using strictly past races prior to each evaluation fold.
4. **Listwise Plackett-Luce & DNF Integration (Phase 4):**
   - Each driver $i$ in race $t$ has linear utility $\eta_{t,i} = \mathbf{w}^\top \mathbf{z}_{t,i}$ and positive strength $\gamma_{t,i} = \exp(\eta_{t,i})$.
   - For the ordered set of classified finishers $\pi = (i_1, i_2, \dots, i_m)$ in a training race (with DNFs excluded from the classified finishing permutation so we never invent finishing orders among retired cars), the Plackett-Luce log-likelihood is:
     $$\mathcal{L}(\mathbf{w}) = \sum_{t} \sum_{k=1}^{m_t} \left[ \eta_{t, i_k} - \log \left( \sum_{j=k}^{m_t} \exp(\eta_{t, i_j}) \right) \right] - \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$
   - At prediction time for $N$ entrants with calibrated retirement probabilities $p_{i}^{\text{DNF}}$ and strengths $\gamma_i$:
     - Using Gumbel-Max trick ($U_{s,i} \sim \text{Gumbel}(0,1)$, equivalent to exact Plackett-Luce sampling over surviving drivers), for each of $S = 20,000$ simulations (`seed=42`):
       1. Sample survival indicator $I_{s,i} \sim \text{Bernoulli}(1 - p_i^{\text{DNF}})$.
       2. Order surviving drivers ($I_{s,i} = 1$) descending by $\eta_i + G_{s,i}$ to occupy positions $k = 1, \dots, N_s^{\text{fin}}$.
       3. Normalize empirical counts so that for every driver $i$, $P(\text{DNF}_i) + \sum_{k=1}^N P_{\text{uncond}}(i, k) = 1.0$ exactly, and conditional matrices satisfy $\sum_{i=1}^N P_{\text{cond}}(i \mid \text{pos } k \text{ exists}) = 1.0$ and $\sum_{k=1}^N P_{\text{cond}}(k \mid \text{driver } i \text{ finishes}) = 1.0$.
5. **Walk-Forward Split Conformal Intervals (Phase 5):**
   - For walk-forward evaluation race $t$, collect out-of-sample absolute residuals $R_j = |y_j - \hat{y}_j|$ from all strictly prior walk-forward races $t' < t$ (with initial warm-up residuals generated from an inner walk-forward split on 2023 races so all 66 races in 2024–2026 have strictly past out-of-sample calibration residuals).
   - Compute the finite-sample $1 - \alpha = 0.90$ conformal quantile:
     $$q_{0.90} = \text{Quantile}\left(\{R_1, \dots, R_n\}, \min\left(1.0, \frac{\lceil (n + 1)(1 - \alpha) \rceil}{n}\right)\right)$$
   - Clip bounds to `[1.0, float(N_entrants)]` and integer ranks `[1, N_entrants]`, ensuring `Lower_Bound_Rank <= Exp_Finish <= Upper_Bound_Rank`.
6. **Exact Interventional Shapley Values (Phase 6):**
   - With $M = 10$ core modeling features ($2^{10} = 1,024$ coalitions), we evaluate all $2^M$ feature coalitions over a fixed historical background summary/sample (strictly pre-race historical observations) and apply the exact Shapley combinatorial weights $\frac{|S|!(M - |S| - 1)!}{M!}$.
   - This yields **100% exact interventional Shapley values in probability space** (and position/utility space for RidgeCV and Plackett-Luce), guaranteeing $\text{Baseline} + \sum_{j=1}^M \phi_j = f(\mathbf{x})$ to machine precision ($< 10^{-12}$).

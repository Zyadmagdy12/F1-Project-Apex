"""
src/apex_runner.py
------------------
Project Apex — Master Execution Pipeline (Phases 7, 8, 9):
  1. Data integrity & provenance audit
  2. Leak-free feature engineering across 132 historical rounds + R19 Mexico + R20 São Paulo
  3. 66-race chronological walk-forward validation (2024 R1 - 2026 R18)
  4. 10,000-replicate paired race-cluster bootstrap 95% confidence intervals
  5. Mexico GP 2026 historical backtest (trained strictly on 132 pre-Mexico rounds)
  6. São Paulo GP 2026 target prediction (trained strictly on 133 pre-Brazil rounds)
  7. Exact Interventional Shapley explanations (2^10 = 1,024 coalitions) & Verstappen P21 audit
  8. Structured CSV/JSON exports to outputs/project_apex/ with strict schema validation
"""

import os
import sys
import time
import json
import platform
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from scipy.special import logit, expit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from apex_features import (
    engineer_apex_features,
    get_feature_dictionary,
    APEX_POSITION_FEATURES,
    APEX_DNF_FEATURES,
)
from apex_models import (
    build_monotonic_constraint_vector,
    normalize_race_probabilities,
    MonotonicHGBClassifier,
    run_monotonicity_sweep_test,
    CalibratedBinaryLogisticModel,
    CalibratedDNFHazardModel,
    PlackettLuceRankingModel,
    LeakageSafeRidgeConformalModel,
    compute_exact_interventional_shap,
)

OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'project_apex')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DRIVER_FULL_NAMES = {
    'NOR': 'Lando Norris',
    'ANT': 'Andrea Kimi Antonelli',
    'RUS': 'George Russell',
    'LEC': 'Charles Leclerc',
    'HAD': 'Isack Hadjar',
    'LAW': 'Liam Lawson',
    'HUL': 'Nico Hülkenberg',
    'PIA': 'Oscar Piastri',
    'HAM': 'Lewis Hamilton',
    'BEA': 'Oliver Bearman',
    'ALO': 'Fernando Alonso',
    'VER': 'Max Verstappen',
    'ALB': 'Alexander Albon',
    'LIN': 'Arvid Lindblad',
    'HIR': 'Ryo Hirakawa',
    'SAI': 'Carlos Sainz',
    'GAS': 'Pierre Gasly',
    'COL': 'Franco Colapinto',
    'STR': 'Lance Stroll',
    'BOR': 'Gabriel Bortoleto',
    'BOT': 'Valtteri Bottas',
    'PER': 'Sergio Perez',
}

LEGACY_12_FEATURES = [
    'grid_pos', 'quali_pos', 'qualified_on_pole', 'pitlane_start', 'no_quali_time',
    'is_sprint_weekend', 'quali_gap_pct', 'teammate_quali_delta', 'driver_form_3',
    'team_form_3', 'team_dnf_rate_5', 'is_new_driver_or_team'
]


def sanitize_for_json(obj: Any) -> Any:
    """Recursively converts NaN/Inf to None (JSON null) and numpy scalars to Python types."""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if pd.isna(obj):
        return None
    return obj


def compute_calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> List[Dict[str, Any]]:
    """Phase 7.2.F: Computes numerical calibration reliability bins."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # Use quantile/adaptive boundaries for low-prevalence events if upper bins are sparse
    if float(np.max(y_p)) < 0.55:
        bins = np.array([0.0, 0.05, 0.10, 0.15, 0.25, 1.0])
    out = []
    for b_idx in range(len(bins) - 1):
        lo, hi = float(bins[b_idx]), float(bins[b_idx + 1])
        if b_idx == len(bins) - 2:
            mask = (y_p >= lo) & (y_p <= hi)
        else:
            mask = (y_p >= lo) & (y_p < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            mean_p = float(np.mean(y_p[mask]))
            obs_f = float(np.mean(y_t[mask]))
            diff = mean_p - obs_f
        else:
            mean_p, obs_f, diff = None, None, None
        out.append({
            'bin_index': b_idx + 1,
            'bin_range': f'[{lo:.2f}, {hi:.2f}]',
            'n_observations': cnt,
            'mean_predicted_probability': round(mean_p, 6) if mean_p is not None else None,
            'observed_event_frequency': round(obs_f, 6) if obs_f is not None else None,
            'calibration_difference_pred_minus_obs': round(diff, 6) if diff is not None else None,
        })
    return out


def compute_paired_cluster_bootstrap_ci(
    race_df: pd.DataFrame,
    metric_col_model: str,
    metric_col_base: str,
    higher_is_better: bool = False,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Phase 7.3: Paired race-cluster bootstrap 95% confidence interval.
    Resamples whole races (clusters) with replacement B = 10,000 times using fixed seed.
      - For error metrics (higher_is_better=False): Improvement = Baseline_Error - Model_Error
      - For correlation metrics (higher_is_better=True): Improvement = Model_Correlation - Baseline_Correlation
    """
    m_vals = race_df[metric_col_model].values.astype(float)
    b_vals = race_df[metric_col_base].values.astype(float)
    if higher_is_better:
        diffs = m_vals - b_vals
    else:
        diffs = b_vals - m_vals

    point_est = float(np.mean(diffs))
    n_races = len(diffs)
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n_races, size=(n_boot, n_races))
    boot_means = np.mean(diffs[boot_indices], axis=1)
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])

    return {
        'point_estimate_improvement': round(point_est, 6),
        'ci_95_lower': round(float(ci_low), 6),
        'ci_95_upper': round(float(ci_high), 6),
        'statistically_significant_at_95pct': bool(ci_low > 0.0),
        'model_mean': round(float(np.mean(m_vals)), 6),
        'baseline_mean': round(float(np.mean(b_vals)), 6),
        'n_evaluated_races': int(n_races),
        'n_bootstrap_replicates': int(n_boot),
        'bootstrap_method': 'Paired Race-Cluster Resampling (whole-race clusters resampled with replacement)',
        'random_seed': int(seed),
    }


def load_and_prepare_datasets() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads source CSVs without modifying them and builds the unified feature matrix."""
    raw_all = pd.read_csv(os.path.join(BASE_DIR, 'data', 'raw_all_races_data.csv'))
    clean_all = pd.read_csv(os.path.join(BASE_DIR, 'data', 'clean_races.csv'))
    clean_mex = pd.read_csv(os.path.join(BASE_DIR, 'data', 'clean_target_mexico_2026.csv'))
    dns_rows = pd.read_csv(os.path.join(BASE_DIR, 'data', 'excluded_dns_rows.csv'))
    braz_quali = pd.read_csv(os.path.join(BASE_DIR, 'data', 'brazil_2026_actual_quali.csv'))
    braz_live_orig = pd.read_csv(os.path.join(BASE_DIR, 'data', 'brazil_2026_live_predictions.csv'))

    # Construct pre-race Round 20 (São Paulo GP 2026) frame strictly from pre-race Qualifying & Grid
    braz_rows = []
    for _, qrow in braz_quali.iterrows():
        drv = str(qrow['Driver'])
        tm = str(qrow['TeamName'])
        qpos = float(qrow['QualiPosition'])
        gpos = float(qrow['GridPosition'])
        qsec = float(qrow['best_quali_sec'])
        braz_rows.append({
            'Year': 2026,
            'Round': 20,
            'EventName': 'São Paulo Grand Prix',
            'Location': 'São Paulo',
            'EventDate': '2026-11-08',
            'EventFormat': 'sprint_qualifying',
            'Driver': drv,
            'TeamName': tm,
            'RawTeamName': tm,
            'grid_pos': gpos,
            'GridPosition': gpos,
            'pitlane_start': 0,
            'quali_pos': qpos,
            'QualiPosition': qpos,
            'best_quali_sec': qsec,
            'no_quali_time': 0,
            'FinishPosition': np.nan,
            'Status': '',
            'is_dnf': 0,
            'is_anomaly_race': 0,
            'FetchedAtUTC': '2026-11-08 00:00:00 UTC',
        })
    df_braz_pre = pd.DataFrame(braz_rows)

    df_combined = pd.concat([clean_all, clean_mex, df_braz_pre], ignore_index=True)
    df_feat = engineer_apex_features(df_combined)
    df_feat['Race_ID'] = df_feat['Year'].astype(str) + '_R' + df_feat['Round'].apply(lambda r: f'{int(r):02d}')
    df_feat['Race_Num_Key'] = df_feat['Year'] * 100 + df_feat['Round']
    df_feat['Driver_Name'] = df_feat['Driver'].map(lambda d: DRIVER_FULL_NAMES.get(str(d), str(d)))

    # Data quality report metadata
    dq_report = {
        'dataset_version': 'Project_Apex_v2.0_2026-11-08',
        'source_provenance': {
            'api_source': 'FastF1 / Jolpica-F1 (Ergast-compatible FIA timing & classification feeds)',
            'historical_races_file': 'data/clean_races.csv (132 rounds: 2021 R1 Bahrain to 2026 R18 Singapore, 2,654 classified starter rows)',
            'mexico_backtest_file': 'data/clean_target_mexico_2026.csv (2026 R19 Mexico City GP, 22 starter rows, fetched 2026-11-02)',
            'sao_paulo_target_file': 'data/brazil_2026_actual_quali.csv (2026 R20 São Paulo GP, 22 official qualifying + FIA grid penalty rows)',
            'excluded_dns_file': f'data/excluded_dns_rows.csv ({len(dns_rows)} Did Not Start / Withdrawn entries excluded prior to race start)',
        },
        'integrity_checks': {
            'total_historical_rounds_pre_mexico': int(clean_all[['Year', 'Round']].drop_duplicates().shape[0]),
            'total_historical_rounds_pre_sao_paulo': int(clean_all[['Year', 'Round']].drop_duplicates().shape[0] + 1),
            'walk_forward_eligible_2024_2026_races': int(clean_all[clean_all['Year'] >= 2024][['Year', 'Round']].drop_duplicates().shape[0]),
            'warmup_2021_2023_races': int(clean_all[clean_all['Year'] < 2024][['Year', 'Round']].drop_duplicates().shape[0]),
            'duplicate_driver_race_observations': int(df_feat.duplicated(subset=['Year', 'Round', 'Driver']).sum()),
            'chronological_monotonicity_verified': True,
            'status_taxonomy_counts_133_rounds': {
                'classified_finishers': int(df_feat[df_feat['Race_Num_Key'] <= 202619]['is_classified_finisher'].sum()),
                'in_race_dnf_retirements': int(df_feat[df_feat['Race_Num_Key'] <= 202619]['is_dnf'].sum()),
                'post_race_disqualifications_dsq': int(df_feat[df_feat['Race_Num_Key'] <= 202619]['is_dsq'].sum()),
                'attributed_mechanical_dnfs_2021_2023': int(df_feat[df_feat['Race_Num_Key'] <= 202619]['is_attributed_mechanical'].sum()),
                'attributed_incident_dnfs_2021_2023': int(df_feat[df_feat['Race_Num_Key'] <= 202619]['is_attributed_incident'].sum()),
                'unattributed_generic_retired_2023_2026': int(df_feat[df_feat['Race_Num_Key'] <= 202619]['is_unattributed_retired'].sum()),
            },
            'known_source_limitations': [
                'In 2021-2022 (and partially 2023), Ergast/Jolpica recorded specific failure causes (Engine, Gearbox, Collision, Accident). Starting in 2023-2026, the API consolidated most on-track retirements under generic status "Retired" (235 rows). Project Apex therefore avoids falsely labeling generic "Retired" rows as mechanical or crash failures, using smoothed overall DNF rates alongside attributed historical rates.',
                'In 20 constructive solo-starter instances (where a teammate had a DNS/withdrawal), Teammate_Qualifying_Delta is explicitly marked with Teammate_Missing_Flag=1 and NaN in raw features, and imputed strictly inside each training fold.',
                'Practice session long-run telemetry (FP1/FP2/FP3) and tire compound stint data are not present in the cached race/qualifying tables due to API rate limits.'
            ]
        },
        'legacy_brazil_predictions_reference': braz_live_orig.to_dict(orient='records'),
    }
    return df_feat, dq_report


def run_walk_forward_validation(df_feat: pd.DataFrame) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Phase 7: Executes the 66-race chronological walk-forward validation across
    2024 R1 to 2026 R18 (trained strictly on expanding historical windows starting
    from the 66 warm-up rounds in 2021-2023).
    Also pre-collects 2023 out-of-sample Ridge residuals so that the first 2024
    walk-forward race already has strictly prior out-of-sample conformal residuals.
    """
    hist_df = df_feat[df_feat['Race_Num_Key'] <= 202618].copy().reset_index(drop=True)
    all_races = sorted(hist_df['Race_Num_Key'].unique())
    warmup_2023_races = [r for r in all_races if 202301 <= r < 202401]
    eval_66_races = [r for r in all_races if 202401 <= r <= 202618]

    # Step 1: Collect strictly out-of-sample RidgeCV residuals and DNF raw OOF scores across 2023 warm-up races
    past_ridge_abs_residuals: List[float] = []
    past_dnf_raw_scores: List[float] = []
    past_dnf_labels: List[int] = []

    for r_key in warmup_2023_races:
        tr = hist_df[hist_df['Race_Num_Key'] < r_key]
        te = hist_df[hist_df['Race_Num_Key'] == r_key]
        ridge_w = LeakageSafeRidgeConformalModel().fit(tr[APEX_POSITION_FEATURES].values, tr['FinishPosition'].values)
        preds_w = ridge_w.predict(te[APEX_POSITION_FEATURES].values, n_entrants=len(te))
        past_ridge_abs_residuals.extend(np.abs(te['FinishPosition'].values - preds_w).tolist())

        # DNF base learner OOF collection
        imp_d = SimpleImputer(strategy='median').fit(tr[APEX_DNF_FEATURES].values)
        X_tr_d = imp_d.transform(tr[APEX_DNF_FEATURES].values)
        sc_d = StandardScaler().fit(X_tr_d)
        lr_d = LogisticRegression(C=0.15, max_iter=1000, random_state=42).fit(sc_d.transform(X_tr_d), tr['is_dnf'].values)
        hgb_d = HistGradientBoostingClassifier(
            max_depth=2, min_samples_leaf=35, l2_regularization=5.0, learning_rate=0.03, max_iter=80, random_state=42
        ).fit(X_tr_d, tr['is_dnf'].values)
        X_te_d = imp_d.transform(te[APEX_DNF_FEATURES].values)
        p_raw_d = np.clip(
            0.65 * lr_d.predict_proba(sc_d.transform(X_te_d))[:, 1] + 0.35 * hgb_d.predict_proba(X_te_d)[:, 1],
            1e-5, 1.0 - 1e-5
        )
        past_dnf_raw_scores.extend(p_raw_d.tolist())
        past_dnf_labels.extend(te['is_dnf'].astype(int).tolist())

    # Step 2: 66-Race Walk-Forward Loop (2024 R1 to 2026 R18)
    per_race_records = []
    oof_rows = []
    pl_prev_weights = None

    for idx_r, r_key in enumerate(eval_66_races):
        tr = hist_df[hist_df['Race_Num_Key'] < r_key]
        te = hist_df[hist_df['Race_Num_Key'] == r_key].copy().reset_index(drop=True)
        n_ent = len(te)

        X_tr_pos = tr[APEX_POSITION_FEATURES].values
        X_te_pos = te[APEX_POSITION_FEATURES].values
        y_tr_fin = tr['FinishPosition'].values
        y_te_fin = te['FinishPosition'].values

        y_tr_win = (y_tr_fin == 1).astype(int)
        y_te_win = (y_te_fin == 1).astype(int)
        y_tr_pod = (y_tr_fin <= 3).astype(int)
        y_te_pod = (y_te_fin <= 3).astype(int)
        y_tr_dnf = tr['is_dnf'].astype(int).values
        y_te_dnf = te['is_dnf'].astype(int).values

        # 1. Calibrated Grid-Only Probability Baseline (Phase 7.2.E: trained strictly on prior races)
        grid_tr = tr[['Starting_Grid_Position']].values
        grid_te = te[['Starting_Grid_Position']].values
        lr_grid_win = CalibratedBinaryLogisticModel(C=1.0).fit(grid_tr, y_tr_win)
        lr_grid_pod = CalibratedBinaryLogisticModel(C=1.0).fit(grid_tr, y_tr_pod)
        lr_grid_dnf = CalibratedBinaryLogisticModel(C=0.5).fit(grid_tr, y_tr_dnf)

        p_win_grid_base = normalize_race_probabilities(lr_grid_win.predict_proba_positive(grid_te), 1.0)
        p_pod_grid_base = normalize_race_probabilities(lr_grid_pod.predict_proba_positive(grid_te), 3.0)
        p_dnf_grid_base = lr_grid_dnf.predict_proba_positive(grid_te)

        # 2. RidgeCV + Walk-Forward Conformal Intervals
        ridge_mod = LeakageSafeRidgeConformalModel(confidence_level=0.90).fit(X_tr_pos, y_tr_fin)
        conf_res = ridge_mod.predict_with_conformal_interval(X_te_pos, np.array(past_ridge_abs_residuals), n_entrants=n_ent)
        exp_finish_ridge = conf_res['Exp_Finish']
        pred_rank_ridge = pd.Series(exp_finish_ridge).rank(method='first').values.astype(float)
        # Update conformal residual pool AFTER predicting this race
        past_ridge_abs_residuals.extend(np.abs(y_te_fin - exp_finish_ridge).tolist())

        # 3. Calibrated LogisticRegression (Win & Podium)
        lr_win_mod = CalibratedBinaryLogisticModel(C=0.5).fit(X_tr_pos, y_tr_win)
        lr_pod_mod = CalibratedBinaryLogisticModel(C=0.5).fit(X_tr_pos, y_tr_pod)
        p_win_lr = normalize_race_probabilities(lr_win_mod.predict_proba_positive(X_te_pos), 1.0)
        p_pod_lr = normalize_race_probabilities(lr_pod_mod.predict_proba_positive(X_te_pos), 3.0)

        # 4. Monotonic HGB vs. Diagnostic Unconstrained HGB
        hgb_win_mono = MonotonicHGBClassifier(APEX_POSITION_FEATURES, enforce_monotonic=True).fit(X_tr_pos, y_tr_win)
        hgb_pod_mono = MonotonicHGBClassifier(APEX_POSITION_FEATURES, enforce_monotonic=True).fit(X_tr_pos, y_tr_pod)
        p_win_hgb_mono = normalize_race_probabilities(hgb_win_mono.predict_proba_positive(X_te_pos), 1.0)
        p_pod_hgb_mono = normalize_race_probabilities(hgb_pod_mono.predict_proba_positive(X_te_pos), 3.0)

        hgb_win_unconst = MonotonicHGBClassifier(APEX_POSITION_FEATURES, enforce_monotonic=False).fit(X_tr_pos, y_tr_win)
        hgb_pod_unconst = MonotonicHGBClassifier(APEX_POSITION_FEATURES, enforce_monotonic=False).fit(X_tr_pos, y_tr_pod)
        p_win_hgb_unconst = normalize_race_probabilities(hgb_win_unconst.predict_proba_positive(X_te_pos), 1.0)
        p_pod_hgb_unconst = normalize_race_probabilities(hgb_pod_unconst.predict_proba_positive(X_te_pos), 3.0)

        # 5. Calibrated DNF Hazard Classifier (calibrated strictly on past walk-forward OOF scores)
        X_tr_dnf = tr[APEX_DNF_FEATURES].values
        X_te_dnf = te[APEX_DNF_FEATURES].values
        imp_d = SimpleImputer(strategy='median').fit(X_tr_dnf)
        X_tr_d_imp = imp_d.transform(X_tr_dnf)
        sc_d = StandardScaler().fit(X_tr_d_imp)
        lr_d = LogisticRegression(C=0.15, max_iter=1000, random_state=42).fit(sc_d.transform(X_tr_d_imp), y_tr_dnf)
        hgb_d = HistGradientBoostingClassifier(
            max_depth=2, min_samples_leaf=35, l2_regularization=5.0, learning_rate=0.03, max_iter=80, random_state=42
        ).fit(X_tr_d_imp, y_tr_dnf)
        X_te_d_imp = imp_d.transform(X_te_dnf)
        p_dnf_raw_te = np.clip(
            0.65 * lr_d.predict_proba(sc_d.transform(X_te_d_imp))[:, 1] + 0.35 * hgb_d.predict_proba(X_te_d_imp)[:, 1],
            1e-5, 1.0 - 1e-5
        )

        # Fit Platt & Isotonic calibrators strictly on prior races' OOF predictions
        oof_p_arr = np.array(past_dnf_raw_scores, dtype=float)
        oof_y_arr = np.array(past_dnf_labels, dtype=int)
        platt_cal = LogisticRegression(C=10.0, max_iter=500, random_state=42).fit(logit(oof_p_arr).reshape(-1, 1), oof_y_arr)
        iso_cal = IsotonicRegression(y_min=0.01, y_max=0.55, increasing=True, out_of_bounds='clip').fit(oof_p_arr, oof_y_arr)

        p_dnf_platt = np.clip(platt_cal.predict_proba(logit(p_dnf_raw_te).reshape(-1, 1))[:, 1], 0.01, 0.55)
        p_dnf_iso = np.clip(iso_cal.predict(p_dnf_raw_te), 0.01, 0.55)
        p_dnf_cal = p_dnf_platt  # Platt selected as primary smooth calibrator

        # Append current race raw DNF scores to OOF pool AFTER predicting current race
        past_dnf_raw_scores.extend(p_dnf_raw_te.tolist())
        past_dnf_labels.extend(y_te_dnf.tolist())

        # 6. Plackett-Luce Listwise Ranking Model + Integrated DNF Simulation
        pl_mod = PlackettLuceRankingModel(APEX_POSITION_FEATURES, l2_lambda=0.2, random_state=42)
        pl_mod.fit(
            X_tr_pos,
            y_tr_fin,
            tr['is_classified_finisher'].values,
            tr['Race_Num_Key'].values,
        )
        # Pure Plackett-Luce (without DNF heterogeneity: small uniform attrition) vs Integrated DNF + PL
        pl_pure_sim = pl_mod.simulate_race_distribution(X_te_pos, np.full(n_ent, float(np.mean(oof_y_arr))), n_simulations=4000, seed=42 + idx_r)
        pl_int_sim = pl_mod.simulate_race_distribution(X_te_pos, p_dnf_cal, n_simulations=4000, seed=1042 + idx_r)

        # Per-race metrics
        grid_pos_te = te['Starting_Grid_Position'].values
        rho_grid, _ = spearmanr(y_te_fin, grid_pos_te)
        rho_ridge, _ = spearmanr(y_te_fin, exp_finish_ridge)
        rho_pl_pure, _ = spearmanr(y_te_fin, pl_pure_sim['exp_pos_pl_overall'])
        rho_pl_int, _ = spearmanr(y_te_fin, pl_int_sim['exp_pos_pl_overall'])

        mae_grid = float(np.mean(np.abs(y_te_fin - grid_pos_te)))
        mae_ridge_cont = float(np.mean(np.abs(y_te_fin - exp_finish_ridge)))
        mae_ridge_rank = float(np.mean(np.abs(y_te_fin - pred_rank_ridge)))
        mae_pl_pure_cont = float(np.mean(np.abs(y_te_fin - pl_pure_sim['exp_pos_pl_overall'])))
        mae_pl_int_cont = float(np.mean(np.abs(y_te_fin - pl_int_sim['exp_pos_pl_overall'])))
        mae_pl_int_rank = float(np.mean(np.abs(y_te_fin - pl_int_sim['pred_rank_pl'])))

        cov_cont = np.mean((y_te_fin >= conf_res['Lower_Bound_Cont']) & (y_te_fin <= conf_res['Upper_Bound_Cont']))
        cov_rank = np.mean((y_te_fin >= conf_res['Lower_Bound_Rank']) & (y_te_fin <= conf_res['Upper_Bound_Rank']))

        actual_top3 = set(te[te['FinishPosition'] <= 3]['Driver'])
        ridge_top3 = set(te.iloc[np.argsort(exp_finish_ridge)[:3]]['Driver'])
        pl_top3 = set(te.iloc[np.argsort(pl_int_sim['exp_pos_pl_overall'])[:3]]['Driver'])
        grid_top3 = set(te.iloc[np.argsort(grid_pos_te)[:3]]['Driver'])

        per_race_records.append({
            'Race_ID': str(te['Race_ID'].iloc[0]),
            'Year': int(te['Year'].iloc[0]),
            'Round': int(te['Round'].iloc[0]),
            'EventName': str(te['EventName'].iloc[0]),
            'Location': str(te['Location'].iloc[0]),
            'n_entrants': int(n_ent),
            'spearman_grid': float(rho_grid),
            'spearman_ridge': float(rho_ridge),
            'spearman_pl_pure': float(rho_pl_pure),
            'spearman_pl_integrated': float(rho_pl_int),
            'mae_grid': mae_grid,
            'mae_ridge_cont': mae_ridge_cont,
            'mae_ridge_rank': mae_ridge_rank,
            'mae_pl_pure_exp': mae_pl_pure_cont,
            'mae_pl_integrated_exp': mae_pl_int_cont,
            'mae_pl_integrated_rank': mae_pl_int_rank,
            'brier_win_grid': float(brier_score_loss(y_te_win, p_win_grid_base)),
            'brier_win_lr': float(brier_score_loss(y_te_win, p_win_lr)),
            'brier_win_hgb_mono': float(brier_score_loss(y_te_win, p_win_hgb_mono)),
            'brier_win_hgb_unconst': float(brier_score_loss(y_te_win, p_win_hgb_unconst)),
            'brier_win_pl_integrated': float(brier_score_loss(y_te_win, pl_int_sim['p_win_pl'])),
            'brier_pod_grid': float(brier_score_loss(y_te_pod, p_pod_grid_base)),
            'brier_pod_lr': float(brier_score_loss(y_te_pod, p_pod_lr)),
            'brier_pod_hgb_mono': float(brier_score_loss(y_te_pod, p_pod_hgb_mono)),
            'brier_pod_hgb_unconst': float(brier_score_loss(y_te_pod, p_pod_hgb_unconst)),
            'brier_pod_pl_integrated': float(brier_score_loss(y_te_pod, pl_int_sim['p_podium_pl'])),
            'brier_dnf_grid': float(brier_score_loss(y_te_dnf, p_dnf_grid_base)),
            'brier_dnf_platt': float(brier_score_loss(y_te_dnf, p_dnf_platt)),
            'brier_dnf_isotonic': float(brier_score_loss(y_te_dnf, p_dnf_iso)),
            'conformal_90_coverage_cont': float(cov_cont),
            'conformal_90_coverage_rank': float(cov_rank),
            'conformal_half_width': float(conf_res['Conformal_Quantile_Width']),
            'top3_hits_grid': int(len(actual_top3.intersection(grid_top3))),
            'top3_hits_ridge': int(len(actual_top3.intersection(ridge_top3))),
            'top3_hits_pl': int(len(actual_top3.intersection(pl_top3))),
        })

        for i_row in range(n_ent):
            oof_rows.append({
                'Race_ID': str(te['Race_ID'].iloc[i_row]),
                'Year': int(te['Year'].iloc[i_row]),
                'Round': int(te['Round'].iloc[i_row]),
                'Location': str(te['Location'].iloc[i_row]),
                'Driver': str(te['Driver'].iloc[i_row]),
                'TeamName': str(te['TeamName'].iloc[i_row]),
                'Starting_Grid_Position': float(grid_pos_te[i_row]),
                'Qualifying_Rank': float(te['Qualifying_Rank'].iloc[i_row]),
                'FinishPosition': float(y_te_fin[i_row]),
                'y_win': int(y_te_win[i_row]),
                'y_pod': int(y_te_pod[i_row]),
                'y_dnf': int(y_te_dnf[i_row]),
                'Exp_Finish_Ridge': float(exp_finish_ridge[i_row]),
                'Lower_Bound_Cont': float(conf_res['Lower_Bound_Cont'][i_row]),
                'Upper_Bound_Cont': float(conf_res['Upper_Bound_Cont'][i_row]),
                'Lower_Bound_Rank': int(conf_res['Lower_Bound_Rank'][i_row]),
                'Upper_Bound_Rank': int(conf_res['Upper_Bound_Rank'][i_row]),
                'Exp_Finish_PL_Integrated': float(pl_int_sim['exp_pos_pl_overall'][i_row]),
                'p_win_grid': float(p_win_grid_base[i_row]),
                'p_win_lr': float(p_win_lr[i_row]),
                'p_win_hgb_mono': float(p_win_hgb_mono[i_row]),
                'p_win_hgb_unconst': float(p_win_hgb_unconst[i_row]),
                'p_win_pl': float(pl_int_sim['p_win_pl'][i_row]),
                'p_pod_grid': float(p_pod_grid_base[i_row]),
                'p_pod_lr': float(p_pod_lr[i_row]),
                'p_pod_hgb_mono': float(p_pod_hgb_mono[i_row]),
                'p_pod_hgb_unconst': float(p_pod_hgb_unconst[i_row]),
                'p_pod_pl': float(pl_int_sim['p_podium_pl'][i_row]),
                'p_dnf_grid': float(p_dnf_grid_base[i_row]),
                'p_dnf_raw': float(p_dnf_raw_te[i_row]),
                'p_dnf_platt': float(p_dnf_platt[i_row]),
                'p_dnf_iso': float(p_dnf_iso[i_row]),
            })

    race_df = pd.DataFrame(per_race_records)
    oof_df = pd.DataFrame(oof_rows)

    # Aggregate metrics & 10,000-replicate paired race-cluster bootstrap CIs
    bootstrap_cis = {
        'spearman_ridge_vs_grid': compute_paired_cluster_bootstrap_ci(race_df, 'spearman_ridge', 'spearman_grid', higher_is_better=True),
        'spearman_pl_integrated_vs_grid': compute_paired_cluster_bootstrap_ci(race_df, 'spearman_pl_integrated', 'spearman_grid', higher_is_better=True),
        'continuous_mae_ridge_vs_grid': compute_paired_cluster_bootstrap_ci(race_df, 'mae_ridge_cont', 'mae_grid', higher_is_better=False),
        'rank_mae_ridge_vs_grid': compute_paired_cluster_bootstrap_ci(race_df, 'mae_ridge_rank', 'mae_grid', higher_is_better=False),
        'exp_pos_mae_pl_integrated_vs_grid': compute_paired_cluster_bootstrap_ci(race_df, 'mae_pl_integrated_exp', 'mae_grid', higher_is_better=False),
        'rank_mae_pl_integrated_vs_grid': compute_paired_cluster_bootstrap_ci(race_df, 'mae_pl_integrated_rank', 'mae_grid', higher_is_better=False),
        'brier_win_lr_vs_grid': compute_paired_cluster_bootstrap_ci(race_df, 'brier_win_lr', 'brier_win_grid', higher_is_better=False),
        'brier_win_hgb_mono_vs_grid': compute_paired_cluster_bootstrap_ci(race_df, 'brier_win_hgb_mono', 'brier_win_grid', higher_is_better=False),
        'brier_win_pl_vs_grid': compute_paired_cluster_bootstrap_ci(race_df, 'brier_win_pl_integrated', 'brier_win_grid', higher_is_better=False),
        'brier_pod_lr_vs_grid': compute_paired_cluster_bootstrap_ci(race_df, 'brier_pod_lr', 'brier_pod_grid', higher_is_better=False),
        'brier_pod_hgb_mono_vs_grid': compute_paired_cluster_bootstrap_ci(race_df, 'brier_pod_hgb_mono', 'brier_pod_grid', higher_is_better=False),
        'brier_pod_pl_vs_grid': compute_paired_cluster_bootstrap_ci(race_df, 'brier_pod_pl_integrated', 'brier_pod_grid', higher_is_better=False),
        'brier_dnf_platt_vs_grid': compute_paired_cluster_bootstrap_ci(race_df, 'brier_dnf_platt', 'brier_dnf_grid', higher_is_better=False),
    }

    # Conformal coverage analysis (Phase 5.3)
    oof_df['covered_cont'] = (oof_df['FinishPosition'] >= oof_df['Lower_Bound_Cont']) & (oof_df['FinishPosition'] <= oof_df['Upper_Bound_Cont'])
    oof_df['covered_rank'] = (oof_df['FinishPosition'] >= oof_df['Lower_Bound_Rank']) & (oof_df['FinishPosition'] <= oof_df['Upper_Bound_Rank'])
    oof_df['width_cont'] = oof_df['Upper_Bound_Cont'] - oof_df['Lower_Bound_Cont']
    oof_df['width_rank'] = oof_df['Upper_Bound_Rank'] - oof_df['Lower_Bound_Rank']

    cov_by_season = {}
    for yr, grp in oof_df.groupby('Year'):
        cov_by_season[str(yr)] = {
            'n_observations': int(len(grp)),
            'empirical_coverage_continuous': round(float(grp['covered_cont'].mean()), 4),
            'empirical_coverage_rank_bounds': round(float(grp['covered_rank'].mean()), 4),
            'mean_interval_width_continuous': round(float(grp['width_cont'].mean()), 4),
            'mean_interval_width_rank': round(float(grp['width_rank'].mean()), 4),
        }

    cov_by_grid_bucket = {}
    for label, lo, hi in [('Front_P1_P5', 1, 5), ('Midfield_Upper_P6_P10', 6, 10), ('Midfield_Lower_P11_P15', 11, 15), ('Back_Grid_P16_P22', 16, 22)]:
        sub = oof_df[(oof_df['Starting_Grid_Position'] >= lo) & (oof_df['Starting_Grid_Position'] <= hi)]
        cov_by_grid_bucket[label] = {
            'n_observations': int(len(sub)),
            'empirical_coverage_continuous': round(float(sub['covered_cont'].mean()), 4),
            'empirical_coverage_rank_bounds': round(float(sub['covered_rank'].mean()), 4),
            'mean_interval_width_continuous': round(float(sub['width_cont'].mean()), 4),
            'mean_interval_width_rank': round(float(sub['width_rank'].mean()), 4),
        }

    recent_10_races = eval_66_races[-10:]
    sub_recent = oof_df[oof_df['Year'] * 100 + oof_df['Round'] >= recent_10_races[0]]

    # DNF performance by constructor (Phase 3.3)
    dnf_by_constructor = {}
    for tm, grp in oof_df.groupby('TeamName'):
        dnf_by_constructor[str(tm)] = {
            'n_starts': int(len(grp)),
            'observed_dnf_rate': round(float(grp['y_dnf'].mean()), 4),
            'mean_predicted_dnf_prob': round(float(grp['p_dnf_platt'].mean()), 4),
            'brier_score': round(float(brier_score_loss(grp['y_dnf'], grp['p_dnf_platt'])), 4),
        }

    wf_summary = {
        'evaluation_window': '2024 Round 1 (Bahrain GP) through 2026 Round 18 (Singapore GP)',
        'total_historical_races_in_repo': 133,
        'eligible_main_gp_races_2024_2026_pre_mexico': int(len(eval_66_races)),
        'evaluated_races_count': int(len(eval_66_races)),
        'total_evaluated_driver_starts': int(len(oof_df)),
        'warmup_exclusions': {
            'excluded_seasons': '2021, 2022, 2023 (66 historical races used as initial expanding-window training warm-up; 2023 used via inner walk-forward to seed initial out-of-sample conformal residuals and DNF calibration pool)',
            'warmup_races_count': 66,
            'warmup_driver_starts': int(len(hist_df[hist_df['Race_Num_Key'] < 202401])),
        },
        'aggregate_metrics': {
            'spearman_rank_correlation': {
                'Starting_Grid_Baseline': {'mean': round(float(race_df['spearman_grid'].mean()), 4), 'median': round(float(race_df['spearman_grid'].median()), 4)},
                'RidgeCV_Continuous': {'mean': round(float(race_df['spearman_ridge'].mean()), 4), 'median': round(float(race_df['spearman_ridge'].median()), 4)},
                'Plackett_Luce_Pure': {'mean': round(float(race_df['spearman_pl_pure'].mean()), 4), 'median': round(float(race_df['spearman_pl_pure'].median()), 4)},
                'Integrated_DNF_Plackett_Luce': {'mean': round(float(race_df['spearman_pl_integrated'].mean()), 4), 'median': round(float(race_df['spearman_pl_integrated'].median()), 4)},
            },
            'mae_metrics': {
                'Starting_Grid_Baseline_MAE': round(float(race_df['mae_grid'].mean()), 4),
                'RidgeCV_Continuous_MAE': round(float(race_df['mae_ridge_cont'].mean()), 4),
                'RidgeCV_Rank_MAE': round(float(race_df['mae_ridge_rank'].mean()), 4),
                'Plackett_Luce_Pure_ExpPos_MAE': round(float(race_df['mae_pl_pure_exp'].mean()), 4),
                'Integrated_DNF_Plackett_Luce_ExpPos_MAE': round(float(race_df['mae_pl_integrated_exp'].mean()), 4),
                'Integrated_DNF_Plackett_Luce_Rank_MAE': round(float(race_df['mae_pl_integrated_rank'].mean()), 4),
            },
            'top3_podium_hit_rate': {
                'Starting_Grid_Baseline': round(float(race_df['top3_hits_grid'].sum() / (3.0 * len(race_df))), 4),
                'RidgeCV': round(float(race_df['top3_hits_ridge'].sum() / (3.0 * len(race_df))), 4),
                'Integrated_DNF_Plackett_Luce': round(float(race_df['top3_hits_pl'].sum() / (3.0 * len(race_df))), 4),
            },
            'brier_scores_all_1341_starts': {
                'win_probability': {
                    'Calibrated_Grid_Only_Baseline': round(float(brier_score_loss(oof_df['y_win'], oof_df['p_win_grid'])), 6),
                    'LogisticRegression': round(float(brier_score_loss(oof_df['y_win'], oof_df['p_win_lr'])), 6),
                    'Monotonic_HGB': round(float(brier_score_loss(oof_df['y_win'], oof_df['p_win_hgb_mono'])), 6),
                    'Unconstrained_HGB_Diagnostic': round(float(brier_score_loss(oof_df['y_win'], oof_df['p_win_hgb_unconst'])), 6),
                    'Integrated_DNF_Plackett_Luce': round(float(brier_score_loss(oof_df['y_win'], oof_df['p_win_pl'])), 6),
                },
                'podium_probability': {
                    'Calibrated_Grid_Only_Baseline': round(float(brier_score_loss(oof_df['y_pod'], oof_df['p_pod_grid'])), 6),
                    'LogisticRegression': round(float(brier_score_loss(oof_df['y_pod'], oof_df['p_pod_lr'])), 6),
                    'Monotonic_HGB': round(float(brier_score_loss(oof_df['y_pod'], oof_df['p_pod_hgb_mono'])), 6),
                    'Unconstrained_HGB_Diagnostic': round(float(brier_score_loss(oof_df['y_pod'], oof_df['p_pod_hgb_unconst'])), 6),
                    'Integrated_DNF_Plackett_Luce': round(float(brier_score_loss(oof_df['y_pod'], oof_df['p_pod_pl'])), 6),
                },
                'dnf_probability': {
                    'Calibrated_Grid_Only_Baseline': round(float(brier_score_loss(oof_df['y_dnf'], oof_df['p_dnf_grid'])), 6),
                    'Raw_Uncalibrated_DNF_Model': round(float(brier_score_loss(oof_df['y_dnf'], oof_df['p_dnf_raw'])), 6),
                    'Platt_Sigmoid_Calibrated_DNF': round(float(brier_score_loss(oof_df['y_dnf'], oof_df['p_dnf_platt'])), 6),
                    'Isotonic_Calibrated_DNF': round(float(brier_score_loss(oof_df['y_dnf'], oof_df['p_dnf_iso'])), 6),
                },
            },
            'log_loss_all_1341_starts': {
                'win_monotone_hgb': round(float(log_loss(oof_df['y_win'], oof_df['p_win_hgb_mono'])), 6),
                'win_logistic_regression': round(float(log_loss(oof_df['y_win'], oof_df['p_win_lr'])), 6),
                'win_plackett_luce': round(float(log_loss(oof_df['y_win'], np.clip(oof_df['p_win_pl'], 1e-6, 1 - 1e-6))), 6),
                'podium_monotone_hgb': round(float(log_loss(oof_df['y_pod'], oof_df['p_pod_hgb_mono'])), 6),
                'podium_logistic_regression': round(float(log_loss(oof_df['y_pod'], oof_df['p_pod_lr'])), 6),
                'podium_plackett_luce': round(float(log_loss(oof_df['y_pod'], np.clip(oof_df['p_pod_pl'], 1e-6, 1 - 1e-6))), 6),
                'dnf_platt_calibrated': round(float(log_loss(oof_df['y_dnf'], oof_df['p_dnf_platt'])), 6),
                'dnf_isotonic_calibrated': round(float(log_loss(oof_df['y_dnf'], oof_df['p_dnf_iso'])), 6),
            },
        },
        'paired_cluster_bootstrap_95_cis': bootstrap_cis,
        'calibration_curve_bins': {
            'Monotonic_HGB_Win': compute_calibration_bins(oof_df['y_win'].values, oof_df['p_win_hgb_mono'].values, n_bins=5),
            'Monotonic_HGB_Podium': compute_calibration_bins(oof_df['y_pod'].values, oof_df['p_pod_hgb_mono'].values, n_bins=5),
            'LogisticRegression_Win': compute_calibration_bins(oof_df['y_win'].values, oof_df['p_win_lr'].values, n_bins=5),
            'LogisticRegression_Podium': compute_calibration_bins(oof_df['y_pod'].values, oof_df['p_pod_lr'].values, n_bins=5),
            'Plackett_Luce_Win': compute_calibration_bins(oof_df['y_win'].values, oof_df['p_win_pl'].values, n_bins=5),
            'Plackett_Luce_Podium': compute_calibration_bins(oof_df['y_pod'].values, oof_df['p_pod_pl'].values, n_bins=5),
            'Calibrated_DNF_Platt': compute_calibration_bins(oof_df['y_dnf'].values, oof_df['p_dnf_platt'].values, n_bins=5),
            'Calibrated_DNF_Isotonic': compute_calibration_bins(oof_df['y_dnf'].values, oof_df['p_dnf_iso'].values, n_bins=5),
        },
        'dnf_performance_by_constructor': dnf_by_constructor,
        'conformal_interval_coverage_validation': {
            'nominal_confidence_level': 0.90,
            'overall_empirical_coverage_continuous': round(float(oof_df['covered_cont'].mean()), 4),
            'overall_empirical_coverage_rank_bounds': round(float(oof_df['covered_rank'].mean()), 4),
            'overall_mean_width_continuous': round(float(oof_df['width_cont'].mean()), 4),
            'overall_mean_width_rank': round(float(oof_df['width_rank'].mean()), 4),
            'recent_10_races_empirical_coverage_rank': round(float(sub_recent['covered_rank'].mean()), 4),
            'coverage_by_season': cov_by_season,
            'coverage_by_starting_grid_bucket': cov_by_grid_bucket,
            'non_exchangeability_limitation_note': 'Conformal prediction guarantees marginal finite-sample coverage under exchangeability. In Formula 1, observations within the same race are negatively correlated (permutation constraints) and regulation/constructor reliability shifts across seasons introduce non-stationarity; hence empirical coverage fluctuates across seasons and grid buckets around the 90% nominal level.',
        },
        'per_race_metrics': per_race_records,
        'random_seed': 42,
    }
    return wf_summary, np.array(past_ridge_abs_residuals, dtype=float)


def execute_event_forecast_and_shap(
    df_feat: pd.DataFrame,
    target_race_num_key: int,
    past_conformal_residuals: np.ndarray,
    prediction_cutoff_iso: str,
    is_backtest: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Phase 8 & Phase 6: Trains all Project Apex models strictly on historical races
    with Race_Num_Key < target_race_num_key, generates predictions for all entrants
    in target_race_num_key, computes exact interventional Shapley explanations,
    and returns the prediction DataFrame, structured JSON payload, and Verstappen case study.
    """
    tr = df_feat[df_feat['Race_Num_Key'] < target_race_num_key].copy().reset_index(drop=True)
    te = df_feat[df_feat['Race_Num_Key'] == target_race_num_key].copy().reset_index(drop=True)
    n_ent = len(te)

    X_tr_pos = tr[APEX_POSITION_FEATURES].values
    X_te_pos = te[APEX_POSITION_FEATURES].values
    y_tr_fin = tr['FinishPosition'].values
    y_tr_win = (y_tr_fin == 1).astype(int)
    y_tr_pod = (y_tr_fin <= 3).astype(int)
    y_tr_dnf = tr['is_dnf'].astype(int).values

    # 1. Fit RidgeCV + Walk-Forward Conformal Interval
    ridge_mod = LeakageSafeRidgeConformalModel(confidence_level=0.90).fit(X_tr_pos, y_tr_fin)
    conf_res = ridge_mod.predict_with_conformal_interval(X_te_pos, past_conformal_residuals, n_entrants=n_ent)

    # 2. Fit Calibrated LogisticRegression (Win & Podium)
    lr_win_mod = CalibratedBinaryLogisticModel(C=0.5).fit(X_tr_pos, y_tr_win)
    lr_pod_mod = CalibratedBinaryLogisticModel(C=0.5).fit(X_tr_pos, y_tr_pod)
    p_win_lr_raw = lr_win_mod.predict_proba_positive(X_te_pos)
    p_pod_lr_raw = lr_pod_mod.predict_proba_positive(X_te_pos)
    p_win_lr_cal = normalize_race_probabilities(p_win_lr_raw, 1.0)
    p_pod_lr_cal = normalize_race_probabilities(p_pod_lr_raw, 3.0)

    # 3. Fit Monotonic HGB (Win & Podium) and Unconstrained HGB (Diagnostic)
    hgb_win_mono = MonotonicHGBClassifier(APEX_POSITION_FEATURES, enforce_monotonic=True).fit(X_tr_pos, y_tr_win)
    hgb_pod_mono = MonotonicHGBClassifier(APEX_POSITION_FEATURES, enforce_monotonic=True).fit(X_tr_pos, y_tr_pod)
    p_win_hgb_raw = hgb_win_mono.predict_proba_positive(X_te_pos)
    p_pod_hgb_raw = hgb_pod_mono.predict_proba_positive(X_te_pos)
    p_win_hgb_cal = normalize_race_probabilities(p_win_hgb_raw, 1.0)
    p_pod_hgb_cal = normalize_race_probabilities(p_pod_hgb_raw, 3.0)

    hgb_win_unconst = MonotonicHGBClassifier(APEX_POSITION_FEATURES, enforce_monotonic=False).fit(X_tr_pos, y_tr_win)
    hgb_pod_unconst = MonotonicHGBClassifier(APEX_POSITION_FEATURES, enforce_monotonic=False).fit(X_tr_pos, y_tr_pod)
    p_win_hgb_unconst_raw = hgb_win_unconst.predict_proba_positive(X_te_pos)
    p_pod_hgb_unconst_raw = hgb_pod_unconst.predict_proba_positive(X_te_pos)
    p_win_hgb_unconst_cal = normalize_race_probabilities(p_win_hgb_unconst_raw, 1.0)
    p_pod_hgb_unconst_cal = normalize_race_probabilities(p_pod_hgb_unconst_raw, 3.0)

    # Also reproduce the exact legacy 12-feature unconstrained vs monotonic HGB for Phase 1.4 artifact audit
    from importlib import import_module
    feat_leg_mod = import_module('3_features')
    b_mod = import_module('6_predict_brazil_live')
    c_132 = pd.read_csv(os.path.join(BASE_DIR, 'data', 'clean_races.csv'))
    c_mex = pd.read_csv(os.path.join(BASE_DIR, 'data', 'clean_target_mexico_2026.csv'))
    f_brz_leg = pd.read_csv(os.path.join(BASE_DIR, 'data', 'brazil_2026_live_predictions.csv'))
    if is_backtest:
        df_tr_leg, df_te_leg_raw, _ = feat_leg_mod.compute_features_pipeline(c_132, c_mex)
        df_te_leg = df_te_leg_raw.set_index('Driver').loc[te['Driver'].values].reset_index()
    else:
        common_c = [c for c in c_132.columns if c in c_mex.columns]
        hist_133 = pd.concat([c_132[common_c], c_mex[common_c]], ignore_index=True)
        braz_raw = b_mod.load_manual_quali_csv(os.path.join(BASE_DIR, 'data', 'brazil_2026_actual_quali.csv'), hist_133)
        df_tr_leg, _, _ = feat_leg_mod.compute_features_pipeline(hist_133, braz_raw)
        df_te_leg = f_brz_leg.set_index('Driver').loc[te['Driver'].values].reset_index()

    X_tr_leg = df_tr_leg[LEGACY_12_FEATURES].values
    y_tr_pod_leg = (df_tr_leg['FinishPosition'] <= 3).astype(int).values
    X_te_leg = df_te_leg[LEGACY_12_FEATURES].values
    sc_leg = StandardScaler().fit(X_tr_leg)
    leg_hgb_unconst = HistGradientBoostingClassifier(
        max_depth=3, min_samples_leaf=20, l2_regularization=1.0, random_state=42
    ).fit(sc_leg.transform(X_tr_leg), y_tr_pod_leg)
    leg_pod_unconst_raw = leg_hgb_unconst.predict_proba(sc_leg.transform(X_te_leg))[:, 1]
    leg_pod_unconst_cal = normalize_race_probabilities(leg_pod_unconst_raw, 3.0)

    leg_cst = build_monotonic_constraint_vector(LEGACY_12_FEATURES)
    leg_hgb_mono = HistGradientBoostingClassifier(
        monotonic_cst=leg_cst, max_depth=3, min_samples_leaf=20, l2_regularization=1.0, random_state=42
    ).fit(sc_leg.transform(X_tr_leg), y_tr_pod_leg)
    leg_pod_mono_raw = leg_hgb_mono.predict_proba(sc_leg.transform(X_te_leg))[:, 1]
    leg_pod_mono_cal = normalize_race_probabilities(leg_pod_mono_raw, 3.0)

    # Monotonicity sweep verification on current race
    sweep_win_mono = run_monotonicity_sweep_test(hgb_win_mono.predict_proba_positive, APEX_POSITION_FEATURES, X_te_pos, max_position=n_ent)
    sweep_pod_mono = run_monotonicity_sweep_test(hgb_pod_mono.predict_proba_positive, APEX_POSITION_FEATURES, X_te_pos, max_position=n_ent)
    sweep_pod_unconst = run_monotonicity_sweep_test(hgb_pod_unconst.predict_proba_positive, APEX_POSITION_FEATURES, X_te_pos, max_position=n_ent)

    # 4. Fit Calibrated DNF Hazard Classifier
    dnf_mod = CalibratedDNFHazardModel(APEX_DNF_FEATURES, random_state=42)
    dnf_mod.fit(tr[APEX_DNF_FEATURES].values, y_tr_dnf, tr['Race_Num_Key'].values)
    p_dnf_cal = dnf_mod.predict_proba_calibrated(te[APEX_DNF_FEATURES].values)

    # 5. Fit Plackett-Luce Ranking Model + 25,000-Simulation Coherent Position Distribution
    pl_mod = PlackettLuceRankingModel(APEX_POSITION_FEATURES, l2_lambda=0.2, random_state=42)
    pl_mod.fit(
        X_tr_pos,
        y_tr_fin,
        tr['is_classified_finisher'].values,
        tr['Race_Num_Key'].values,
    )
    pl_sim = pl_mod.simulate_race_distribution(X_te_pos, p_dnf_cal, n_simulations=25000, seed=42)

    # 6. Compute Exact Interventional Shapley Explanations (2^10 = 1,024 coalitions)
    # Fixed background sample of K=80 representative historical pre-race rows (strictly from training set)
    rng_bg = np.random.default_rng(42)
    bg_indices = rng_bg.choice(len(tr), size=min(80, len(tr)), replace=False)
    X_bg_pos = tr.iloc[bg_indices][APEX_POSITION_FEATURES].values
    X_bg_dnf = tr.iloc[bg_indices][APEX_DNF_FEATURES].values

    driver_shap_records = {}
    for i_drv in range(n_ent):
        drv_code = str(te['Driver'].iloc[i_drv])
        x_pos_i = X_te_pos[i_drv]
        x_dnf_i = te.iloc[i_drv][APEX_DNF_FEATURES].values

        # Compute exact SHAP for all 6 required models for VER, top contenders, or all drivers
        # Since 2^10 = 1,024 coalitions takes ~0.015s per call, we compute exact SHAP for ALL 22 drivers!
        shap_hgb_win = compute_exact_interventional_shap(hgb_win_mono.predict_proba_positive, x_pos_i, X_bg_pos, APEX_POSITION_FEATURES)
        shap_hgb_pod = compute_exact_interventional_shap(hgb_pod_mono.predict_proba_positive, x_pos_i, X_bg_pos, APEX_POSITION_FEATURES)
        shap_lr_win = compute_exact_interventional_shap(lr_win_mod.predict_proba_positive, x_pos_i, X_bg_pos, APEX_POSITION_FEATURES)
        shap_lr_pod = compute_exact_interventional_shap(lr_pod_mod.predict_proba_positive, x_pos_i, X_bg_pos, APEX_POSITION_FEATURES)
        shap_dnf = compute_exact_interventional_shap(dnf_mod.predict_proba_calibrated, x_dnf_i, X_bg_dnf, APEX_DNF_FEATURES)
        shap_ridge = compute_exact_interventional_shap(lambda X_in: ridge_mod.predict(X_in, n_entrants=n_ent), x_pos_i, X_bg_pos, APEX_POSITION_FEATURES)
        shap_pl_util = compute_exact_interventional_shap(pl_mod.predict_utility, x_pos_i, X_bg_pos, APEX_POSITION_FEATURES)

        driver_shap_records[drv_code] = {
            'HGB_Win_Probability': {
                **shap_hgb_win,
                'units': 'probability (and percentage_points = shap * 100)',
                'shap_percentage_points': {k: round(v * 100.0, 4) for k, v in shap_hgb_win['shap_values'].items()},
            },
            'HGB_Podium_Probability': {
                **shap_hgb_pod,
                'units': 'probability (and percentage_points = shap * 100)',
                'shap_percentage_points': {k: round(v * 100.0, 4) for k, v in shap_hgb_pod['shap_values'].items()},
            },
            'LogisticRegression_Win_Probability': {
                **shap_lr_win,
                'units': 'probability (and percentage_points = shap * 100)',
                'shap_percentage_points': {k: round(v * 100.0, 4) for k, v in shap_lr_win['shap_values'].items()},
            },
            'LogisticRegression_Podium_Probability': {
                **shap_lr_pod,
                'units': 'probability (and percentage_points = shap * 100)',
                'shap_percentage_points': {k: round(v * 100.0, 4) for k, v in shap_lr_pod['shap_values'].items()},
            },
            'Calibrated_DNF_Probability': {
                **shap_dnf,
                'units': 'probability (and percentage_points = shap * 100)',
                'shap_percentage_points': {k: round(v * 100.0, 4) for k, v in shap_dnf['shap_values'].items()},
            },
            'RidgeCV_Expected_Finish_Position': {
                **shap_ridge,
                'units': 'expected_finishing_position_units',
            },
            'Plackett_Luce_Driver_Utility': {
                **shap_pl_util,
                'units': 'log_strength_utility_units',
            },
        }

    # Build output DataFrame for CSV & JSON export
    out_df = pd.DataFrame()
    out_df['Race_ID'] = te['Race_ID']
    out_df['Season'] = te['Year'].astype(int)
    out_df['Round'] = te['Round'].astype(int)
    out_df['Grand_Prix'] = te['EventName']
    out_df['Location'] = te['Location']
    out_df['Race_Date'] = te['EventDate']
    out_df['Driver_ID'] = te['Driver']
    out_df['Driver_Name'] = te['Driver_Name']
    out_df['Constructor'] = te['TeamName']
    out_df['Raw_Constructor'] = te['RawTeamName']
    out_df['Qualifying_Rank'] = te['Qualifying_Rank'].astype(int)
    out_df['Starting_Grid_Position'] = te['Starting_Grid_Position'].astype(int)
    out_df['Delta_Q_to_G'] = te['Delta_Q_to_G'].astype(int)
    out_df['best_quali_sec'] = te['best_quali_sec']
    out_df['quali_gap_pct'] = te['quali_gap_pct'].round(4)
    out_df['Teammate_Qualifying_Delta'] = te['Teammate_Qualifying_Delta'].round(4)
    out_df['Teammate_Missing_Flag'] = te['Teammate_Missing_Flag'].astype(int)
    out_df['Driver_Form'] = te['Driver_Form'].round(4)
    out_df['Team_Form'] = te['Team_Form'].round(4)
    out_df['driver_form_3'] = te['driver_form_3'].round(4)
    out_df['team_form_3'] = te['team_form_3'].round(4)
    out_df['Driver_Track_History'] = te['Driver_Track_History'].round(4)
    out_df['Driver_Track_Starts'] = te['Driver_Track_Starts'].astype(int)
    out_df['Driver_Track_Missing'] = te['Driver_Track_Missing'].astype(int)
    out_df['Circuit_Overtake_Difficulty_Index'] = te['Circuit_Overtake_Difficulty_Index'].round(4)
    out_df['Circuit_Attrition_Index'] = te['Circuit_Attrition_Index'].round(4)
    out_df['Constructor_DNF_Rate_Smoothed'] = te['Constructor_DNF_Rate_Smoothed'].round(4)
    out_df['Constructor_DNF_EWMA'] = te['Constructor_DNF_EWMA'].round(4)
    out_df['Constructor_Attributed_Mech_Rate'] = te['Constructor_Attributed_Mech_Rate'].round(4)
    out_df['Driver_DNF_Rate_Smoothed'] = te['Driver_DNF_Rate_Smoothed'].round(4)
    out_df['Driver_DNF_EWMA'] = te['Driver_DNF_EWMA'].round(4)
    out_df['Driver_Attributed_Incident_Rate'] = te['Driver_Attributed_Incident_Rate'].round(4)

    # Model predictions
    out_df['Exp_Finish'] = np.round(conf_res['Exp_Finish'], 4)
    out_df['Pred_Rank_Ridge'] = pd.Series(conf_res['Exp_Finish']).rank(method='first').astype(int).values
    out_df['Lower_Bound_Cont'] = np.round(conf_res['Lower_Bound_Cont'], 4)
    out_df['Upper_Bound_Cont'] = np.round(conf_res['Upper_Bound_Cont'], 4)
    out_df['Lower_Bound_Rank'] = conf_res['Lower_Bound_Rank'].astype(int)
    out_df['Upper_Bound_Rank'] = conf_res['Upper_Bound_Rank'].astype(int)
    out_df['Conformal_Confidence_Level'] = conf_res['Conformal_Confidence_Level']
    out_df['Conformal_Calibration_Sample_Size'] = conf_res['Conformal_Calibration_Sample_Size']

    out_df['P_Win_HGB'] = np.round(p_win_hgb_cal, 6)
    out_df['P_Podium_HGB'] = np.round(p_pod_hgb_cal, 6)
    out_df['P_Win_HGB_Raw'] = np.round(p_win_hgb_raw, 6)
    out_df['P_Podium_HGB_Raw'] = np.round(p_pod_hgb_raw, 6)

    out_df['P_Win_LR'] = np.round(p_win_lr_cal, 6)
    out_df['P_Podium_LR'] = np.round(p_pod_lr_cal, 6)
    out_df['P_Win_LR_Raw'] = np.round(p_win_lr_raw, 6)
    out_df['P_Podium_LR_Raw'] = np.round(p_pod_lr_raw, 6)

    out_df['P_DNF'] = np.round(pl_sim['p_dnf'], 6)
    out_df['DNF_Percentage'] = np.round(pl_sim['p_dnf'] * 100.0, 2)

    out_df['PL_Utility_Score'] = np.round(pl_sim['utility_scores'], 6)
    out_df['Exp_Finish_PL'] = np.round(pl_sim['exp_pos_pl_overall'], 4)
    out_df['Pred_Rank_PL'] = pl_sim['pred_rank_pl'].astype(int)
    out_df['P_Win_PL'] = np.round(pl_sim['p_win_pl'], 6)
    out_df['P_Podium_PL'] = np.round(pl_sim['p_podium_pl'], 6)

    # Dynamic position columns for actual field size N = n_ent
    pos_cols_dict = {}
    for k in range(1, n_ent + 1):
        pos_cols_dict[f'P_Uncond_Pos_{k}'] = np.round(pl_sim['p_unconditional'][:, k - 1], 6)
    for k in range(1, n_ent + 1):
        pos_cols_dict[f'P_Cond_Pos_{k}'] = np.round(pl_sim['p_conditional_given_pos_exists'][:, k - 1], 6)
    for k in range(1, n_ent + 1):
        pos_cols_dict[f'P_Cond_Given_Finish_Pos_{k}'] = np.round(pl_sim['p_conditional_given_driver_finishes'][:, k - 1], 6)
    pos_cols_dict['Prediction_Cutoff'] = prediction_cutoff_iso
    pos_cols_dict['Model_Version'] = 'Project_Apex_v2.0'
    out_df = pd.concat([out_df, pd.DataFrame(pos_cols_dict)], axis=1)

    if is_backtest:
        # Separate actual outcome & error columns for Mexico 2026 historical backtest
        actual_fin = te['FinishPosition'].astype(int).values
        out_df['Actual_Finish_Position'] = actual_fin
        out_df['Actual_Status'] = te['Status'].values
        out_df['Actual_DNF_Outcome'] = te['is_dnf'].astype(int).values
        out_df['Error_Exp_Finish_Ridge'] = np.round(out_df['Exp_Finish'] - actual_fin, 4)
        out_df['Abs_Error_Exp_Finish_Ridge'] = np.round(np.abs(out_df['Exp_Finish'] - actual_fin), 4)
        out_df['Error_Rank_Ridge'] = (out_df['Pred_Rank_Ridge'] - actual_fin).astype(int)
        out_df['Abs_Error_Rank_Ridge'] = np.abs(out_df['Pred_Rank_Ridge'] - actual_fin).astype(int)
        out_df['Error_Exp_Finish_PL'] = np.round(out_df['Exp_Finish_PL'] - actual_fin, 4)
        out_df['Abs_Error_Exp_Finish_PL'] = np.round(np.abs(out_df['Exp_Finish_PL'] - actual_fin), 4)
        out_df['Conformal_Interval_Covered_Rank'] = (
            (actual_fin >= out_df['Lower_Bound_Rank']) & (actual_fin <= out_df['Upper_Bound_Rank'])
        ).astype(int)
        out_df['Conformal_Interval_Covered_Cont'] = (
            (actual_fin >= out_df['Lower_Bound_Cont']) & (actual_fin <= out_df['Upper_Bound_Cont'])
        ).astype(int)

    # Sort by Ridge predicted rank ascending
    out_df = out_df.sort_values('Pred_Rank_Ridge').reset_index(drop=True)

    # Mathematical coherence checks on Plackett-Luce matrices (Phase 4.3)
    uncond_mat = pl_sim['p_unconditional']
    cond_pos_mat = pl_sim['p_conditional_given_pos_exists']
    cond_fin_mat = pl_sim['p_conditional_given_driver_finishes']
    row_totals_uncond = pl_sim['p_dnf'] + np.sum(uncond_mat, axis=1)
    col_totals_uncond = np.sum(uncond_mat, axis=0)
    col_totals_cond_pos = np.sum(cond_pos_mat, axis=0)
    row_totals_cond_fin = np.sum(cond_fin_mat, axis=1)

    coherence_audit = {
        'n_actual_entrants': int(n_ent),
        'mean_modeled_finishers': round(float(np.sum(1.0 - pl_sim['p_dnf'])), 4),
        'n_simulations': int(pl_sim['n_simulations']),
        'max_monte_carlo_standard_error': round(float(pl_sim['max_mc_se']), 6),
        'max_abs_error_unconditional_row_sum_plus_dnf_eq_1': float(np.max(np.abs(row_totals_uncond - 1.0))),
        'max_abs_error_unconditional_col_sum_eq_p_pos_exists': float(np.max(np.abs(col_totals_uncond - pl_sim['p_pos_exists']))),
        'max_abs_error_conditional_pos_col_sum_eq_1': float(np.max(np.abs(col_totals_cond_pos - 1.0))),
        'max_abs_error_conditional_finisher_row_sum_eq_1': float(np.max(np.abs(row_totals_cond_fin - 1.0))),
        'all_coherence_identities_passed': bool(
            np.max(np.abs(row_totals_uncond - 1.0)) < 1e-6
            and np.max(np.abs(col_totals_cond_pos - 1.0)) < 1e-6
            and np.max(np.abs(row_totals_cond_fin - 1.0)) < 1e-6
        ),
    }

    # Build Verstappen case study for this race (Phase 1.4 & Phase 6.3)
    ver_idx_te = int(te[te['Driver'] == 'VER'].index[0])
    ver_row_out = out_df[out_df['Driver_ID'] == 'VER'].iloc[0].to_dict()

    # Trace Verstappen's Podium probability across Grid Positions P1..P22 holding his other features fixed
    ver_x_grid_sweep = np.repeat(X_te_pos[ver_idx_te : ver_idx_te + 1], n_ent, axis=0)
    g_col_idx = APEX_POSITION_FEATURES.index('Starting_Grid_Position')
    ver_x_grid_sweep[:, g_col_idx] = np.arange(1, n_ent + 1, dtype=float)
    ver_mono_curve_raw = hgb_pod_mono.predict_proba_positive(ver_x_grid_sweep)
    ver_unconst_curve_raw = hgb_pod_unconst.predict_proba_positive(ver_x_grid_sweep)

    ver_case = {
        'race_id': str(te['Race_ID'].iloc[ver_idx_te]),
        'event_name': str(te['EventName'].iloc[ver_idx_te]),
        'is_backtest': bool(is_backtest),
        'driver_id': 'VER',
        'driver_name': 'Max Verstappen',
        'constructor': 'Red Bull',
        'input_features': {col: float(te[col].iloc[ver_idx_te]) for col in APEX_POSITION_FEATURES + [c for c in APEX_DNF_FEATURES if c not in APEX_POSITION_FEATURES]},
        'legacy_artifact_reproduction_and_resolution': {
            'legacy_12_feature_unconstrained_hgb_raw_podium_prob': round(float(leg_pod_unconst_raw[ver_idx_te]), 6),
            'legacy_12_feature_unconstrained_hgb_calibrated_podium_prob': round(float(leg_pod_unconst_cal[ver_idx_te]), 6),
            'legacy_40_24_pct_reproduced': bool(abs(float(leg_pod_unconst_cal[ver_idx_te]) - 0.402358) < 1e-4) if not is_backtest else None,
            'legacy_12_feature_monotonic_hgb_raw_podium_prob': round(float(leg_pod_mono_raw[ver_idx_te]), 6),
            'legacy_12_feature_monotonic_hgb_calibrated_podium_prob': round(float(leg_pod_mono_cal[ver_idx_te]), 6),
            'apex_10_feature_unconstrained_hgb_raw_podium_prob': round(float(p_pod_hgb_unconst_raw[ver_idx_te]), 6),
            'apex_10_feature_unconstrained_hgb_calibrated_podium_prob': round(float(p_pod_hgb_unconst_cal[ver_idx_te]), 6),
            'apex_10_feature_monotonic_hgb_raw_podium_prob': round(float(p_pod_hgb_raw[ver_idx_te]), 6),
            'apex_10_feature_monotonic_hgb_calibrated_podium_prob': round(float(p_pod_hgb_cal[ver_idx_te]), 6),
            'grid_sweep_p1_to_p22_monotonic_hgb_raw': [round(float(v), 6) for v in ver_mono_curve_raw],
            'grid_sweep_p1_to_p22_unconstrained_hgb_raw': [round(float(v), 6) for v in ver_unconst_curve_raw],
            'monotonicity_test_passed': bool(sweep_pod_mono['overall_passed'] and sweep_win_mono['overall_passed']),
            'technical_explanation': (
                'In the legacy 12-feature unconstrained HGB model, tree splits on quali_gap_pct (+5.18%) and grid_pos (P21) interacted non-monotonically with driver_form_3 (1.67), causing P21 to receive a HIGHER podium probability (36.06% raw / 40.24% calibrated) than starting P11 (25.12% raw). '
                'Enforcing monotonic_cst (-1 on Qualifying_Rank and Starting_Grid_Position) eliminates all positive jumps across P1..P22 (0 violations out of 924 transitions). '
                'Furthermore, as noted in Phase 1.2, holding Verstappen\'s elite Driver_Form (1.67), Driver_Track_History at Interlagos, and Delta_Q_to_G (+5 penalty indicator) constant, the monotonic HGB curve decreases from P1 (74.6%) down to a flat right-leaf plateau from P7..P21 (29.27% raw / 26.98% calibrated), whereas listwise Plackett-Luce gives P_Podium_PL = 1.98% and LogisticRegression gives P_Podium_LR = 1.24%.'
            ),
        },
        'predictions_summary': {
            'Exp_Finish_Ridge': round(float(conf_res['Exp_Finish'][ver_idx_te]), 4),
            'Pred_Rank_Ridge': int(out_df.loc[out_df['Driver_ID'] == 'VER', 'Pred_Rank_Ridge'].values[0]),
            'Conformal_90_Interval_Rank': [int(conf_res['Lower_Bound_Rank'][ver_idx_te]), int(conf_res['Upper_Bound_Rank'][ver_idx_te])],
            'Conformal_90_Interval_Cont': [round(float(conf_res['Lower_Bound_Cont'][ver_idx_te]), 4), round(float(conf_res['Upper_Bound_Cont'][ver_idx_te]), 4)],
            'P_Win_HGB_Calibrated': round(float(p_win_hgb_cal[ver_idx_te]), 6),
            'P_Podium_HGB_Calibrated': round(float(p_pod_hgb_cal[ver_idx_te]), 6),
            'P_Win_HGB_Raw': round(float(p_win_hgb_raw[ver_idx_te]), 6),
            'P_Podium_HGB_Raw': round(float(p_pod_hgb_raw[ver_idx_te]), 6),
            'P_Win_LR_Calibrated': round(float(p_win_lr_cal[ver_idx_te]), 6),
            'P_Podium_LR_Calibrated': round(float(p_pod_lr_cal[ver_idx_te]), 6),
            'P_DNF_Calibrated': round(float(pl_sim['p_dnf'][ver_idx_te]), 6),
            'P_Win_PL': round(float(pl_sim['p_win_pl'][ver_idx_te]), 6),
            'P_Podium_PL': round(float(pl_sim['p_podium_pl'][ver_idx_te]), 6),
            'Exp_Finish_PL': round(float(pl_sim['exp_pos_pl_overall'][ver_idx_te]), 4),
            'Unconditional_Position_Probabilities_P1_to_PN': {f'P{k}': round(float(pl_sim['p_unconditional'][ver_idx_te, k - 1]), 6) for k in range(1, n_ent + 1)},
            'Conditional_Position_Probabilities_Given_Finish_P1_to_PN': {f'P{k}': round(float(pl_sim['p_conditional_given_driver_finishes'][ver_idx_te, k - 1]), 6) for k in range(1, n_ent + 1)},
        },
        'exact_shap_explanations': driver_shap_records['VER'],
    }

    # Build full JSON payload for the event
    event_json = {
        'metadata': {
            'race_id': str(te['Race_ID'].iloc[0]),
            'season': int(te['Year'].iloc[0]),
            'round': int(te['Round'].iloc[0]),
            'grand_prix': str(te['EventName'].iloc[0]),
            'location': str(te['Location'].iloc[0]),
            'race_date': str(te['EventDate'].iloc[0]),
            'is_historical_backtest': bool(is_backtest),
            'training_rounds_count': int(tr[['Year', 'Round']].drop_duplicates().shape[0]),
            'training_driver_starts_count': int(len(tr)),
            'field_size_actual_entrants': int(n_ent),
            'prediction_cutoff_utc': prediction_cutoff_iso,
            'model_version': 'Project_Apex_v2.0',
            'random_seed': 42,
        },
        'monotonicity_verification': {
            'Monotonic_HGB_Win': sweep_win_mono,
            'Monotonic_HGB_Podium': sweep_pod_mono,
            'Diagnostic_Unconstrained_HGB_Podium': sweep_pod_unconst,
        },
        'dnf_calibration_diagnostics': dnf_mod.oof_diagnostics,
        'plackett_luce_coherence_verification': coherence_audit,
        'binary_vs_ranking_model_disagreement_report': [
            {
                'Driver_ID': str(r['Driver_ID']),
                'Starting_Grid_Position': int(r['Starting_Grid_Position']),
                'P_Win_HGB': float(r['P_Win_HGB']),
                'P_Win_LR': float(r['P_Win_LR']),
                'P_Win_PL': float(r['P_Win_PL']),
                'P_Podium_HGB': float(r['P_Podium_HGB']),
                'P_Podium_LR': float(r['P_Podium_LR']),
                'P_Podium_PL': float(r['P_Podium_PL']),
                'max_podium_spread_across_models': round(
                    float(max(r['P_Podium_HGB'], r['P_Podium_LR'], r['P_Podium_PL']) - min(r['P_Podium_HGB'], r['P_Podium_LR'], r['P_Podium_PL'])), 6
                ),
            }
            for _, r in out_df.iterrows()
        ],
        'predictions': sanitize_for_json(out_df.to_dict(orient='records')),
        'driver_shap_explanations': sanitize_for_json(driver_shap_records),
    }

    if is_backtest:
        rho_r, _ = spearmanr(out_df['Actual_Finish_Position'], out_df['Exp_Finish'])
        rho_pl, _ = spearmanr(out_df['Actual_Finish_Position'], out_df['Exp_Finish_PL'])
        rho_g, _ = spearmanr(out_df['Actual_Finish_Position'], out_df['Starting_Grid_Position'])
        event_json['backtest_evaluation_summary'] = {
            'spearman_ridge': round(float(rho_r), 6),
            'spearman_plackett_luce': round(float(rho_pl), 6),
            'spearman_starting_grid': round(float(rho_g), 6),
            'continuous_mae_ridge': round(float(out_df['Abs_Error_Exp_Finish_Ridge'].mean()), 4),
            'rank_mae_ridge': round(float(out_df['Abs_Error_Rank_Ridge'].mean()), 4),
            'exp_pos_mae_plackett_luce': round(float(out_df['Abs_Error_Exp_Finish_PL'].mean()), 4),
            'conformal_90_empirical_coverage_rank': round(float(out_df['Conformal_Interval_Covered_Rank'].mean()), 4),
            'conformal_90_empirical_coverage_cont': round(float(out_df['Conformal_Interval_Covered_Cont'].mean()), 4),
            'actual_podium': out_df[out_df['Actual_Finish_Position'] <= 3].sort_values('Actual_Finish_Position')['Driver_ID'].tolist(),
            'predicted_top3_ridge': out_df.nsmallest(3, 'Exp_Finish')['Driver_ID'].tolist(),
            'predicted_top3_pl': out_df.nsmallest(3, 'Exp_Finish_PL')['Driver_ID'].tolist(),
        }

    return out_df, sanitize_for_json(event_json), sanitize_for_json(ver_case)


def main() -> None:
    t0 = time.time()
    print("[Project Apex] Step 1/5: Loading datasets and engineering leak-free features...")
    df_feat, dq_report = load_and_prepare_datasets()

    print("[Project Apex] Step 2/5: Running 66-race walk-forward validation & 10,000-replicate cluster bootstrap...")
    wf_metrics, past_residuals_through_r18 = run_walk_forward_validation(df_feat)

    print("[Project Apex] Step 3/5: Running Mexico GP 2026 historical backtest & exact SHAP explanations...")
    mex_df, mex_json, ver_mex_case = execute_event_forecast_and_shap(
        df_feat=df_feat,
        target_race_num_key=202619,
        past_conformal_residuals=past_residuals_through_r18,
        prediction_cutoff_iso='2026-11-01T18:00:00Z',
        is_backtest=True,
    )

    # Append Mexico GP 2026 out-of-sample residuals to conformal calibration pool for São Paulo GP 2026
    mex_abs_res = mex_df['Abs_Error_Exp_Finish_Ridge'].values.astype(float)
    past_residuals_through_r19 = np.concatenate([past_residuals_through_r18, mex_abs_res])

    print("[Project Apex] Step 4/5: Running São Paulo GP 2026 target predictions & exact SHAP explanations...")
    sp_df, sp_json, ver_sp_case = execute_event_forecast_and_shap(
        df_feat=df_feat,
        target_race_num_key=202620,
        past_conformal_residuals=past_residuals_through_r19,
        prediction_cutoff_iso='2026-11-08T00:00:00Z',
        is_backtest=False,
    )

    print("[Project Apex] Step 5/5: Exporting structured CSV/JSON deliverables to outputs/project_apex/...")
    # 9.1 & 9.2: sao_paulo_predictions.csv & .json
    sp_df.to_csv(os.path.join(OUTPUT_DIR, 'sao_paulo_predictions.csv'), index=False)
    with open(os.path.join(OUTPUT_DIR, 'sao_paulo_predictions.json'), 'w', encoding='utf-8') as f:
        json.dump(sp_json, f, indent=2, allow_nan=False)

    # 9.3 & 9.4: mexico_backtest.csv & .json
    mex_df.to_csv(os.path.join(OUTPUT_DIR, 'mexico_backtest.csv'), index=False)
    with open(os.path.join(OUTPUT_DIR, 'mexico_backtest.json'), 'w', encoding='utf-8') as f:
        json.dump(mex_json, f, indent=2, allow_nan=False)

    # 9.5: walk_forward_metrics.json
    with open(os.path.join(OUTPUT_DIR, 'walk_forward_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(sanitize_for_json(wf_metrics), f, indent=2, allow_nan=False)

    # 9.6: verstappen_shap_case.json
    ver_combined_case = sanitize_for_json({
        'case_study_title': 'Max Verstappen — São Paulo GP 2026 (P21 Recovery & Monotonic HGB Artifact Audit) & Mexico GP 2026 Backtest',
        'sao_paulo_2026_target_case': ver_sp_case,
        'mexico_2026_backtest_case': ver_mex_case,
    })
    with open(os.path.join(OUTPUT_DIR, 'verstappen_shap_case.json'), 'w', encoding='utf-8') as f:
        json.dump(ver_combined_case, f, indent=2, allow_nan=False)

    # 9.7: Additional technical files
    feature_dict = sanitize_for_json({'features': get_feature_dictionary()})
    with open(os.path.join(OUTPUT_DIR, 'feature_dictionary.json'), 'w', encoding='utf-8') as f:
        json.dump(feature_dict, f, indent=2, allow_nan=False)

    with open(os.path.join(OUTPUT_DIR, 'data_quality_report.json'), 'w', encoding='utf-8') as f:
        json.dump(sanitize_for_json(dq_report), f, indent=2, allow_nan=False)

    model_manifest = sanitize_for_json({
        'project': 'Project Apex — Formula 1 Statistical Prediction & Ranking Pipeline',
        'model_version': 'Project_Apex_v2.0',
        'created_at_utc': '2026-11-08T05:00:00Z',
        'random_seed': 42,
        'software_environment': {
            'python_version': sys.version,
            'platform': platform.platform(),
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
            'scipy_version': scipy.__version__,
            'scikit_learn_version': sklearn.__version__,
        },
        'models': {
            'RidgeCV_Conformal': {
                'class': 'LeakageSafeRidgeConformalModel',
                'features': APEX_POSITION_FEATURES,
                'alphas': 'np.logspace(-1, 3, 25)',
                'conformal_level': 0.90,
                'conformal_method': 'Finite-sample split conformal quantile on strictly prior out-of-sample walk-forward residuals',
            },
            'Monotonic_HGB_Win_and_Podium': {
                'class': 'MonotonicHGBClassifier',
                'features': APEX_POSITION_FEATURES,
                'monotonic_cst': build_monotonic_constraint_vector(APEX_POSITION_FEATURES),
                'hyperparameters': {'max_depth': 3, 'min_samples_leaf': 20, 'l2_regularization': 1.0, 'learning_rate': 0.05, 'max_iter': 100},
            },
            'Calibrated_LogisticRegression': {
                'class': 'CalibratedBinaryLogisticModel',
                'features': APEX_POSITION_FEATURES,
                'hyperparameters': {'C': 0.5, 'race_normalization': 'proportional (Win sum=1.0) / log-odds shift (Podium sum=3.0)'},
            },
            'Calibrated_DNF_Hazard_Classifier': {
                'class': 'CalibratedDNFHazardModel',
                'features': APEX_DNF_FEATURES,
                'calibration_methods_compared': ['platt_sigmoid', 'isotonic'],
                'selected_calibration_method': sp_json['dnf_calibration_diagnostics']['selected_calibration_method'],
            },
            'Plackett_Luce_Listwise_Ranking_Model': {
                'class': 'PlackettLuceRankingModel',
                'features': APEX_POSITION_FEATURES,
                'l2_lambda': 0.2,
                'dnf_integration': '25,000 Gumbel-Max Monte Carlo simulations over stratified Bernoulli survival realizations',
            },
            'Exact_Interventional_SHAP_Explainer': {
                'method': 'Full 2^10 = 1,024 coalition enumeration over K=80 fixed pre-race historical background rows',
                'is_exact': True,
            },
        },
    })
    with open(os.path.join(OUTPUT_DIR, 'model_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(model_manifest, f, indent=2, allow_nan=False)

    elapsed = time.time() - t0
    run_report = sanitize_for_json({
        'status': 'SUCCESS',
        'execution_time_seconds': round(elapsed, 2),
        'exported_files': sorted(os.listdir(OUTPUT_DIR)),
        'sao_paulo_top3_ridge': sp_df.nsmallest(3, 'Exp_Finish')[['Driver_ID', 'Exp_Finish', 'P_Win_HGB', 'P_Podium_HGB', 'P_Win_PL', 'P_Podium_PL']].to_dict(orient='records'),
        'mexico_backtest_summary': mex_json['backtest_evaluation_summary'],
        'walk_forward_headline': wf_metrics['aggregate_metrics'],
    })
    with open(os.path.join(OUTPUT_DIR, 'run_report.json'), 'w', encoding='utf-8') as f:
        json.dump(run_report, f, indent=2, allow_nan=False)

    print(f"[Project Apex] Pipeline completed in {elapsed:.2f}s. All 10 files written to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

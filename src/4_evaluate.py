"""
src/4_evaluate.py
-----------------
Stages 4 & 5: Walk-Forward Model Evaluation, Baseline Benchmarking,
and Calibrated Win/Podium Probability Models.

Outputs:
  - data/walkforward_predictions_2024_2026.csv
  - data/evaluation_summary.csv
  - outputs/calibration_plot.png
"""

import os
import sys
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.special import logit, expit
from scipy.optimize import brentq
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Configure UTF-8 encoding for Windows console
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

FEATURES_CSV = os.path.join(DATA_DIR, 'features.csv')
WF_PREDICTIONS_CSV = os.path.join(DATA_DIR, 'walkforward_predictions_2024_2026.csv')
EVAL_SUMMARY_CSV = os.path.join(DATA_DIR, 'evaluation_summary.csv')
CALIBRATION_PLOT_PNG = os.path.join(OUTPUTS_DIR, 'calibration_plot.png')

FEATURE_COLUMNS = [
    'grid_pos',
    'quali_pos',
    'qualified_on_pole',
    'pitlane_start',
    'no_quali_time',
    'is_sprint_weekend',
    'quali_gap_pct',
    'teammate_quali_delta',
    'driver_form_3',
    'team_form_3',
    'team_dnf_rate_5',
    'is_new_driver_or_team'
]

HEURISTIC_FEATURES = ['grid_pos', 'team_form_3']


def normalize_race_probs(raw_probs: np.ndarray, target_sum: float = 1.0) -> np.ndarray:
    """
    Normalizes intra-race probabilities so that sum(p) == target_sum
    and each p_i lies strictly within (0.0, 1.0) without clipping to 1.0.
    Uses proportional normalization for target_sum=1.0 and log-odds (logit) shifting
    for target_sum=3.0 (Bregman projection under logistic loss).
    """
    p = np.clip(np.asarray(raw_probs, dtype=float), 1e-5, 1.0 - 1e-5)
    if target_sum == 1.0:
        total = p.sum()
        if total > 0:
            p = p / total
        else:
            p = np.ones_like(p) / len(p)
        return np.clip(p, 1e-5, 1.0 - 1e-5)

    # Log-odds shift so sum(expit(logit(p) + delta)) == target_sum
    lo = logit(p)
    try:
        delta = brentq(lambda d: np.sum(expit(lo + d)) - target_sum, -20.0, 20.0)
        p_norm = expit(lo + delta)
    except Exception:
        p_norm = np.clip(p * (target_sum / p.sum()), 1e-5, 1.0 - 1e-5)
    return np.clip(p_norm, 1e-5, 1.0 - 1e-5)


def compute_empirical_grid_probs(train_df: pd.DataFrame, test_grid_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Laplace-smoothed empirical P(Win) and P(Podium) by starting grid slot
    using strictly past training data (< t), normalized within the test race.
    """
    win_by_grid = train_df.groupby('grid_pos')['y_win'].agg(['sum', 'count'])
    pod_by_grid = train_df.groupby('grid_pos')['y_podium'].agg(['sum', 'count'])

    raw_win = np.zeros(len(test_grid_pos), dtype=float)
    raw_pod = np.zeros(len(test_grid_pos), dtype=float)

    for i, gp in enumerate(test_grid_pos):
        # Fallback to back-of-grid (>=18) if gp not in training (e.g. grid_pos 21, 22 in 2026)
        if gp in win_by_grid.index and win_by_grid.loc[gp, 'count'] >= 5:
            w_sum = win_by_grid.loc[gp, 'sum']
            w_cnt = win_by_grid.loc[gp, 'count']
            p_sum = pod_by_grid.loc[gp, 'sum']
            p_cnt = pod_by_grid.loc[gp, 'count']
        else:
            tail = train_df[train_df['grid_pos'] >= 18]
            w_sum = tail['y_win'].sum()
            w_cnt = len(tail)
            p_sum = tail['y_podium'].sum()
            p_cnt = len(tail)

        # Laplace smoothing
        raw_win[i] = (w_sum + 0.05) / (w_cnt + 1.0)
        raw_pod[i] = (p_sum + 0.15) / (p_cnt + 1.0)

    norm_win = normalize_race_probs(raw_win, target_sum=1.0)
    norm_pod = normalize_race_probs(raw_pod, target_sum=3.0)
    return norm_win, norm_pod


def rank_predictions_in_race(yhat: np.ndarray, grid_pos: np.ndarray, quali_pos: np.ndarray) -> np.ndarray:
    """Converts continuous predictions into unique intra-race integer ranks 1..N."""
    order = np.lexsort((quali_pos, grid_pos, yhat))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(yhat) + 1, dtype=float)
    return ranks


def run_walk_forward(df: pd.DataFrame, exclude_anomaly: bool = True) -> pd.DataFrame:
    """
    Executes strict round-by-round walk-forward evaluation across 2024, 2025, and 2026.
    """
    data = df.copy()
    data['y_win'] = (data['FinishPosition'] == 1.0).astype(int)
    data['y_podium'] = (data['FinishPosition'] <= 3.0).astype(int)

    # Identify chronological rounds
    rounds_df = data[['Year', 'Round', 'EventName', 'EventDate']].drop_duplicates().sort_values(
        by=['Year', 'Round']
    ).reset_index(drop=True)

    # Test rounds: 2024 R1 to 2026 R18
    test_rounds = rounds_df[rounds_df['Year'] >= 2024].copy()

    ridge_alphas = np.logspace(-1, 3, 25)
    all_test_results = []

    for _, r_info in test_rounds.iterrows():
        t_year = int(r_info['Year'])
        t_round = int(r_info['Round'])

        # Strict past split (< t)
        train_mask = (data['Year'] < t_year) | ((data['Year'] == t_year) & (data['Round'] < t_round))
        if exclude_anomaly:
            train_mask = train_mask & (data['is_anomaly_race'] == 0)

        test_mask = (data['Year'] == t_year) & (data['Round'] == t_round)

        train_split = data[train_mask].copy()
        test_split = data[test_mask].copy()

        # Fit StandardScaler strictly on train_split
        scaler_full = StandardScaler()
        X_train_full = scaler_full.fit_transform(train_split[FEATURE_COLUMNS])
        X_test_full = scaler_full.transform(test_split[FEATURE_COLUMNS])

        scaler_heur = StandardScaler()
        X_train_heur = scaler_heur.fit_transform(train_split[HEURISTIC_FEATURES])
        X_test_heur = scaler_heur.transform(test_split[HEURISTIC_FEATURES])

        y_train_pos = train_split['FinishPosition'].values
        grid_test = test_split['grid_pos'].values
        quali_test = test_split['quali_pos'].values

        # --- STAGE 4 MODELS ---
        # 1. Baseline 1: Starting Grid
        pred_b1_grid = grid_test.copy()

        # 2. Baseline 2: Qualifying Rank
        pred_b2_quali = quali_test.copy()

        # 3. Baseline 3: Simple Linear Heuristic (grid_pos + team_form_3)
        heur_model = Ridge(alpha=10.0, random_state=42)
        heur_model.fit(X_train_heur, y_train_pos)
        pred_b3_cont = heur_model.predict(X_test_heur)
        pred_b3_rank = rank_predictions_in_race(pred_b3_cont, grid_test, quali_test)

        # 4. Full Ridge Regression (RidgeCV)
        ridge_cv = RidgeCV(alphas=ridge_alphas)
        ridge_cv.fit(X_train_full, y_train_pos)
        pred_ridge_cont = ridge_cv.predict(X_test_full)
        pred_ridge_rank = rank_predictions_in_race(pred_ridge_cont, grid_test, quali_test)

        # --- STAGE 5 PROBABILITY MODELS ---
        y_train_win = train_split['y_win'].values
        y_train_pod = train_split['y_podium'].values

        # Empirical Grid Baseline Probabilities
        p_win_emp, p_pod_emp = compute_empirical_grid_probs(train_split, grid_test)

        # Model A: Logistic Regression
        lr_win = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr_win.fit(X_train_full, y_train_win)
        p_win_lr = normalize_race_probs(lr_win.predict_proba(X_test_full)[:, 1], target_sum=1.0)

        lr_pod = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr_pod.fit(X_train_full, y_train_pod)
        p_pod_lr = normalize_race_probs(lr_pod.predict_proba(X_test_full)[:, 1], target_sum=3.0)

        # Model B: HistGradientBoostingClassifier
        hgb_win = HistGradientBoostingClassifier(
            max_depth=3, min_samples_leaf=20, l2_regularization=1.0, random_state=42
        )
        hgb_win.fit(X_train_full, y_train_win)
        p_win_hgb = normalize_race_probs(hgb_win.predict_proba(X_test_full)[:, 1], target_sum=1.0)

        hgb_pod = HistGradientBoostingClassifier(
            max_depth=3, min_samples_leaf=20, l2_regularization=1.0, random_state=42
        )
        hgb_pod.fit(X_train_full, y_train_pod)
        p_pod_hgb = normalize_race_probs(hgb_pod.predict_proba(X_test_full)[:, 1], target_sum=3.0)

        # Store predictions
        res_df = test_split[['Year', 'Round', 'EventName', 'EventDate', 'Driver', 'TeamName',
                             'grid_pos', 'quali_pos', 'FinishPosition', 'is_dnf', 'y_win', 'y_podium']].copy()
        res_df['pred_b1_grid'] = pred_b1_grid
        res_df['pred_b2_quali'] = pred_b2_quali
        res_df['pred_b3_heur_cont'] = pred_b3_cont
        res_df['pred_b3_heur_rank'] = pred_b3_rank
        res_df['pred_ridge_cont'] = pred_ridge_cont
        res_df['pred_ridge_rank'] = pred_ridge_rank
        res_df['ridge_best_alpha'] = ridge_cv.alpha_

        res_df['p_win_emp'] = p_win_emp
        res_df['p_win_lr'] = p_win_lr
        res_df['p_win_hgb'] = p_win_hgb

        res_df['p_pod_emp'] = p_pod_emp
        res_df['p_pod_lr'] = p_pod_lr
        res_df['p_pod_hgb'] = p_pod_hgb

        all_test_results.append(res_df)

    return pd.concat(all_test_results, ignore_index=True)


def evaluate_position_predictions(wf_df: pd.DataFrame, model_col: str, baseline_col: str = 'pred_b1_grid',
                                  n_boot: int = 1000, seed: int = 42) -> Dict[str, float]:
    """
    Computes Race-Averaged MAE, Paired Delta MAE vs Starting Grid with 95% Bootstrap CI,
    Mean Spearman rho, Top-3 Podium Hit Rate (%), and Winner Hit Rate (%).
    """
    race_maes = []
    base_maes = []
    spearmans = []
    top3_hits = []
    win_hits = []

    for (yr, rnd), grp in wf_df.groupby(['Year', 'Round']):
        y_true = grp['FinishPosition'].values
        y_pred = grp[model_col].values
        y_base = grp[baseline_col].values
        grid_p = grp['grid_pos'].values
        quali_p = grp['quali_pos'].values

        race_maes.append(np.mean(np.abs(y_true - y_pred)))
        base_maes.append(np.mean(np.abs(y_true - y_base)))

        rho, _ = spearmanr(y_true, y_pred)
        spearmans.append(rho if not np.isnan(rho) else 0.0)

        # Deterministic ranking of y_pred for Top-1 and Top-3 hit rates
        pred_order = np.lexsort((quali_p, grid_p, y_pred))
        pred_top3_drivers = set(grp.iloc[pred_order[:3]]['Driver'])
        true_top3_drivers = set(grp[grp['FinishPosition'] <= 3.0]['Driver'])
        top3_hits.append((len(pred_top3_drivers & true_top3_drivers) / 3.0) * 100.0)

        pred_winner = grp.iloc[pred_order[0]]['Driver']
        true_winners = set(grp[grp['FinishPosition'] == 1.0]['Driver'])
        win_hits.append(100.0 if pred_winner in true_winners else 0.0)

    race_maes = np.array(race_maes)
    base_maes = np.array(base_maes)
    diffs = race_maes - base_maes  # Negative means better than grid baseline

    rng = np.random.default_rng(seed)
    n_races = len(diffs)
    boot_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_races, size=n_races)
        boot_means[b] = np.mean(diffs[idx])

    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return {
        'n_races': n_races,
        'MAE': float(np.mean(race_maes)),
        'Delta_MAE_vs_Grid': float(np.mean(diffs)),
        'CI_95_Low': float(ci_low),
        'CI_95_High': float(ci_high),
        'Spearman_Rho': float(np.mean(spearmans)),
        'Top3_Podium_Hit_Pct': float(np.mean(top3_hits)),
        'Winner_Hit_Pct': float(np.mean(win_hits))
    }


def evaluate_probability_predictions(wf_df: pd.DataFrame, prob_col: str, target_col: str) -> Dict[str, float]:
    """Computes Brier Score and Log Loss on out-of-sample probability predictions."""
    y_true = wf_df[target_col].values
    y_prob = np.clip(wf_df[prob_col].values, 1e-6, 1.0 - 1e-6)
    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    return {
        'Brier_Score': float(brier),
        'Log_Loss': float(ll)
    }


def generate_calibration_plot(wf_df: pd.DataFrame, save_path: str):
    """Generates and saves a two-panel Reliability / Calibration Plot for P(Win) and P(Podium)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)

    # Panel 1: P(Win)
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', alpha=0.7)
    for col, lbl, color, marker in [
        ('p_win_emp', 'Empirical Grid Baseline', '#7f8c8d', '^'),
        ('p_win_lr', 'Logistic Regression (L2)', '#2980b9', 'o'),
        ('p_win_hgb', 'HistGradientBoosting', '#e67e22', 's')
    ]:
        prob_true, prob_pred = calibration_curve(wf_df['y_win'], np.clip(wf_df[col], 0, 1), n_bins=7, strategy='uniform')
        brier = brier_score_loss(wf_df['y_win'], wf_df[col])
        ax.plot(prob_pred, prob_true, marker=marker, color=color, linewidth=2, label=f"{lbl} (Brier={brier:.4f})")

    ax.set_title('Win Probability Calibration — P(Win) [2024–2026]', fontsize=12, fontweight='bold')
    ax.set_xlabel('Mean Predicted Probability', fontsize=10)
    ax.set_ylabel('Observed Fraction of Positives', fontsize=10)
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=9)

    # Panel 2: P(Podium)
    ax = axes[1]
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', alpha=0.7)
    for col, lbl, color, marker in [
        ('p_pod_emp', 'Empirical Grid Baseline', '#7f8c8d', '^'),
        ('p_pod_lr', 'Logistic Regression (L2)', '#27ae60', 'o'),
        ('p_pod_hgb', 'HistGradientBoosting', '#c0392b', 's')
    ]:
        prob_true, prob_pred = calibration_curve(wf_df['y_podium'], np.clip(wf_df[col], 0, 1), n_bins=8, strategy='uniform')
        brier = brier_score_loss(wf_df['y_podium'], wf_df[col])
        ax.plot(prob_pred, prob_true, marker=marker, color=color, linewidth=2, label=f"{lbl} (Brier={brier:.4f})")

    ax.set_title('Podium Probability Calibration — P(Podium) [2024–2026]', fontsize=12, fontweight='bold')
    ax.set_xlabel('Mean Predicted Probability', fontsize=10)
    ax.set_ylabel('Observed Fraction of Positives', fontsize=10)
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    print("=================================================================")
    print("STAGES 4 & 5: WALK-FORWARD MODEL EVALUATION & PROBABILITY MODELS")
    print("=================================================================")

    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded features dataset: {len(df)} rows across {len(df.groupby(['Year', 'Round']))} rounds.")

    # 1. Ablation Check: Spa 2021 Anomaly Race (Exclude vs. Keep)
    print("\n--- 1. Spa 2021 Anomaly Race Ablation Check (66 Test Rounds: 2024–2026) ---")
    wf_excl = run_walk_forward(df, exclude_anomaly=True)
    wf_keep = run_walk_forward(df, exclude_anomaly=False)

    m_excl_cont = evaluate_position_predictions(wf_excl, 'pred_ridge_cont')['MAE']
    m_keep_cont = evaluate_position_predictions(wf_keep, 'pred_ridge_cont')['MAE']
    m_excl_rank = evaluate_position_predictions(wf_excl, 'pred_ridge_rank')['MAE']
    m_keep_rank = evaluate_position_predictions(wf_keep, 'pred_ridge_rank')['MAE']

    print(f"Excluding Spa 2021 (is_anomaly_race == 1): Ridge Continuous MAE = {m_excl_cont:.4f} | Ridge Rank MAE = {m_excl_rank:.4f}")
    print(f"Keeping Spa 2021   (is_anomaly_race == 1): Ridge Continuous MAE = {m_keep_cont:.4f} | Ridge Rank MAE = {m_keep_rank:.4f}")

    # Choose the cleaner option (excluding the 1-lap behind-Safety-Car anomaly)
    use_excl = (m_excl_cont <= m_keep_cont)
    wf_df = wf_excl if use_excl else wf_keep
    print(f"Selected Training Protocol: {'Exclude Spa 2021 Anomaly (Cleaner & Lower/Equal MAE)' if use_excl else 'Keep Spa 2021'}")

    # Save round-by-round walk-forward predictions
    wf_df.to_csv(WF_PREDICTIONS_CSV, index=False)
    print(f"Saved walk-forward predictions ({len(wf_df)} rows, 66 rounds) to: {WF_PREDICTIONS_CSV}")

    # 2. Stage 4 Position Prediction Evaluation
    models_to_eval = [
        ('Baseline 1: Starting Grid (grid_pos)', 'pred_b1_grid'),
        ('Baseline 2: Qualifying Rank (quali_pos)', 'pred_b2_quali'),
        ('Baseline 3: Heuristic [grid + team_form_3] (Continuous)', 'pred_b3_heur_cont'),
        ('Baseline 3: Heuristic [grid + team_form_3] (Rank-Ordered)', 'pred_b3_heur_rank'),
        ('Full RidgeCV (Continuous yhat)', 'pred_ridge_cont'),
        ('Full RidgeCV (Intra-Race Rank-Ordered)', 'pred_ridge_rank')
    ]

    summary_rows = []
    for slice_name, sub_df in [('Overall_2024_2026 (66 Rounds)', wf_df),
                               ('Season_2026_Standalone (18 Rounds)', wf_df[wf_df['Year'] == 2026])]:
        print(f"\n=================================================================")
        print(f"STAGE 4 EVALUATION TABLE — {slice_name}")
        print("=================================================================")
        rows_slice = []
        for m_name, m_col in models_to_eval:
            stats = evaluate_position_predictions(sub_df, m_col, baseline_col='pred_b1_grid')
            row_dict = {
                'Evaluation_Slice': slice_name,
                'Model': m_name,
                'N_Races': stats['n_races'],
                'MAE': round(stats['MAE'], 4),
                'Delta_MAE_vs_Grid': round(stats['Delta_MAE_vs_Grid'], 4),
                '95%_CI': f"[{stats['CI_95_Low']:+.4f}, {stats['CI_95_High']:+.4f}]",
                'Spearman_Rho': round(stats['Spearman_Rho'], 4),
                'Top3_Hit_%': round(stats['Top3_Podium_Hit_Pct'], 2),
                'Winner_Hit_%': round(stats['Winner_Hit_Pct'], 2)
            }
            rows_slice.append(row_dict)
            summary_rows.append(row_dict)
        print(pd.DataFrame(rows_slice).drop(columns=['Evaluation_Slice']).to_string(index=False))

    # 3. Stage 5 Probability Evaluation
    prob_models = [
        ('P(Win)', 'y_win', 'Empirical Grid Baseline', 'p_win_emp'),
        ('P(Win)', 'y_win', 'Model A: Logistic Regression (L2, C=1.0)', 'p_win_lr'),
        ('P(Win)', 'y_win', 'Model B: HistGradientBoosting (max_depth=3)', 'p_win_hgb'),
        ('P(Podium)', 'y_podium', 'Empirical Grid Baseline', 'p_pod_emp'),
        ('P(Podium)', 'y_podium', 'Model A: Logistic Regression (L2, C=1.0)', 'p_pod_lr'),
        ('P(Podium)', 'y_podium', 'Model B: HistGradientBoosting (max_depth=3)', 'p_pod_hgb')
    ]

    prob_rows = []
    for slice_name, sub_df in [('Overall_2024_2026 (66 Rounds)', wf_df),
                               ('Season_2026_Standalone (18 Rounds)', wf_df[wf_df['Year'] == 2026])]:
        print(f"\n=================================================================")
        print(f"STAGE 5 PROBABILITY EVALUATION TABLE — {slice_name}")
        print("=================================================================")
        p_slice = []
        for target_lbl, target_col, m_name, p_col in prob_models:
            p_stats = evaluate_probability_predictions(sub_df, p_col, target_col)
            r_dict = {
                'Evaluation_Slice': slice_name,
                'Target': target_lbl,
                'Model': m_name,
                'Brier_Score': round(p_stats['Brier_Score'], 5),
                'Log_Loss': round(p_stats['Log_Loss'], 5)
            }
            p_slice.append(r_dict)
            prob_rows.append(r_dict)
        print(pd.DataFrame(p_slice).drop(columns=['Evaluation_Slice']).to_string(index=False))

    # Save summary CSV
    pd.DataFrame(summary_rows).to_csv(EVAL_SUMMARY_CSV, index=False)

    # Generate Calibration Plot
    generate_calibration_plot(wf_df, CALIBRATION_PLOT_PNG)
    print(f"\nSaved Calibration / Reliability Plot to: {CALIBRATION_PLOT_PNG}")

    # 4. Full Dataset RidgeCV Fit & Standardized Coefficients
    print("\n=================================================================")
    print("STANDARDIZED RIDGECV COEFFICIENTS (FULL PRE-MEXICO DATASET)")
    print("=================================================================")
    full_train = df[df['is_anomaly_race'] == 0].copy() if use_excl else df.copy()
    scaler_final = StandardScaler()
    X_full = scaler_final.fit_transform(full_train[FEATURE_COLUMNS])
    y_full = full_train['FinishPosition'].values

    ridge_final = RidgeCV(alphas=np.logspace(-1, 3, 25))
    ridge_final.fit(X_full, y_full)
    print(f"Optimal Regularization Alpha (alpha*): {ridge_final.alpha_:.4f}")
    print(f"Model Intercept (Mean Finish Position): {ridge_final.intercept_:.4f}\n")

    coef_df = pd.DataFrame({
        'Feature': FEATURE_COLUMNS,
        'Std_Coefficient': ridge_final.coef_,
        'Abs_Importance': np.abs(ridge_final.coef_)
    }).sort_values(by='Abs_Importance', ascending=False).reset_index(drop=True)
    print(coef_df[['Feature', 'Std_Coefficient']].to_string(index=False))


if __name__ == '__main__':
    main()

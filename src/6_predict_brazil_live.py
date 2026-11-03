"""
src/6_predict_brazil_live.py
----------------------------
Stage 7: Live & Pre-Weekend Prediction Pipeline for the 2026 São Paulo (Brazil) Grand Prix (Round 20).

Combines data/clean_races.csv (132 rounds, 2,654 rows) + data/clean_target_mexico_2026.csv (Round 19, 22 rows)
= 133 historical rounds (2,676 driver-starts) to fit the production models and compute the exact lagged
rolling form (driver_form_3, team_form_3, team_dnf_rate_5) entering Round 20 (Nov 8, 2026).

Usage:
  1. Pre-Qualifying Readiness & Power Rankings Test (before Saturday Qualifying):
       python src/6_predict_brazil_live.py --dry-run
  2. Live Post-Qualifying Prediction (once 2026 Round 20 Qualifying finishes):
       python src/6_predict_brazil_live.py
  3. Live Prediction with Starting Grid Penalty Overrides:
       python src/6_predict_brazil_live.py --grid-overrides "VER:10,LAW:22"
"""

import os
import sys
import argparse
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.optimize import brentq
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configure UTF-8 encoding for console output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
CACHE_DIR = os.path.join(BASE_DIR, 'f1_cache')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

CLEAN_RACES_CSV = os.path.join(DATA_DIR, 'clean_races.csv')
CLEAN_MEXICO_CSV = os.path.join(DATA_DIR, 'clean_target_mexico_2026.csv')
LIVE_OUTPUT_CSV = os.path.join(DATA_DIR, 'brazil_2026_live_predictions.csv')
LIVE_PLOT_PNG = os.path.join(OUTPUTS_DIR, 'brazil_2026_quali_vs_predicted.png')

# Import cleaning & feature engineering helpers
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
from importlib import import_module
clean_mod = import_module('2_clean_data')
feat_mod = import_module('3_features')

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


def normalize_race_probs(raw_probs: np.ndarray, target_sum: float = 1.0) -> np.ndarray:
    """Normalizes probabilities so sum(p) == target_sum using log-odds shift for target_sum > 1.0."""
    p = np.clip(np.asarray(raw_probs, dtype=float), 1e-5, 1.0 - 1e-5)
    if target_sum == 1.0:
        total = p.sum()
        return p / total if total > 0 else np.ones_like(p) / len(p)

    lo = logit(p)
    try:
        delta = brentq(lambda d: np.sum(expit(lo + d)) - target_sum, -20.0, 20.0)
        p_norm = expit(lo + delta)
    except Exception:
        p_norm = p * (target_sum / p.sum())
    return np.clip(p_norm, 1e-5, 1.0 - 1e-5)


def rank_predictions_in_race(yhat: np.ndarray, grid_pos: np.ndarray, quali_pos: np.ndarray) -> np.ndarray:
    """Converts continuous predictions into unique intra-race integer ranks 1..N."""
    order = np.lexsort((quali_pos, grid_pos, yhat))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(yhat) + 1, dtype=float)
    return ranks


def parse_grid_overrides(overrides_str: str) -> dict:
    """Parses '--grid-overrides \"VER:10,LAW:22\"' into {'VER': 10.0, 'LAW': 22.0}."""
    overrides = {}
    if not overrides_str or not overrides_str.strip():
        return overrides
    for item in overrides_str.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(f"Invalid grid override '{item}'. Expected format 'DRIVER:GRID_POS' (e.g. 'VER:10').")
        drv, pos_str = item.split(':', 1)
        overrides[drv.strip().upper()] = float(pos_str.strip())
    return overrides


def apply_grid_overrides(df_race: pd.DataFrame, overrides: dict) -> pd.DataFrame:
    """
    Applies exact starting grid overrides (e.g. engine/gearbox grid penalties or pitlane starts).
    Overridden drivers are placed at their exact requested grid slot in 1..N (or at the back
    with pitlane_start=1 if override <= 0), while non-overridden drivers fill the remaining
    slots in their original relative order.
    """
    if not overrides:
        return df_race

    out = df_race.copy().sort_values(by=['grid_pos', 'quali_pos', 'Driver']).reset_index(drop=True)
    n_starters = len(out)

    valid_overrides = {}
    for drv, new_pos in overrides.items():
        if drv not in out['Driver'].values:
            print(f"  [WARNING] Override driver '{drv}' not found in Brazil 2026 entry list.")
            continue
        valid_overrides[drv] = new_pos

    if not valid_overrides:
        return out

    assigned_slots = {}
    taken_slots = set()

    # 1. Assign pitlane starts (new_pos <= 0) to the very back slots (N, N-1, ...)
    next_back_slot = n_starters
    for drv, new_pos in valid_overrides.items():
        if new_pos <= 0:
            out.loc[out['Driver'] == drv, 'pitlane_start'] = 1
            while next_back_slot in taken_slots and next_back_slot > 1:
                next_back_slot -= 1
            assigned_slots[drv] = float(next_back_slot)
            taken_slots.add(next_back_slot)

    # 2. Assign explicit grid slots (1..N)
    for drv, new_pos in valid_overrides.items():
        if new_pos > 0:
            slot = int(np.clip(round(new_pos), 1, n_starters))
            while slot in taken_slots and slot < n_starters:
                slot += 1
            while slot in taken_slots and slot > 1:
                slot -= 1
            assigned_slots[drv] = float(slot)
            taken_slots.add(slot)

    # 3. Fill remaining open slots 1..N in original relative order
    open_slots = [s for s in range(1, n_starters + 1) if s not in taken_slots]
    open_idx = 0
    for idx, row in out.iterrows():
        drv = row['Driver']
        if drv in assigned_slots:
            out.loc[idx, 'grid_pos'] = assigned_slots[drv]
        else:
            out.loc[idx, 'grid_pos'] = float(open_slots[open_idx])
            open_idx += 1

    out['GridPosition'] = out['grid_pos']
    return out


def build_dry_run_brazil_skeleton(history_133_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the 22-driver Brazil 2026 Round 20 pre-qualifying skeleton using each active
    driver's rolling 3-race mean qualifying performance (Rounds 17, 18, 19 of 2026).
    """
    # Active 22 drivers from Round 19 (Mexico 2026)
    r19 = history_133_df[(history_133_df['Year'] == 2026) & (history_133_df['Round'] == 19)].copy()
    recent_3 = history_133_df[(history_133_df['Year'] == 2026) & (history_133_df['Round'].isin([17, 18, 19]))].copy()

    # Compute each race's pole time to get recent 3-race quali_gap_pct
    race_gaps = []
    for (yr, rnd), grp in recent_3.groupby(['Year', 'Round']):
        pole_t = grp['best_quali_sec'].dropna().min()
        g = grp.copy()
        g['_q_gap'] = (((g['best_quali_sec'] - pole_t) / pole_t) * 100.0).clip(0.0, 7.0).fillna(7.0)
        race_gaps.append(g)
    recent_3 = pd.concat(race_gaps, ignore_index=True)

    drv_quali_stats = recent_3.groupby('Driver', as_index=False).agg(
        mean_quali_pos=('quali_pos', 'mean'),
        mean_quali_gap=('_q_gap', 'mean')
    )

    brazil = r19[['Driver', 'TeamName', 'RawTeamName']].merge(drv_quali_stats, on='Driver', how='left')
    brazil.sort_values(by=['mean_quali_pos', 'mean_quali_gap', 'Driver'], inplace=True)
    brazil.reset_index(drop=True, inplace=True)

    n_drivers = len(brazil)
    brazil['Year'] = 2026
    brazil['Round'] = 20
    brazil['EventName'] = 'São Paulo Grand Prix'
    brazil['Location'] = 'São Paulo'
    brazil['EventDate'] = '2026-11-08'
    brazil['EventFormat'] = 'conventional'

    # Baseline qualifying rank 1..22 and synthetic best_quali_sec anchored to a 70.000s Interlagos pole
    brazil['quali_pos'] = np.arange(1, n_drivers + 1, dtype=float)
    brazil['grid_pos'] = brazil['quali_pos'].copy()
    brazil['GridPosition'] = brazil['grid_pos'].copy()
    brazil['QualiPosition'] = brazil['quali_pos'].copy()

    # Shift gap so P1 has 0.000% gap (70.000s pole reference) while preserving every driver's true 3-race mean quali gap
    min_gap = brazil['mean_quali_gap'].min()
    rel_gap_pct = (brazil['mean_quali_gap'] - min_gap).clip(lower=0.0, upper=7.0).values
    rel_gap_pct[0] = 0.0
    brazil['best_quali_sec'] = 70.0 * (1.0 + rel_gap_pct / 100.0)
    brazil['Q1_sec'] = brazil['best_quali_sec']
    brazil['Q2_sec'] = np.nan
    brazil['Q3_sec'] = np.nan

    brazil['pitlane_start'] = 0
    brazil['no_quali_time'] = 0
    brazil['FinishPosition'] = np.nan
    brazil['Status'] = 'Pending'
    brazil['Points'] = np.nan
    brazil['is_dnf'] = np.nan
    brazil['is_anomaly_race'] = 0
    brazil['FetchedAtUTC'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    return brazil


def fetch_live_brazil_qualifying() -> pd.DataFrame:
    """
    Fetches live 2026 Round 20 (São Paulo Grand Prix) Qualifying ('Q') results via FastF1
    and cleans them using Stage 2 standardization rules.
    """
    import fastf1
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    print("Connecting to FastF1 for 2026 Round 20 (São Paulo Grand Prix) Qualifying ('Q')...")
    session = fastf1.get_session(2026, 20, 'Q')
    session.load(laps=False, telemetry=False, weather=False, messages=False)

    if session.results is None or session.results.empty:
        raise RuntimeError(
            "2026 São Paulo GP Qualifying ('Q') results are not yet available in FastF1. "
            "Run with '--dry-run' before Qualifying, or run again after Qualifying concludes on Nov 7, 2026."
        )

    res = session.results.copy()
    if res['Position'].isna().all() and res['Q1'].isna().all():
        raise RuntimeError(
            "2026 São Paulo GP Qualifying ('Q') session exists in schedule but contains no recorded lap times yet. "
            "Run with '--dry-run' for Pre-Weekend Power Rankings."
        )

    rows = []
    fetched_utc = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    for _, r in res.iterrows():
        drv = str(r.get('Abbreviation', '')).strip()
        if not drv:
            continue
        raw_team = str(r.get('TeamName', '')).strip()
        q1_s = r['Q1'].total_seconds() if pd.notna(r.get('Q1')) else np.nan
        q2_s = r['Q2'].total_seconds() if pd.notna(r.get('Q2')) else np.nan
        q3_s = r['Q3'].total_seconds() if pd.notna(r.get('Q3')) else np.nan
        valid_qs = [t for t in [q1_s, q2_s, q3_s] if pd.notna(t)]
        best_q = min(valid_qs) if valid_qs else np.nan
        q_pos = float(r['Position']) if pd.notna(r.get('Position')) else np.nan

        rows.append({
            'Year': 2026,
            'Round': 20,
            'EventName': 'São Paulo Grand Prix',
            'Location': 'São Paulo',
            'EventDate': '2026-11-08',
            'EventFormat': 'conventional',
            'Driver': drv,
            'RawTeamName': raw_team,
            'TeamName': clean_mod.TEAM_MAPPING.get(raw_team, raw_team),
            'GridPosition': q_pos,
            'QualiPosition': q_pos,
            'Q1_sec': q1_s,
            'Q2_sec': q2_s,
            'Q3_sec': q3_s,
            'best_quali_sec': best_q,
            'FinishPosition': np.nan,
            'Status': 'Pending',
            'Points': np.nan,
            'is_dnf': np.nan,
            'pitlane_start': 0,
            'no_quali_time': 0 if pd.notna(best_q) else 1,
            'is_anomaly_race': 0,
            'FetchedAtUTC': fetched_utc
        })

    df_live = pd.DataFrame(rows)
    n_drivers = len(df_live)
    df_live.sort_values(by=['QualiPosition', 'best_quali_sec', 'Driver'], na_position='last', inplace=True)
    df_live['quali_pos'] = np.arange(1, n_drivers + 1, dtype=float)
    df_live['grid_pos'] = df_live['quali_pos'].copy()
    df_live.reset_index(drop=True, inplace=True)
    return df_live


def plot_brazil_predictions(df_pred: pd.DataFrame, save_path: str, is_dry_run: bool = False):
    """Generates a scatter plot of Qualifying Gap (%) vs. Predicted Finish Position for Brazil 2026."""
    fig, ax = plt.subplots(figsize=(11.5, 7.5), dpi=150)

    ax.set_ylim(23.2, 0.2)
    ax.set_yticks(np.arange(1, 23))
    max_gap = max(4.5, float(df_pred['quali_gap_pct'].max()) + 0.8)
    ax.set_xlim(-0.25, max_gap)

    top3_drivers = set(df_pred.nsmallest(3, 'pred_ridge_rank')['Driver'])
    other_drivers = df_pred[~df_pred['Driver'].isin(top3_drivers)]

    ax.scatter(
        other_drivers['quali_gap_pct'], other_drivers['pred_ridge_rank'],
        color='#34495e', s=85, alpha=0.78, edgecolors='black', linewidth=0.8,
        label='Projected Field (P4–P22)', zorder=3
    )

    podium_palette = ['#f1c40f', '#e67e22', '#2ecc71']
    for idx, (_, row) in enumerate(df_pred.nsmallest(3, 'pred_ridge_rank').iterrows()):
        drv = row['Driver']
        ax.scatter(
            row['quali_gap_pct'], row['pred_ridge_rank'],
            color=podium_palette[idx], s=220, marker='*', edgecolors='black', linewidth=1.2,
            label=f"Projected P{int(row['pred_ridge_rank'])}: {drv} ({row['TeamName']}) | P(Pod)={row['p_pod_hgb']*100:.1f}%",
            zorder=4
        )

    for _, row in df_pred.iterrows():
        drv = row['Driver']
        x = row['quali_gap_pct']
        y = row['pred_ridge_rank']
        pred_y = int(y)
        label = f"{drv} (Grid P{int(row['grid_pos'])} → Pred P{pred_y} | Pod {row['p_pod_hgb']*100:.0f}%)"
        fontweight = 'bold' if drv in top3_drivers else 'normal'
        ax.annotate(
            label, (x, y), xytext=(x + 0.08, y),
            fontsize=8.5, fontweight=fontweight, color='#111111',
            arrowprops=dict(arrowstyle='-', color='#888888', lw=0.5, alpha=0.6)
        )

    mode_subtitle = "Pre-Weekend Readiness Simulation (3-Race Rolling Quali Baseline)" if is_dry_run else "Live Post-Qualifying Prediction"
    ax.set_title(
        f"2026 São Paulo Grand Prix (Brazil): Qualifying Gap (%) vs. Predicted Finish Position\n"
        f"{mode_subtitle} | Trained on 133 Historical Rounds (2021–2026 R19)",
        fontsize=11.5, fontweight='bold', pad=12, color='#1a252f'
    )
    ax.set_xlabel("Qualifying Gap to Pole (%) [Capped at 7.0%]", fontsize=10.5, labelpad=8)
    ax.set_ylabel("Predicted Finish Position (P1 = Winner, Top)", fontsize=10.5, labelpad=8)
    ax.grid(True, linestyle=':', alpha=0.5, zorder=1)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.92, bbox_to_anchor=(0.98, 0.95))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 7: 2026 Brazil GP Live & Pre-Weekend Predictor")
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Run pre-qualifying readiness test and print Pre-Weekend Power Rankings without overwriting live CSV."
    )
    parser.add_argument(
        '--grid-overrides',
        type=str,
        default="",
        help="Optional comma-separated starting grid overrides, e.g. 'VER:10,LAW:22' (use 0 for pitlane start)."
    )
    args = parser.parse_args()

    print("=================================================================")
    print("STAGE 7: 2026 SÃO PAULO (BRAZIL) GRAND PRIX PREDICTION PIPELINE")
    print("=================================================================")

    # 1. Load and combine all 133 historical rounds (132 pre-Mexico + Round 19 Mexico 2026)
    clean_132 = pd.read_csv(CLEAN_RACES_CSV)
    clean_mex = pd.read_csv(CLEAN_MEXICO_CSV)

    # Align common columns
    common_cols = [c for c in clean_132.columns if c in clean_mex.columns]
    history_133 = pd.concat([clean_132[common_cols], clean_mex[common_cols]], ignore_index=True)
    n_rounds_hist = len(history_133.groupby(['Year', 'Round']))
    print(f"Combined historical training corpus: {len(history_133)} driver-starts across {n_rounds_hist} rounds (2021 R1 – 2026 R19 Mexico).")
    assert n_rounds_hist == 133 and len(history_133) == 2676, "Expected exactly 133 rounds and 2,676 rows!"

    # 2. Construct Round 20 Brazil input (Dry-Run vs Live Qualifying)
    if args.dry_run:
        print("Mode: --dry-run (Pre-Qualifying Readiness Test using 3-race rolling qualifying baseline R17-R19)")
        brazil_raw = build_dry_run_brazil_skeleton(history_133)
    else:
        print("Mode: LIVE QUALIFYING INGESTION (2026 Round 20 São Paulo GP)")
        brazil_raw = fetch_live_brazil_qualifying()

    # Apply optional grid penalty overrides before feature engineering
    overrides = parse_grid_overrides(args.grid_overrides)
    if overrides:
        print(f"Applying starting grid overrides: {overrides}")
        brazil_raw = apply_grid_overrides(brazil_raw, overrides)

    # 3. Engineer Features across all 133 historical rounds + Round 20 Brazil
    train_feat_df, brazil_feat_df, _ = feat_mod.compute_features_pipeline(history_133, brazil_raw)

    # Verify zero NaNs in training & target feature matrices
    assert train_feat_df[FEATURE_COLUMNS].isna().sum().sum() == 0, "NaNs found in 133-round training features!"
    assert brazil_feat_df[FEATURE_COLUMNS].isna().sum().sum() == 0, "NaNs found in Brazil 2026 features!"

    # 4. Fit Production Models on all 133 Historical Rounds (2,676 rows)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feat_df[FEATURE_COLUMNS])
    y_train_pos = train_feat_df['FinishPosition'].values
    y_train_win = (train_feat_df['FinishPosition'] == 1.0).astype(int).values
    y_train_pod = (train_feat_df['FinishPosition'] <= 3.0).astype(int).values

    ridge_cv = RidgeCV(alphas=np.logspace(-1, 3, 25))
    ridge_cv.fit(X_train, y_train_pos)
    alpha_star = float(ridge_cv.alpha_)
    print(f"Fitted 133-Round RidgeCV optimal alpha*: {alpha_star:.4f}")

    lr_win = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_win.fit(X_train, y_train_win)

    hgb_win = HistGradientBoostingClassifier(max_depth=3, min_samples_leaf=20, l2_regularization=1.0, random_state=42)
    hgb_win.fit(X_train, y_train_win)

    lr_pod = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_pod.fit(X_train, y_train_pod)

    hgb_pod = HistGradientBoostingClassifier(max_depth=3, min_samples_leaf=20, l2_regularization=1.0, random_state=42)
    hgb_pod.fit(X_train, y_train_pod)

    # 5. Predict Round 20 (São Paulo Grand Prix)
    X_brazil = scaler.transform(brazil_feat_df[FEATURE_COLUMNS])
    grid_bz = brazil_feat_df['grid_pos'].values
    quali_bz = brazil_feat_df['quali_pos'].values

    pred_ridge_cont = ridge_cv.predict(X_brazil)
    pred_ridge_rank = rank_predictions_in_race(pred_ridge_cont, grid_bz, quali_bz)

    p_win_lr = normalize_race_probs(lr_win.predict_proba(X_brazil)[:, 1], target_sum=1.0)
    p_win_hgb = normalize_race_probs(hgb_win.predict_proba(X_brazil)[:, 1], target_sum=1.0)
    p_pod_lr = normalize_race_probs(lr_pod.predict_proba(X_brazil)[:, 1], target_sum=3.0)
    p_pod_hgb = normalize_race_probs(hgb_pod.predict_proba(X_brazil)[:, 1], target_sum=3.0)

    pred_df = brazil_feat_df.copy()
    pred_df['pred_ridge_cont'] = pred_ridge_cont
    pred_df['pred_ridge_rank'] = pred_ridge_rank
    pred_df['p_win_lr'] = p_win_lr
    pred_df['p_win_hgb'] = p_win_hgb
    pred_df['p_pod_hgb'] = p_pod_hgb
    pred_df['p_pod_lr'] = p_pod_lr
    pred_df['PredictedAtUTC'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    pred_df.sort_values(by='pred_ridge_rank', inplace=True)
    pred_df.reset_index(drop=True, inplace=True)

    # 6. Save Live Artifacts (Only in Live Mode so Dry-Run never overwrites live predictions)
    if not args.dry_run:
        pred_df.to_csv(LIVE_OUTPUT_CSV, index=False)
        print(f"Saved live Brazil 2026 predictions to: {LIVE_OUTPUT_CSV}")
        plot_brazil_predictions(pred_df, LIVE_PLOT_PNG, is_dry_run=False)
    else:
        print("Dry-run complete: skipped writing data/brazil_2026_live_predictions.csv (reserved for live post-Q run).")

    # 7. Terminal Output Report
    banner_title = (
        "🇧🇷 PRE-WEEKEND FORM & POWER RANKINGS ENTERING BRAZIL 2026 (ROUND 20 — DRY RUN) 🇧🇷"
        if args.dry_run else
        "🇧🇷 LIVE POST-QUALIFYING PREDICTIONS — 2026 SÃO PAULO GRAND PRIX (ROUND 20) 🇧🇷"
    )
    print("\n" + "=" * 95)
    print(banner_title)
    print("=" * 95)
    top3 = pred_df.head(3)
    for _, r in top3.iterrows():
        print(
            f"  Projected P{int(r['pred_ridge_rank'])}: {r['Driver']} ({r['TeamName']}) | "
            f"Grid: P{int(r['grid_pos'])} | Exp Pos: {r['pred_ridge_cont']:.2f} | "
            f"P(Win) [LR/HGB]: {r['p_win_lr']*100:.1f}% / {r['p_win_hgb']*100:.1f}% | "
            f"P(Podium) [HGB/LR]: {r['p_pod_hgb']*100:.1f}% / {r['p_pod_lr']*100:.1f}%"
        )
    print("=" * 95)

    disp_cols = [
        'pred_ridge_rank', 'Driver', 'TeamName', 'grid_pos', 'quali_pos', 'quali_gap_pct',
        'driver_form_3', 'team_form_3', 'team_dnf_rate_5',
        'pred_ridge_cont', 'p_win_lr', 'p_win_hgb', 'p_pod_hgb', 'p_pod_lr'
    ]
    disp = pred_df[disp_cols].copy()
    disp.rename(columns={
        'pred_ridge_rank': 'Rank',
        'grid_pos': 'Grid',
        'quali_pos': 'Quali',
        'quali_gap_pct': 'Q_Gap%',
        'driver_form_3': 'Drv_Form3',
        'team_form_3': 'Team_Form3',
        'team_dnf_rate_5': 'DNF_Rate5',
        'pred_ridge_cont': 'Exp_Pos',
        'p_win_lr': 'P(Win)_LR',
        'p_win_hgb': 'P(Win)_HGB',
        'p_pod_hgb': 'P(Pod)_HGB',
        'p_pod_lr': 'P(Pod)_LR'
    }, inplace=True)

    disp['Rank'] = disp['Rank'].astype(int)
    disp['Grid'] = disp['Grid'].astype(int)
    disp['Quali'] = disp['Quali'].astype(int)
    disp['Q_Gap%'] = disp['Q_Gap%'].map('{:.2f}%'.format)
    disp['Drv_Form3'] = disp['Drv_Form3'].map('{:.2f}'.format)
    disp['Team_Form3'] = disp['Team_Form3'].map('{:.2f}'.format)
    disp['DNF_Rate5'] = (disp['DNF_Rate5'] * 100).map('{:.0f}%'.format)
    disp['Exp_Pos'] = disp['Exp_Pos'].map('{:.2f}'.format)
    disp['P(Win)_LR'] = (disp['P(Win)_LR'] * 100).map('{:.1f}%'.format)
    disp['P(Win)_HGB'] = (disp['P(Win)_HGB'] * 100).map('{:.1f}%'.format)
    disp['P(Pod)_HGB'] = (disp['P(Pod)_HGB'] * 100).map('{:.1f}%'.format)
    disp['P(Pod)_LR'] = (disp['P(Pod)_LR'] * 100).map('{:.1f}%'.format)

    print("\n--- COMPLETE 22-DRIVER BRAZIL 2026 TABLE ---")
    print(disp.to_string(index=False))
    print(f"\nProbability Sums Check -> Sum P(Win)_LR: {pred_df['p_win_lr'].sum():.4f} | "
          f"Sum P(Win)_HGB: {pred_df['p_win_hgb'].sum():.4f} | "
          f"Sum P(Pod)_HGB: {pred_df['p_pod_hgb'].sum():.4f} | "
          f"Sum P(Pod)_LR: {pred_df['p_pod_lr'].sum():.4f}")


if __name__ == '__main__':
    main()

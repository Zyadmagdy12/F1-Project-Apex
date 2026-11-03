"""
src/3_features.py
-----------------
Stage 3: Feature Engineering & Automated Data Leakage Verification.
Computes 11 strictly pre-race features for F1 race result prediction.

Outputs:
  - data/features.csv (132 historical rounds, 2021-2026 pre-Mexico)
  - data/target_mexico_2026_features.csv (Mexico GP 2026 backtest target)
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List

# Configure UTF-8 encoding for console output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

CLEAN_RACES_CSV = os.path.join(DATA_DIR, 'clean_races.csv')
CLEAN_TARGET_CSV = os.path.join(DATA_DIR, 'clean_target_mexico_2026.csv')

FEATURES_CSV = os.path.join(DATA_DIR, 'features.csv')
TARGET_FEATURES_CSV = os.path.join(DATA_DIR, 'target_mexico_2026_features.csv')

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
    'team_dnf_rate_5'
]

ALL_EXPORT_COLUMNS = [
    'Year', 'Round', 'EventName', 'Location', 'EventDate', 'EventFormat',
    'Driver', 'TeamName', 'RawTeamName',
    'grid_pos', 'quali_pos', 'qualified_on_pole', 'pitlane_start', 'no_quali_time',
    'is_sprint_weekend', 'quali_gap_pct', 'teammate_quali_delta',
    'driver_form_3', 'team_form_3', 'team_dnf_rate_5', 'is_new_driver_or_team',
    'FinishPosition', 'Status', 'is_dnf', 'is_anomaly_race', 'FetchedAtUTC'
]


def normalize_race_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Enforces strict unique integer rank normalization for quali_pos and grid_pos in [1, N_starters]."""
    out = df.copy()
    processed = []
    for (yr, rnd), grp in out.groupby(['Year', 'Round'], sort=False):
        g = grp.copy()
        n_starters = len(g)
        # Sort by (quali_pos, best_quali_sec, grid_pos) and assign unique 1..N ranks via method='first'
        q_order = g.sort_values(
            by=['quali_pos', 'best_quali_sec', 'grid_pos', 'Driver'],
            ascending=[True, True, True, True],
            na_position='last'
        ).index
        g.loc[q_order, 'quali_pos'] = np.arange(1, n_starters + 1, dtype=float)

        # Sort by (grid_pos, quali_pos, best_quali_sec) and assign unique 1..N ranks via method='first'
        g_order = g.sort_values(
            by=['grid_pos', 'quali_pos', 'best_quali_sec', 'Driver'],
            ascending=[True, True, True, True],
            na_position='last'
        ).index
        g.loc[g_order, 'grid_pos'] = np.arange(1, n_starters + 1, dtype=float)
        processed.append(g)
    return pd.concat(processed, ignore_index=True)


def compute_features_pipeline(clean_races_df: pd.DataFrame,
                              clean_target_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Engineers all 11 features on unified chronological sequence, then splits out target."""
    stats: Dict[str, Any] = {}

    # 1. Unify chronologically
    target_copy = clean_target_df.copy()
    target_copy['is_target_race'] = 1
    races_copy = clean_races_df.copy()
    races_copy['is_target_race'] = 0

    combined = pd.concat([races_copy, target_copy], ignore_index=True)
    combined['EventDate'] = pd.to_datetime(combined['EventDate'])
    combined.sort_values(by=['EventDate', 'Year', 'Round', 'Driver'], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # 2. Strict Rank Normalization
    combined = normalize_race_ranks(combined)

    # 3. Static & Session-Level Features
    combined['qualified_on_pole'] = (combined['quali_pos'] == 1.0).astype(int)
    combined['pitlane_start'] = combined['pitlane_start'].astype(int)
    combined['no_quali_time'] = combined['no_quali_time'].astype(int)
    combined['is_sprint_weekend'] = combined['EventFormat'].astype(str).str.contains('sprint', case=False, na=False).astype(int)

    # 4. Qualifying Gap Percentage (quali_gap_pct)
    capped_count = 0
    total_quali_count = len(combined)

    def calc_race_quali_gap(race_grp: pd.DataFrame) -> pd.Series:
        nonlocal capped_count
        best_times = race_grp['best_quali_sec']
        valid_times = best_times.dropna()
        if len(valid_times) == 0:
            return pd.Series(7.0, index=race_grp.index)
        pole_time = valid_times.min()
        raw_gap = ((best_times - pole_time) / pole_time) * 100.0
        # Count capped or missing
        capped_mask = (raw_gap > 7.0) | (best_times.isna())
        capped_count += capped_mask.sum()
        # Clip at [0.0, 7.0] and impute NaN with 7.0
        final_gap = raw_gap.clip(lower=0.0, upper=7.0).fillna(7.0)
        return final_gap

    quali_gaps = []
    for (yr, rnd), r_grp in combined.groupby(['Year', 'Round'], sort=False):
        quali_gaps.append(calc_race_quali_gap(r_grp))
    combined['quali_gap_pct'] = pd.concat(quali_gaps).sort_index()

    stats['total_observations'] = total_quali_count
    stats['quali_gap_capped_count'] = int(capped_count)
    stats['quali_gap_capped_pct'] = float((capped_count / total_quali_count) * 100.0)

    # 5. Teammate Qualifying Delta (teammate_quali_delta)
    def calc_teammate_delta(team_grp: pd.DataFrame) -> pd.Series:
        if len(team_grp) <= 1:
            return pd.Series(0.0, index=team_grp.index)
        mean_team_gap = team_grp['quali_gap_pct'].mean()
        return team_grp['quali_gap_pct'] - mean_team_gap

    tm_deltas = []
    for (yr, rnd, tm), tm_grp in combined.groupby(['Year', 'Round', 'TeamName'], sort=False):
        tm_deltas.append(calc_teammate_delta(tm_grp))
    combined['teammate_quali_delta'] = pd.concat(tm_deltas).sort_index()

    # 6. Driver Form (driver_form_3)
    driver_form_series = pd.Series(index=combined.index, dtype=float)
    new_driver_flag = pd.Series(0, index=combined.index, dtype=int)

    for drv, d_grp in combined.groupby('Driver', sort=False):
        d_sorted = d_grp.sort_values(by=['EventDate', 'Year', 'Round'])
        shifted_roll = d_sorted['FinishPosition'].shift(1).rolling(3, min_periods=1).mean()
        for idx, val in shifted_roll.items():
            if pd.isna(val):
                new_driver_flag.loc[idx] = 1
                season_yr = combined.loc[idx, 'Year']
                driver_form_series.loc[idx] = 10.5 if season_yr <= 2025 else 11.5
            else:
                driver_form_series.loc[idx] = val

    combined['driver_form_3'] = driver_form_series
    combined['is_new_driver_or_team'] = new_driver_flag

    # 7. Team Form (team_form_3) - ANTI-LEAKAGE AGGREGATION
    team_race_agg = combined.groupby(['EventDate', 'Year', 'Round', 'TeamName'], as_index=False).agg(
        team_finish_mean=('FinishPosition', 'mean'),
        team_dnf_mean=('is_dnf', 'mean')
    ).sort_values(by=['EventDate', 'Year', 'Round'])

    team_race_agg['team_form_3'] = team_race_agg.groupby('TeamName')['team_finish_mean'].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    team_race_agg['team_dnf_rate_5'] = team_race_agg.groupby('TeamName')['team_dnf_mean'].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )

    merged_team = pd.merge(
        combined[['EventDate', 'Year', 'Round', 'TeamName']],
        team_race_agg[['EventDate', 'Year', 'Round', 'TeamName', 'team_form_3', 'team_dnf_rate_5']],
        on=['EventDate', 'Year', 'Round', 'TeamName'],
        how='left'
    )
    
    combined['team_form_3'] = merged_team['team_form_3'].fillna(11.5)
    combined['team_dnf_rate_5'] = merged_team['team_dnf_rate_5'].fillna(0.14)

    # Format EventDate back to YYYY-MM-DD
    combined['EventDate'] = combined['EventDate'].dt.strftime('%Y-%m-%d')

    # Split back into historical features and target Mexico 2026 features
    clean_features_df = combined[combined['is_target_race'] == 0].drop(columns=['is_target_race'])
    target_features_df = combined[combined['is_target_race'] == 1].drop(columns=['is_target_race'])

    # Arrange columns
    export_cols = [c for c in ALL_EXPORT_COLUMNS if c in clean_features_df.columns]
    clean_features_df = clean_features_df[export_cols].sort_values(by=['Year', 'Round', 'FinishPosition']).reset_index(drop=True)
    target_features_df = target_features_df[export_cols].sort_values(by=['FinishPosition']).reset_index(drop=True)

    return clean_features_df, target_features_df, stats


def verify_no_data_leakage(clean_races_df: pd.DataFrame, clean_target_df: pd.DataFrame) -> bool:
    """Verifies that future race results cannot leak into past or present pre-race features."""
    print("\n-----------------------------------------------------------------")
    print("RUNNING AUTOMATED DATA LEAKAGE VERIFICATION TEST")
    print("-----------------------------------------------------------------")

    # Baseline run
    orig_features, orig_target, _ = compute_features_pipeline(clean_races_df, clean_target_df)

    # Mutated copy 1: Corrupt FinishPosition and is_dnf for 2026 Rounds 10 through 19
    mutated_races = clean_races_df.copy()
    mutated_target = clean_target_df.copy()

    mask_races = (mutated_races['Year'] == 2026) & (mutated_races['Round'] >= 10)
    mutated_races.loc[mask_races, 'FinishPosition'] = 999.0
    mutated_races.loc[mask_races, 'is_dnf'] = 1

    mutated_target['FinishPosition'] = 999.0
    mutated_target['is_dnf'] = 1

    # Re-run pipeline on mutated copy
    mut_features, mut_target, _ = compute_features_pipeline(mutated_races, mutated_target)

    # Normalize sorting by [Year, Round, Driver] so driver identity is strictly aligned
    orig_sorted = orig_features.sort_values(by=['Year', 'Round', 'Driver']).reset_index(drop=True)
    mut_sorted = mut_features.sort_values(by=['Year', 'Round', 'Driver']).reset_index(drop=True)

    # Test 1: All features for 2021-2025 and 2026 Rounds 1-9 MUST BE IDENTICAL
    check1_mask = (orig_sorted['Year'] <= 2025) | ((orig_sorted['Year'] == 2026) & (orig_sorted['Round'] < 10))
    orig_sub1 = orig_sorted.loc[check1_mask, FEATURE_COLUMNS]
    mut_sub1 = mut_sorted.loc[check1_mask, FEATURE_COLUMNS]

    diff_count_1 = 0
    for col in FEATURE_COLUMNS:
        if not np.allclose(orig_sub1[col], mut_sub1[col], atol=1e-9, equal_nan=True):
            diff_count_1 += 1
            print(f"❌ LEAK DETECTED in historical features: column '{col}' differed!")

    test1_pass = (diff_count_1 == 0)
    print(f"Test 1 (Historical Invariance across 2021-2025 & 2026 R1-R9): {'PASSED' if test1_pass else 'FAILED'}")

    # Test 2: Round 10 itself (Pre-race features MUST be identical at the start of Round 10)
    check2_mask = (orig_sorted['Year'] == 2026) & (orig_sorted['Round'] == 10)
    orig_sub2 = orig_sorted.loc[check2_mask, FEATURE_COLUMNS]
    mut_sub2 = mut_sorted.loc[check2_mask, FEATURE_COLUMNS]

    diff_count_2 = 0
    for col in FEATURE_COLUMNS:
        if not np.allclose(orig_sub2[col], mut_sub2[col], atol=1e-9, equal_nan=True):
            diff_count_2 += 1
            print(f"❌ LEAK DETECTED in Round 10 pre-race features: column '{col}' differed!")

    test2_pass = (diff_count_2 == 0)
    print(f"Test 2 (Pre-Race Feature Invariance at 2026 Round 10 Entry): {'PASSED' if test2_pass else 'FAILED'}")

    # Test 3: Mutating Mexico 2026 FinishPosition MUST NOT alter Mexico 2026 pre-race features
    mut_target_only = clean_target_df.copy()
    mut_target_only['FinishPosition'] = 777.0
    mut_target_only['is_dnf'] = 1
    _, target_mut_features, _ = compute_features_pipeline(clean_races_df, mut_target_only)

    orig_target_feat = orig_target.sort_values(by='Driver').reset_index(drop=True)[FEATURE_COLUMNS]
    mut_target_feat = target_mut_features.sort_values(by='Driver').reset_index(drop=True)[FEATURE_COLUMNS]

    diff_count_3 = 0
    for col in FEATURE_COLUMNS:
        if not np.allclose(orig_target_feat[col], mut_target_feat[col], atol=1e-9, equal_nan=True):
            diff_count_3 += 1
            print(f"❌ LEAK DETECTED in Mexico 2026 target features: column '{col}' differed!")

    test3_pass = (diff_count_3 == 0)
    print(f"Test 3 (Target Mexico 2026 Pre-Race Features Zero-Leakage): {'PASSED' if test3_pass else 'FAILED'}")

    all_passed = test1_pass and test2_pass and test3_pass
    print(f"\nOverall Anti-Leakage Verification: {'ALL TESTS PASSED' if all_passed else 'VERIFICATION FAILED'}")
    return all_passed


def main():
    print("=================================================================")
    print("STAGE 3: FEATURE ENGINEERING & DATA LEAKAGE VERIFICATION")
    print("=================================================================")

    # Load cleaned data
    clean_races = pd.read_csv(CLEAN_RACES_CSV)
    clean_target = pd.read_csv(CLEAN_TARGET_CSV)
    print(f"Loaded clean_races.csv: {len(clean_races)} rows")
    print(f"Loaded clean_target_mexico_2026.csv: {len(clean_target)} rows")

    # Run automated leakage verification test first
    leakage_passed = verify_no_data_leakage(clean_races, clean_target)
    assert leakage_passed, "Critical Failure: Data leakage detected in feature engineering pipeline!"

    # Compute production features
    features_df, target_features_df, stats = compute_features_pipeline(clean_races, clean_target)

    # Save outputs
    features_df.to_csv(FEATURES_CSV, index=False)
    target_features_df.to_csv(TARGET_FEATURES_CSV, index=False)
    print(f"\n Saved features to: {FEATURES_CSV} ({len(features_df)} rows, {len(features_df.columns)} columns)")
    print(f" Saved target features to: {TARGET_FEATURES_CSV} ({len(target_features_df)} rows, {len(target_features_df.columns)} columns)")

    # Print Stage 3 QA Report
    print("\n=================================================================")
    print("STAGE 3 QA REPORT")
    print("=================================================================")

    # A) Summary Statistics for the 11 Engineered Features
    print("\n--- A) Summary Statistics for 11 Engineered Features (data/features.csv) ---")
    feat_stats = features_df[FEATURE_COLUMNS].describe().T
    feat_stats['nan_count'] = features_df[FEATURE_COLUMNS].isna().sum()
    print(feat_stats[['count', 'mean', 'std', 'min', '50%', 'max', 'nan_count']].to_string())

    # B) Capping Statistics
    print("\n--- B) Qualifying Gap Capping Statistics ---")
    print(f"Total driver-race observations: {stats['total_observations']}")
    print(f"Observations capped or imputed at 7.0%: {stats['quali_gap_capped_count']} ({stats['quali_gap_capped_pct']:.2f}%)")

    # C) Leakage Status
    print("\n--- C) Data Leakage Verification Status ---")
    print(f"Automated Test Status: {'PASSED (Zero lookahead / zero teammate leakage)' if leakage_passed else 'FAILED'}")

    # D) Target Mexico 2026 Top 10 Starters
    print("\n--- D) Mexico GP 2026: Top 10 Starters Engineered Features ---")
    top10_mexico = target_features_df.sort_values(by='grid_pos').head(10)
    display_cols = ['Driver', 'TeamName', 'grid_pos', 'quali_pos', 'qualified_on_pole',
                    'quali_gap_pct', 'teammate_quali_delta', 'driver_form_3', 'team_form_3', 'team_dnf_rate_5']
    print(top10_mexico[display_cols].to_string(index=False))

    print("\n Stage 3 Feature Engineering completed and verified!")


if __name__ == '__main__':
    main()

"""
src/2_clean_data.py
-------------------
Stage 2 Data Cleaning & Schema Standardization.
Cleans raw F1 race datasets (training corpus and Mexico 2026 target).

Outputs:
  - data/clean_races.csv
  - data/clean_target_mexico_2026.csv
  - data/excluded_dns_rows.csv
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

RAW_RACES_CSV = os.path.join(DATA_DIR, 'raw_all_races_data.csv')
RAW_TARGET_CSV = os.path.join(DATA_DIR, 'target_mexico_2026.csv')

CLEAN_RACES_CSV = os.path.join(DATA_DIR, 'clean_races.csv')
CLEAN_TARGET_CSV = os.path.join(DATA_DIR, 'clean_target_mexico_2026.csv')
EXCLUDED_DNS_CSV = os.path.join(DATA_DIR, 'excluded_dns_rows.csv')

# Team name standardization mapping
TEAM_MAPPING = {
    'Red Bull Racing': 'Red Bull',
    'Red Bull': 'Red Bull',
    'Alpine': 'Alpine',
    'Alpine F1 Team': 'Alpine',
    'AlphaTauri': 'RB',
    'RB F1 Team': 'RB',
    'RB': 'RB',
    'Alfa Romeo Racing': 'Sauber_Audi',
    'Alfa Romeo': 'Sauber_Audi',
    'Sauber': 'Sauber_Audi',
    'Kick Sauber': 'Sauber_Audi',
    'Audi': 'Sauber_Audi',
    'Haas F1 Team': 'Haas',
    'Haas': 'Haas',
    'Cadillac F1 Team': 'Cadillac',
    'Cadillac': 'Cadillac',
    'Aston Martin': 'Aston Martin',
    'Aston Martin Cognizant F1 Team': 'Aston Martin',
    'Aston Martin Aramco': 'Aston Martin',
    'Mercedes': 'Mercedes',
    'Ferrari': 'Ferrari',
    'McLaren': 'McLaren',
    'Williams': 'Williams'
}


def is_finisher_status(status_str: Any) -> bool:
    """Checks whether the official status indicates the car finished the race."""
    if not isinstance(status_str, str):
        return False
    s = status_str.strip()
    if s in ['Finished', 'Lapped']:
        return True
    if re.match(r'^\+\d+\s+Laps?$', s, re.IGNORECASE):
        return True
    return False


def is_dns_row(row: pd.Series) -> bool:
    """Identifies Did Not Start (DNS) or withdrawn entries."""
    if pd.isna(row['FinishPosition']) or pd.isna(row['GridPosition']):
        return True
    status = str(row['Status']).strip().lower()
    if status in ['did not start', 'withdrew', 'illness', 'dns']:
        return True
    if 'did not start' in status or 'withdrew' in status or 'illness' in status:
        return True
    return False


def resolve_quali_ties(group: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
    """Resolves duplicate QualiPositions within a single race deterministically."""
    log_messages = []
    q_pos = group['quali_pos'].copy()
    counts = q_pos.value_counts()
    ties = counts[counts > 1].index.tolist()

    if not ties:
        return q_pos, log_messages

    yr = group['Year'].iloc[0]
    rnd = group['Round'].iloc[0]
    event = group['EventName'].iloc[0]

    for tie_val in ties:
        tied_indices = group[q_pos == tie_val].index
        # Sort tied drivers: best_quali_sec ascending (lowest time first), then grid_pos ascending
        tied_subset = group.loc[tied_indices].sort_values(
            by=['best_quali_sec', 'grid_pos'],
            ascending=[True, True],
            na_position='last'
        )
        
        assigned_positions = list(range(int(tie_val), int(tie_val) + len(tied_subset)))
        for rank_offset, (idx, row) in enumerate(tied_subset.iterrows()):
            new_pos = assigned_positions[rank_offset]
            q_pos.loc[idx] = float(new_pos)
            msg = (f"Resolved tie in {yr} R{rnd} ({event}) for {row['Driver']}: "
                   f"QualiPosition {tie_val} -> {new_pos} (time: {row['best_quali_sec']}s, grid: {row['grid_pos']})")
            log_messages.append(msg)

    return q_pos, log_messages


def clean_dataframe(df_raw: pd.DataFrame, is_target: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Applies all Stage 2 cleaning and schema transformation rules."""
    df = df_raw.copy()
    tie_logs = []

    # 1. Chronological Sorting & Datetime Parsing
    df['EventDate'] = pd.to_datetime(df['EventDate']).dt.strftime('%Y-%m-%d')
    df.sort_values(by=['Year', 'Round', 'FinishPosition'], ascending=[True, True, True], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 2. Team Name Standardization
    df['RawTeamName'] = df['TeamName'].copy()
    df['TeamName'] = df['RawTeamName'].map(lambda x: TEAM_MAPPING.get(str(x).strip(), str(x).strip()))

    # 3. DNS Identification & Exclusion
    dns_mask = df.apply(is_dns_row, axis=1)
    dns_df = df[dns_mask].copy()
    clean_df = df[~dns_mask].copy().reset_index(drop=True)

    # 4. Explicit DNF Flag
    clean_df['is_dnf'] = clean_df['Status'].apply(lambda s: 0 if is_finisher_status(s) else 1)

    # 5. Pitlane Start Handling
    clean_df['pitlane_start'] = (clean_df['GridPosition'] == 0).astype(int)
    clean_df['no_quali_time'] = clean_df['best_quali_sec'].isna().astype(int)

    # 6. Race-by-race adjustment for grid_pos and quali_pos
    race_dfs = []
    for (yr, rnd), race_grp in clean_df.groupby(['Year', 'Round'], sort=False):
        grp = race_grp.copy()
        
        # Grid adjustment
        max_grid = max(grp['GridPosition'].max(), len(grp))
        g = grp['GridPosition'].copy()
        pit_indices = grp[g == 0].index
        for i, p_idx in enumerate(pit_indices):
            g.loc[p_idx] = float(max_grid + i)
        grp['grid_pos'] = g

        # Quali adjustment & tie breaking
        back_pos = float(len(grp))
        grp['quali_pos'] = grp['QualiPosition'].fillna(back_pos)
        resolved_q, msgs = resolve_quali_ties(grp)
        grp['quali_pos'] = resolved_q
        tie_logs.extend(msgs)

        race_dfs.append(grp)

    if race_dfs:
        clean_df = pd.concat(race_dfs, ignore_index=True)
    else:
        clean_df['grid_pos'] = clean_df['GridPosition']
        clean_df['quali_pos'] = clean_df['QualiPosition']

    # 7. Belgian GP 2021 Anomaly Flag
    clean_df['is_anomaly_race'] = ((clean_df['Year'] == 2021) & (clean_df['Round'] == 12)).astype(int)

    # Final Schema Arrangement
    final_cols = [
        'Year', 'Round', 'EventName', 'Location', 'EventDate', 'EventFormat',
        'Driver', 'TeamName', 'RawTeamName',
        'grid_pos', 'GridPosition', 'pitlane_start',
        'quali_pos', 'QualiPosition', 'best_quali_sec', 'no_quali_time',
        'FinishPosition', 'Status', 'is_dnf',
        'is_anomaly_race', 'FetchedAtUTC'
    ]
    
    # Keep only columns that exist
    ordered_cols = [c for c in final_cols if c in clean_df.columns]
    clean_df = clean_df[ordered_cols].sort_values(by=['Year', 'Round', 'FinishPosition']).reset_index(drop=True)

    return clean_df, dns_df, tie_logs


def main():
    print("=================================================================")
    print("STAGE 2: DATA CLEANING & SCHEMA STANDARDIZATION")
    print("=================================================================")

    # 1. Clean Historical Training Corpus
    raw_df = pd.read_csv(RAW_RACES_CSV)
    print(f"Loaded raw races dataset: {len(raw_df)} rows from {RAW_RACES_CSV}")
    
    clean_races_df, dns_races_df, ties_races = clean_dataframe(raw_df, is_target=False)
    clean_races_df.to_csv(CLEAN_RACES_CSV, index=False)
    dns_races_df.to_csv(EXCLUDED_DNS_CSV, index=False)
    print(f" Saved cleaned races: {len(clean_races_df)} rows to {CLEAN_RACES_CSV}")
    print(f" Excluded DNS rows: {len(dns_races_df)} rows saved to {EXCLUDED_DNS_CSV}")

    # 2. Clean Mexico 2026 Target Corpus
    target_raw_df = pd.read_csv(RAW_TARGET_CSV)
    print(f"\nLoaded target Mexico 2026 dataset: {len(target_raw_df)} rows from {RAW_TARGET_CSV}")
    
    clean_target_df, dns_target_df, ties_target = clean_dataframe(target_raw_df, is_target=True)
    clean_target_df.to_csv(CLEAN_TARGET_CSV, index=False)
    print(f" Saved cleaned target Mexico 2026: {len(clean_target_df)} rows to {CLEAN_TARGET_CSV}")
    if len(dns_target_df) > 0:
        print(f" Target Mexico 2026 DNS rows excluded: {len(dns_target_df)}")

    # 3. Print QA Verification Summary
    print("\n=================================================================")
    print("STAGE 2 QA VERIFICATION REPORT")
    print("=================================================================")
    
    # A) DNS summary
    print("\n--- A) DNS Exclusion Summary ---")
    print(f"Total raw rows: {len(raw_df)}")
    print(f"Total cleaned rows: {len(clean_races_df)}")
    print(f"Total DNS excluded: {len(dns_races_df)}")
    if len(dns_races_df) > 0:
        dns_summary = dns_races_df.groupby('Year').agg(
            Count=('Driver', 'count'),
            Drivers=('Driver', lambda s: ', '.join(s)),
            Events=('EventName', lambda s: ', '.join(s))
        )
        print("\nDNS breakdown by season:")
        print(dns_summary.to_string())

    # B) Team counts per season
    print("\n--- B) Unique Standardized Team Counts per Season ---")
    for yr in sorted(clean_races_df['Year'].unique()):
        teams = clean_races_df[clean_races_df['Year'] == yr]['TeamName'].unique()
        print(f"Season {yr}: {len(teams)} unique teams -> {sorted(teams)}")
    
    # Target 2026 team count
    target_teams = clean_target_df['TeamName'].unique()
    print(f"Target Mexico 2026: {len(target_teams)} unique teams -> {sorted(target_teams)}")

    # C) DNF rate per season
    print("\n--- C) DNF Rate (is_dnf mean) per Season ---")
    dnf_stats = clean_races_df.groupby('Year').agg(
        TotalDrivers=('is_dnf', 'count'),
        DNFCount=('is_dnf', 'sum'),
        DNFRate=('is_dnf', 'mean')
    )
    for yr, r in dnf_stats.iterrows():
        print(f"Season {yr}: {r['DNFRate']*100:>5.1f}% DNF ({int(r['DNFCount'])} / {int(r['TotalDrivers'])})")

    # Target Mexico 2026 DNF rate
    target_dnf_rate = clean_target_df['is_dnf'].mean()
    target_dnf_count = clean_target_df['is_dnf'].sum()
    print(f"Target Mexico 2026: {target_dnf_rate*100:>5.1f}% DNF ({int(target_dnf_count)} / {len(clean_target_df)})")

    # D) Null audit
    print("\n--- D) Null Value Audit in Cleaned Datasets ---")
    audit_cols = ['grid_pos', 'quali_pos', 'FinishPosition', 'is_dnf', 'pitlane_start', 'no_quali_time']
    races_nulls = clean_races_df[audit_cols].isna().sum()
    target_nulls = clean_target_df[audit_cols].isna().sum()
    print("Null count in clean_races.csv:")
    for col, val in races_nulls.items():
        print(f"  {col:<16}: {val} missing")
    print("Null count in clean_target_mexico_2026.csv:")
    for col, val in target_nulls.items():
        print(f"  {col:<16}: {val} missing")

    # E) Qualifying ties
    print("\n--- E) Intra-Race Qualifying Position Ties Resolved ---")
    all_ties = ties_races + ties_target
    if all_ties:
        for t in all_ties:
            print("  *", t)
    else:
        print("  None detected.")

    print("\n Stage 2 Cleaning successfully executed and verified!")


if __name__ == '__main__':
    main()

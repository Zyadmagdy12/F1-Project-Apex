"""
src/apex_features.py
--------------------
Project Apex — Phase 2 & Phase 3.1/3.2: Leak-Free Feature Engineering Engine
and Status Taxonomy Classifier.

All features for race (Year, Round) are computed strictly from historical
observations completed before that race starts.
"""

import re
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

MECHANICAL_STATUSES = {
    'Engine', 'Gearbox', 'Power Unit', 'Hydraulics', 'Brakes', 'Suspension',
    'Electrical', 'Turbo', 'Driveshaft', 'Oil leak', 'Mechanical', 'Power loss',
    'Fuel pressure', 'Water pressure', 'Water leak', 'Fuel leak', 'Water pump',
    'Cooling system', 'Fuel pump', 'Vibrations', 'Differential', 'Rear wing',
    'Undertray', 'Wheel nut'
}

INCIDENT_STATUSES = {
    'Collision', 'Accident', 'Collision damage', 'Spun off', 'Puncture',
    'Damage', 'Front wing'
}

LAPPED_REGEX = re.compile(r'^\+\d+\s+Laps?$', re.IGNORECASE)


def classify_race_status(status_str: Any) -> Dict[str, int]:
    """
    Classifies a raw FastF1/Ergast Status string into mutually exclusive and
    scientifically defensible indicators:
      - is_classified_finisher: completed race distance or classified lapped
      - is_dsq: post-race technical/sporting disqualification ('Disqualified')
      - is_dnf: in-race retirement (mechanical, incident, or generic 'Retired')
      - is_attributed_mechanical: confirmed mechanical failure status (2021-2023)
      - is_attributed_incident: confirmed crash/collision/damage status (2021-2023)
      - is_unattributed_retired: generic 'Retired' status (2023-2026 API format)
    """
    s = str(status_str).strip() if pd.notna(status_str) else ''
    if s in ('Finished', 'Lapped') or bool(LAPPED_REGEX.match(s)):
        return {
            'is_classified_finisher': 1,
            'is_dsq': 0,
            'is_dnf': 0,
            'is_attributed_mechanical': 0,
            'is_attributed_incident': 0,
            'is_unattributed_retired': 0,
        }
    if s == 'Disqualified':
        return {
            'is_classified_finisher': 0,
            'is_dsq': 1,
            'is_dnf': 0,
            'is_attributed_mechanical': 0,
            'is_attributed_incident': 0,
            'is_unattributed_retired': 0,
        }
    if s in MECHANICAL_STATUSES:
        return {
            'is_classified_finisher': 0,
            'is_dsq': 0,
            'is_dnf': 1,
            'is_attributed_mechanical': 1,
            'is_attributed_incident': 0,
            'is_unattributed_retired': 0,
        }
    if s in INCIDENT_STATUSES:
        return {
            'is_classified_finisher': 0,
            'is_dsq': 0,
            'is_dnf': 1,
            'is_attributed_mechanical': 0,
            'is_attributed_incident': 1,
            'is_unattributed_retired': 0,
        }
    # Generic 'Retired' or any other in-race retirement status
    return {
        'is_classified_finisher': 0,
        'is_dsq': 0,
        'is_dnf': 1,
        'is_attributed_mechanical': 0,
        'is_attributed_incident': 0,
        'is_unattributed_retired': 1 if s == 'Retired' else 0,
    }


# Core feature lists used by models
APEX_POSITION_FEATURES = [
    'Qualifying_Rank',
    'Starting_Grid_Position',
    'Delta_Q_to_G',
    'Teammate_Qualifying_Delta',
    'Driver_Form',
    'Team_Form',
    'Driver_Track_History',
    'Driver_Track_Missing',
    'Circuit_Overtake_Difficulty_Index',
    'Circuit_Attrition_Index',
]

APEX_DNF_FEATURES = [
    'Starting_Grid_Position',
    'Qualifying_Rank',
    'Constructor_DNF_Rate_Smoothed',
    'Constructor_DNF_EWMA',
    'Constructor_Attributed_Mech_Rate',
    'Driver_DNF_Rate_Smoothed',
    'Driver_DNF_EWMA',
    'Driver_Attributed_Incident_Rate',
    'Circuit_Attrition_Index',
    'Circuit_Overtake_Difficulty_Index',
]


def engineer_apex_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the complete Project Apex feature set in strict chronological order.
    Guarantees zero future leakage: for every race (Year, Round), all historical
    statistics use only races strictly prior to (Year, Round).
    """
    df = df_input.copy()
    # Sort strictly by Year, Round, Starting Grid, Driver for deterministic execution
    df = df.sort_values(['Year', 'Round', 'grid_pos', 'Driver']).reset_index(drop=True)

    # 1. Apply status taxonomy classification
    status_dicts = [classify_race_status(s) for s in df['Status']]
    status_df = pd.DataFrame(status_dicts, index=df.index)
    for col in status_df.columns:
        df[col] = status_df[col]
    # Keep legacy is_dnf (which included DSQ) as is_dnf_legacy_incl_dsq for audit transparency
    df['is_dnf_legacy_incl_dsq'] = (df['is_dnf'] | df['is_dsq']).astype(int)

    # 2. Base intra-race pre-start features (known after Saturday Qualifying & FIA final grid)
    df['Qualifying_Rank'] = df['quali_pos'].astype(float)
    df['Starting_Grid_Position'] = df['grid_pos'].astype(float)
    df['Delta_Q_to_G'] = df['Starting_Grid_Position'] - df['Qualifying_Rank']

    # Field size per race
    race_sizes = df.groupby(['Year', 'Round'])['Driver'].transform('count').astype(float)
    df['Field_Size'] = race_sizes

    # Session-level qualifying gap and Teammate_Qualifying_Delta (Phase 2.1)
    quali_gap_pct_list = np.zeros(len(df), dtype=float)
    tm_delta_list = np.full(len(df), np.nan, dtype=float)
    tm_missing_flag = np.zeros(len(df), dtype=int)

    for (yr, rnd), grp in df.groupby(['Year', 'Round'], sort=False):
        idx = grp.index
        q_sec = pd.to_numeric(grp['best_quali_sec'], errors='coerce')
        valid_q = q_sec.dropna()
        if len(valid_q) > 0:
            min_q = float(valid_q.min())
            med_q = float(valid_q.median())
            gap = ((q_sec - min_q) / min_q) * 100.0
            max_gap = float(gap.dropna().max()) if len(gap.dropna()) > 0 else 3.5
            gap = gap.fillna(max(max_gap + 0.5, 3.5)).clip(0.0, 7.0)
        else:
            med_q = 85.0
            gap = pd.Series(0.0, index=idx)
        quali_gap_pct_list[idx] = gap.values

        # Teammate Qualifying Delta within each constructor in this race
        n_field = float(len(grp))
        for team, tgrp in grp.groupby('TeamName', sort=False):
            t_idx = tgrp.index
            if len(tgrp) < 2:
                # Solo driver entry (teammate DNS / withdrawn): explicit missing teammate
                tm_missing_flag[t_idx] = 1
                tm_delta_list[t_idx] = np.nan
            else:
                t_sec = pd.to_numeric(tgrp['best_quali_sec'], errors='coerce')
                t_qpos = tgrp['Quali_Rank_Internal'] if 'Quali_Rank_Internal' in tgrp else tgrp['quali_pos']
                # For each driver, compare against mean of teammate(s)
                for i_pos, row_i in tgrp.iterrows():
                    others = tgrp.drop(index=i_pos)
                    i_sec = pd.to_numeric(row_i.get('best_quali_sec', np.nan), errors='coerce')
                    o_sec = pd.to_numeric(others['best_quali_sec'], errors='coerce').dropna()
                    if pd.notna(i_sec) and len(o_sec) > 0 and med_q > 0:
                        # Percentage time difference normalized by session median lap time
                        delta_val = ((float(i_sec) - float(o_sec.mean())) / med_q) * 100.0
                        tm_delta_list[i_pos] = float(np.clip(delta_val, -5.0, 5.0))
                        tm_missing_flag[i_pos] = 0
                    else:
                        # Documented fallback when lap time is missing: normalized qualifying rank delta
                        o_qpos = others['quali_pos'].mean()
                        delta_rk = ((float(row_i['quali_pos']) - float(o_qpos)) / n_field) * 5.0
                        tm_delta_list[i_pos] = float(np.clip(delta_rk, -5.0, 5.0))
                        tm_missing_flag[i_pos] = 0

    df['quali_gap_pct'] = quali_gap_pct_list
    df['Teammate_Qualifying_Delta'] = tm_delta_list
    df['Teammate_Missing_Flag'] = tm_missing_flag
    # Legacy alias with 0.0 fill for backward compatibility
    df['teammate_quali_delta'] = pd.Series(tm_delta_list, index=df.index).fillna(0.0)

    # 3. Strictly Historical Sequential Loop across chronological (Year, Round)
    unique_races = df[['Year', 'Round', 'Location']].drop_duplicates().sort_values(['Year', 'Round']).values

    # State trackers (updated ONLY after a race's pre-race features are emitted)
    driver_hist_finishes: Dict[str, List[float]] = {}          # Raw FinishPosition
    driver_hist_norm20: Dict[str, List[float]] = {}            # 20-car scaled finish position
    driver_hist_dnfs: Dict[str, List[int]] = {}                # On-track DNF (0/1)
    driver_hist_incidents: Dict[str, List[int]] = {}           # Attributed incident DNF (0/1)
    driver_circuit_norm01: Dict[Tuple[str, str], List[float]] = {}  # (Driver, Location) -> [0..1] finish

    team_hist_finishes: Dict[str, List[float]] = {}            # Per-race mean finish for constructor
    team_hist_norm20: Dict[str, List[float]] = {}              # Per-race mean 20-car scaled finish
    team_hist_dnfs: Dict[str, List[int]] = {}                  # Per-car-start DNF (0/1)
    team_hist_mechs: Dict[str, List[int]] = {}                 # Per-car-start attributed mechanical DNF

    circuit_hist_difficulty: Dict[str, List[float]] = {}       # Location -> list of race difficulty D_r
    circuit_hist_attrition: Dict[str, List[float]] = {}        # Location -> list of race DNF rate A_r

    global_difficulty_history: List[float] = []
    global_attrition_history: List[float] = []
    global_dnf_starts: List[int] = []
    global_mech_starts: List[int] = []
    global_inc_starts: List[int] = []

    # Output arrays
    n_rows = len(df)
    driver_form_3_arr = np.zeros(n_rows, dtype=float)
    team_form_3_arr = np.zeros(n_rows, dtype=float)
    driver_form_ewma_arr = np.zeros(n_rows, dtype=float)
    team_form_ewma_arr = np.zeros(n_rows, dtype=float)
    team_dnf_rate_5_arr = np.zeros(n_rows, dtype=float)
    is_new_arr = np.zeros(n_rows, dtype=int)

    circuit_overtake_arr = np.zeros(n_rows, dtype=float)
    circuit_attrition_arr = np.zeros(n_rows, dtype=float)
    driver_track_hist_arr = np.zeros(n_rows, dtype=float)
    driver_track_starts_arr = np.zeros(n_rows, dtype=int)
    driver_track_missing_arr = np.zeros(n_rows, dtype=int)

    c_dnf_smooth_arr = np.zeros(n_rows, dtype=float)
    c_dnf_ewma_arr = np.zeros(n_rows, dtype=float)
    c_mech_rate_arr = np.zeros(n_rows, dtype=float)
    d_dnf_smooth_arr = np.zeros(n_rows, dtype=float)
    d_dnf_ewma_arr = np.zeros(n_rows, dtype=float)
    d_inc_rate_arr = np.zeros(n_rows, dtype=float)

    for yr, rnd, loc in unique_races:
        mask = (df['Year'] == yr) & (df['Round'] == rnd)
        grp = df.loc[mask]
        n_race = float(len(grp))

        # Historical global priors strictly before this race
        prior_diff = float(np.mean(global_difficulty_history)) if global_difficulty_history else 0.58
        prior_attr = float(np.mean(global_attrition_history)) if global_attrition_history else 0.14
        prior_dnf = float(np.mean(global_dnf_starts)) if global_dnf_starts else 0.14
        prior_mech = float(np.mean(global_mech_starts)) if global_mech_starts else 0.04
        prior_inc = float(np.mean(global_inc_starts)) if global_inc_starts else 0.04

        # Circuit-level features strictly before this race (Phases 2.2 & 2.3)
        c_diffs = circuit_hist_difficulty.get(loc, [])
        c_attrs = circuit_hist_attrition.get(loc, [])
        m_circuit_prior = 2.0
        c_overtake_val = (sum(c_diffs) + m_circuit_prior * prior_diff) / (len(c_diffs) + m_circuit_prior)
        c_attrition_val = (sum(c_attrs) + m_circuit_prior * prior_attr) / (len(c_attrs) + m_circuit_prior)

        # Emit pre-race features for every starter in this race
        for idx, row in grp.iterrows():
            drv = str(row['Driver'])
            tm = str(row['TeamName'])

            circuit_overtake_arr[idx] = c_overtake_val
            circuit_attrition_arr[idx] = c_attrition_val

            # Driver form (legacy 3-race mean & recency-weighted 20-car normalized)
            d_fins = driver_hist_finishes.get(drv, [])
            d_n20 = driver_hist_norm20.get(drv, [])
            if len(d_fins) == 0:
                driver_form_3_arr[idx] = 10.5
                driver_form_ewma_arr[idx] = 10.5
                is_new_drv = 1
            else:
                driver_form_3_arr[idx] = float(np.mean(d_fins[-3:]))
                # Exponential recency weights (decay = 0.6 over up to last 6 starts, shrunk slightly to 10.5 if <3 starts)
                recent_n20 = d_n20[-6:]
                w_d = np.array([0.6 ** (len(recent_n20) - 1 - k) for k in range(len(recent_n20))])
                ewma_d = float(np.sum(w_d * recent_n20) / np.sum(w_d))
                if len(d_fins) < 3:
                    ewma_d = (len(d_fins) * ewma_d + (3 - len(d_fins)) * 10.5) / 3.0
                driver_form_ewma_arr[idx] = ewma_d
                is_new_drv = 0

            # Team form (legacy 3-race mean & recency-weighted 20-car normalized)
            t_fins = team_hist_finishes.get(tm, [])
            t_n20 = team_hist_norm20.get(tm, [])
            if len(t_fins) == 0:
                team_form_3_arr[idx] = 11.5
                team_form_ewma_arr[idx] = 11.5
                is_new_tm = 1
            else:
                team_form_3_arr[idx] = float(np.mean(t_fins[-3:]))
                recent_tn20 = t_n20[-6:]
                w_t = np.array([0.6 ** (len(recent_tn20) - 1 - k) for k in range(len(recent_tn20))])
                ewma_t = float(np.sum(w_t * recent_tn20) / np.sum(w_t))
                if len(t_fins) < 3:
                    ewma_t = (len(t_fins) * ewma_t + (3 - len(t_fins)) * 11.5) / 3.0
                team_form_ewma_arr[idx] = ewma_t
                is_new_tm = 0

            is_new_arr[idx] = 1 if (is_new_drv or is_new_tm) else 0

            # Driver Track History (Phase 2.4)
            dt_list = driver_circuit_norm01.get((drv, loc), [])
            driver_track_starts_arr[idx] = len(dt_list)
            if len(dt_list) == 0:
                driver_track_missing_arr[idx] = 1
                driver_track_hist_arr[idx] = 0.50  # Neutral field-normalized midpoint
            else:
                driver_track_missing_arr[idx] = 0
                w_dt = np.array([0.65 ** (len(dt_list) - 1 - k) for k in range(len(dt_list))])
                # Empirical Bayes shrinkage with m=1.0 neutral start at 0.50
                driver_track_hist_arr[idx] = float((np.sum(w_dt * dt_list) + 1.0 * 0.50) / (np.sum(w_dt) + 1.0))

            # Constructor DNF Hazard Features (Phase 3.2)
            t_dnfs = team_hist_dnfs.get(tm, [])
            t_mechs = team_hist_mechs.get(tm, [])
            if len(t_dnfs) == 0:
                team_dnf_rate_5_arr[idx] = prior_dnf
                c_dnf_smooth_arr[idx] = prior_dnf
                c_dnf_ewma_arr[idx] = prior_dnf
                c_mech_rate_arr[idx] = prior_mech
            else:
                last10_starts = t_dnfs[-10:]  # 5 races * 2 cars = 10 starts
                team_dnf_rate_5_arr[idx] = float(np.mean(last10_starts))
                last20_starts = t_dnfs[-20:]  # 10 races * 2 cars = 20 starts
                c_dnf_smooth_arr[idx] = float((sum(last20_starts) + 4.0 * prior_dnf) / (len(last20_starts) + 4.0))
                w_td = np.array([0.85 ** (len(last20_starts) - 1 - k) for k in range(len(last20_starts))])
                c_dnf_ewma_arr[idx] = float((np.sum(w_td * last20_starts) + 2.0 * prior_dnf) / (np.sum(w_td) + 2.0))
                c_mech_rate_arr[idx] = float((sum(t_mechs) + 6.0 * prior_mech) / (len(t_mechs) + 6.0))

            # Driver DNF / Incident Hazard Features (Phase 3.2)
            d_dnfs = driver_hist_dnfs.get(drv, [])
            d_incs = driver_hist_incidents.get(drv, [])
            if len(d_dnfs) == 0:
                d_dnf_smooth_arr[idx] = prior_dnf
                d_dnf_ewma_arr[idx] = prior_dnf
                d_inc_rate_arr[idx] = prior_inc
            else:
                last10_d = d_dnfs[-10:]
                d_dnf_smooth_arr[idx] = float((sum(last10_d) + 4.0 * prior_dnf) / (len(last10_d) + 4.0))
                w_dd = np.array([0.80 ** (len(last10_d) - 1 - k) for k in range(len(last10_d))])
                d_dnf_ewma_arr[idx] = float((np.sum(w_dd * last10_d) + 2.0 * prior_dnf) / (np.sum(w_dd) + 2.0))
                d_inc_rate_arr[idx] = float((sum(d_incs) + 6.0 * prior_inc) / (len(d_incs) + 6.0))

        # --- UPDATE HISTORICAL STATE ONLY IF THIS RACE HAS COMPLETED OUTCOMES ---
        has_outcomes = grp['FinishPosition'].notna().all()
        if has_outcomes:
            # Update circuit overtake difficulty (excluding DNF rows per Phase 2.2)
            fin_grp = grp[grp['is_dnf'] == 0]
            if len(fin_grp) > 0 and n_race > 1:
                m_r = float(np.mean(np.abs(fin_grp['FinishPosition'] - fin_grp['grid_pos']) / (n_race - 1.0)))
                d_r = float(1.0 - min(1.0, 2.5 * m_r))
            else:
                d_r = prior_diff
            circuit_hist_difficulty.setdefault(loc, []).append(d_r)
            global_difficulty_history.append(d_r)

            # Update circuit attrition index (Phase 2.3)
            a_r = float(grp['is_dnf'].mean())
            circuit_hist_attrition.setdefault(loc, []).append(a_r)
            global_attrition_history.append(a_r)

            # Update driver histories
            for idx, row in grp.iterrows():
                drv = str(row['Driver'])
                fin_pos = float(row['FinishPosition'])
                norm20 = 1.0 + (fin_pos - 1.0) * (19.0 / max(1.0, n_race - 1.0))
                norm01 = (fin_pos - 1.0) / max(1.0, n_race - 1.0)
                dnf_val = int(row['is_dnf'])
                inc_val = int(row['is_attributed_incident'])

                driver_hist_finishes.setdefault(drv, []).append(fin_pos)
                driver_hist_norm20.setdefault(drv, []).append(norm20)
                driver_hist_dnfs.setdefault(drv, []).append(dnf_val)
                driver_hist_incidents.setdefault(drv, []).append(inc_val)
                driver_circuit_norm01.setdefault((drv, loc), []).append(norm01)

                global_dnf_starts.append(dnf_val)
                global_inc_starts.append(inc_val)

            # Update constructor histories
            for tm, tgrp in grp.groupby('TeamName', sort=False):
                tm_str = str(tm)
                mean_fin = float(tgrp['FinishPosition'].mean())
                mean_n20 = 1.0 + (mean_fin - 1.0) * (19.0 / max(1.0, n_race - 1.0))
                team_hist_finishes.setdefault(tm_str, []).append(mean_fin)
                team_hist_norm20.setdefault(tm_str, []).append(mean_n20)
                for _, trow in tgrp.iterrows():
                    dnf_v = int(trow['is_dnf'])
                    mech_v = int(trow['is_attributed_mechanical'])
                    team_hist_dnfs.setdefault(tm_str, []).append(dnf_v)
                    team_hist_mechs.setdefault(tm_str, []).append(mech_v)
                    global_mech_starts.append(mech_v)

    # Assign computed arrays to DataFrame
    df['driver_form_3'] = driver_form_3_arr
    df['team_form_3'] = team_form_3_arr
    df['Driver_Form'] = driver_form_ewma_arr
    df['Team_Form'] = team_form_ewma_arr
    df['team_dnf_rate_5'] = team_dnf_rate_5_arr
    df['is_new_driver_or_team'] = is_new_arr

    df['Circuit_Overtake_Difficulty_Index'] = circuit_overtake_arr
    df['Circuit_Attrition_Index'] = circuit_attrition_arr
    df['Driver_Track_History'] = driver_track_hist_arr
    df['Driver_Track_Starts'] = driver_track_starts_arr
    df['Driver_Track_Missing'] = driver_track_missing_arr

    df['Constructor_DNF_Rate_Smoothed'] = c_dnf_smooth_arr
    df['Constructor_DNF_EWMA'] = c_dnf_ewma_arr
    df['Constructor_Attributed_Mech_Rate'] = c_mech_rate_arr
    df['Driver_DNF_Rate_Smoothed'] = d_dnf_smooth_arr
    df['Driver_DNF_EWMA'] = d_dnf_ewma_arr
    df['Driver_Attributed_Incident_Rate'] = d_inc_rate_arr

    # Ensure binary flags exist
    df['qualified_on_pole'] = (df['Qualifying_Rank'] == 1.0).astype(int)
    if 'is_sprint_weekend' not in df.columns:
        df['is_sprint_weekend'] = df['EventFormat'].astype(str).str.lower().str.contains('sprint').astype(int)

    return df


def get_feature_dictionary() -> List[Dict[str, Any]]:
    """Returns formal feature metadata for outputs/project_apex/feature_dictionary.json."""
    return [
        {
            'feature_name': 'Qualifying_Rank',
            'legacy_alias': 'quali_pos',
            'formula': 'Official qualifying classification rank normalized to integer [1..N_entrants]',
            'source_columns': ['QualiPosition', 'best_quali_sec'],
            'calculation_window': 'Current event Saturday Qualifying session',
            'historical_cutoff': 'Strictly after Saturday Qualifying and before Sunday Race start',
            'missing_value_strategy': 'Drivers failing to set a Q1 time are ranked at the back of Qualifying in starting grid order with no_quali_time=1',
            'expected_direction': '+1 (Higher numerical rank = worse qualifying position = worse expected finish / lower win & podium probability; monotonic constraint = -1 for Win/Podium)',
            'availability_timing': 'After Qualifying'
        },
        {
            'feature_name': 'Starting_Grid_Position',
            'legacy_alias': 'grid_pos',
            'formula': 'Official FIA final starting grid slot normalized to integer [1..N_entrants]',
            'source_columns': ['GridPosition', 'quali_pos'],
            'calculation_window': 'Current event Sunday pre-race final starting grid',
            'historical_cutoff': 'Strictly before Sunday Race lights out (includes all post-Qualifying power unit / gearbox / sporting grid penalties)',
            'missing_value_strategy': 'Pitlane starts placed at back of grid with pitlane_start=1',
            'expected_direction': '+1 (Higher numerical slot = worse starting position = worse expected finish / lower win & podium probability; monotonic constraint = -1 for Win/Podium)',
            'availability_timing': 'After Final Grid Published'
        },
        {
            'feature_name': 'Delta_Q_to_G',
            'legacy_alias': None,
            'formula': 'Starting_Grid_Position - Qualifying_Rank',
            'source_columns': ['grid_pos', 'quali_pos'],
            'calculation_window': 'Current event pre-race grid vs. qualifying classification',
            'historical_cutoff': 'Strictly before Sunday Race start',
            'missing_value_strategy': 'Deterministic difference of complete integer ranks (0 missing values)',
            'expected_direction': 'Positive values indicate net grid drop (penalty) where car pace is faster than starting slot; negative values indicate net grid promotion due to others penalties',
            'availability_timing': 'After Final Grid Published'
        },
        {
            'feature_name': 'Teammate_Qualifying_Delta',
            'legacy_alias': 'teammate_quali_delta',
            'formula': '((Driver_best_quali_sec - Teammate_best_quali_sec) / Session_Median_quali_sec) * 100.0; fallback to ((quali_pos - teammate_quali_pos) / Field_Size) * 5.0 when lap time unavailable',
            'source_columns': ['best_quali_sec', 'quali_pos', 'TeamName'],
            'calculation_window': 'Current event Qualifying session intra-constructor comparison',
            'historical_cutoff': 'Strictly after Qualifying and before Sunday Race start',
            'missing_value_strategy': 'If teammate did not start (solo constructor entry), Teammate_Qualifying_Delta is left as NaN and Teammate_Missing_Flag=1 (imputed via training-fold median inside model pipeline, never silently assumed 0 in raw table)',
            'expected_direction': '+1 (Positive = slower than teammate in qualifying; Negative = faster than teammate)',
            'availability_timing': 'After Qualifying'
        },
        {
            'feature_name': 'Driver_Form',
            'legacy_alias': 'driver_form_3',
            'formula': 'Exponentially recency-weighted (decay=0.60 over up to 6 prior starts) field-size normalized finishing position (scaled to 20-car reference field), shrunk toward 10.5 for <3 starts',
            'source_columns': ['FinishPosition', 'Year', 'Round', 'Driver'],
            'calculation_window': 'Up to last 6 strictly prior races for the driver',
            'historical_cutoff': 'Strictly prior completed races (Year, Round) < target race',
            'missing_value_strategy': 'Cold-start drivers with 0 prior starts receive neutral midpoint 10.5 and is_new_driver_or_team=1',
            'expected_direction': '+1 (Lower value = better recent finishing form; e.g. 1.67 indicates podium/win form)',
            'availability_timing': 'Before Qualifying (Pre-Weekend)'
        },
        {
            'feature_name': 'Team_Form',
            'legacy_alias': 'team_form_3',
            'formula': 'Exponentially recency-weighted (decay=0.60 over up to 6 prior races) constructor mean finishing position (scaled to 20-car reference field), shrunk toward 11.5 for <3 races',
            'source_columns': ['FinishPosition', 'Year', 'Round', 'TeamName'],
            'calculation_window': 'Up to last 6 strictly prior races for the constructor',
            'historical_cutoff': 'Strictly prior completed races (Year, Round) < target race',
            'missing_value_strategy': 'New constructors with 0 prior races receive 11.5 and is_new_driver_or_team=1',
            'expected_direction': '+1 (Lower value = stronger constructor race pace)',
            'availability_timing': 'Before Qualifying (Pre-Weekend)'
        },
        {
            'feature_name': 'Driver_Track_History',
            'legacy_alias': None,
            'formula': 'Empirical-Bayes smoothed (prior m=1.0 at 0.50) exponentially recency-weighted (decay=0.65) field-normalized finishing position (FinishPosition - 1)/(Field_Size - 1) at the specific circuit Location',
            'source_columns': ['FinishPosition', 'Location', 'Driver'],
            'calculation_window': 'All strictly prior appearances by the driver at that circuit Location',
            'historical_cutoff': 'Strictly prior completed races at the circuit before target race',
            'missing_value_strategy': 'Drivers with 0 prior circuit appearances receive neutral prior 0.50 with Driver_Track_Starts=0 and Driver_Track_Missing=1',
            'expected_direction': '+1 (0.0 = historical P1 dominance, 0.5 = midfield/no-history prior, 1.0 = backmarker history)',
            'availability_timing': 'Before Qualifying (Pre-Weekend)'
        },
        {
            'feature_name': 'Circuit_Overtake_Difficulty_Index',
            'legacy_alias': None,
            'formula': 'Empirical-Bayes smoothed (prior m=2.0 races at global mean) of D_r = 1 - min(1, 2.5 * mean(|Finish - Grid| / (Field_Size - 1))) across classified non-DNF finishers at the circuit',
            'source_columns': ['FinishPosition', 'grid_pos', 'is_dnf', 'Location'],
            'calculation_window': 'All strictly prior races completed at the circuit Location',
            'historical_cutoff': 'Strictly prior completed races before target race (excludes target race outcome and excludes DNF rows)',
            'missing_value_strategy': 'First-time circuits receive historical global mean overtake difficulty (~0.58)',
            'expected_direction': 'Higher values in [0, 1] indicate less positional movement / greater overtaking difficulty',
            'availability_timing': 'Before Qualifying (Pre-Weekend)'
        },
        {
            'feature_name': 'Circuit_Attrition_Index',
            'legacy_alias': None,
            'formula': 'Empirical-Bayes smoothed (prior m=2.0 races at global DNF rate) historical in-race retirement frequency at the circuit Location',
            'source_columns': ['is_dnf', 'Location'],
            'calculation_window': 'All strictly prior races completed at the circuit Location',
            'historical_cutoff': 'Strictly prior completed races before target race',
            'missing_value_strategy': 'First-time circuits receive historical global DNF rate (~0.14)',
            'expected_direction': '+1 (Higher values indicate higher historical circuit retirement risk)',
            'availability_timing': 'Before Qualifying (Pre-Weekend)'
        },
        {
            'feature_name': 'Constructor_DNF_Rate_Smoothed',
            'legacy_alias': 'team_dnf_rate_5',
            'formula': 'Empirical-Bayes smoothed (prior m=4 starts at global DNF rate) constructor retirement rate over last 20 car-starts (10 races), paired with Constructor_DNF_EWMA and Constructor_Attributed_Mech_Rate',
            'source_columns': ['Status', 'is_dnf', 'is_attributed_mechanical', 'TeamName'],
            'calculation_window': 'Up to last 20 strictly prior car-starts for the constructor',
            'historical_cutoff': 'Strictly prior completed races before target race',
            'missing_value_strategy': 'New constructors receive historical global DNF prior',
            'expected_direction': '+1 (Higher values indicate higher constructor retirement hazard)',
            'availability_timing': 'Before Qualifying (Pre-Weekend)'
        }
    ]

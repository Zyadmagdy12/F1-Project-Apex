"""
1_fetch_data.py
---------------
Robust, resumable, and rate-limit aware F1 data ingestion pipeline.
Collects official Qualifying (Q) and Main Race (R) results across 2021–2026.

Outputs:
  - data/raw_all_races_data.csv (atomic incremental update)
  - data/target_mexico_2026.csv (isolated target race)
  - data/fetch_metadata.json (runtime tracking)
  - failed_rounds.txt (true non-rate-limit failures only)
  - pending_rounds.txt (rounds queued when rate limit is encountered)
"""

import os
import sys
import time
import json
import argparse
import datetime
from typing import Dict, List, Any, Optional, Tuple, Set

import numpy as np
import pandas as pd
import fastf1

# Configure UTF-8 encoding for console output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Directory layout
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'f1_cache')
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_CSV = os.path.join(DATA_DIR, 'raw_all_races_data.csv')
TEMP_CSV = os.path.join(DATA_DIR, 'raw_all_races_data.tmp.csv')
TARGET_MEXICO_CSV = os.path.join(DATA_DIR, 'target_mexico_2026.csv')
FAILED_ROUNDS_FILE = os.path.join(BASE_DIR, 'failed_rounds.txt')
PENDING_ROUNDS_FILE = os.path.join(BASE_DIR, 'pending_rounds.txt')
METADATA_FILE = os.path.join(DATA_DIR, 'fetch_metadata.json')

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Configure FastF1 persistent cache
try:
    fastf1.Cache.configure(cache_dir=CACHE_DIR)
except Exception:
    try:
        fastf1.Cache.enable_cache(CACHE_DIR)
    except Exception as e:
        print(f"Warning: could not configure cache: {e}")

REQUIRED_COLUMNS = [
    'Year',
    'Round',
    'EventName',
    'Location',
    'EventDate',
    'EventFormat',
    'Driver',
    'TeamName',
    'GridPosition',
    'QualiPosition',
    'best_quali_sec',
    'FinishPosition',
    'Status',
    'FetchedAtUTC'
]


def is_rate_limit(exc: BaseException) -> bool:
    """Recursively checks exception and cause/context for RateLimitExceeded markers."""
    curr: Optional[BaseException] = exc
    visited = set()
    while curr and id(curr) not in visited:
        visited.add(id(curr))
        msg = str(curr).lower()
        cls_name = type(curr).__name__.lower()
        if 'ratelimit' in msg or 'ratelimit' in cls_name or '429' in msg or '500 calls/h' in msg:
            return True
        curr = curr.__cause__ or curr.__context__
    return False


def load_session_with_retry(year: int, round_num: int, session_type: str, max_retries: int = 2) -> fastf1.core.Session:
    """Loads a session with retry logic for transient network failures."""
    delays = [10, 30]
    for attempt in range(max_retries + 1):
        try:
            session = fastf1.get_session(year, round_num, session_type)
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            return session
        except Exception as e:
            if is_rate_limit(e):
                raise
            if attempt < max_retries:
                print(f"      Transient error loading {session_type} ({type(e).__name__}). Retrying in {delays[attempt]}s...")
                time.sleep(delays[attempt])
            else:
                raise


def process_round_data(year: int, round_num: int, event_name: str, location: str,
                       event_date: str, event_format: str) -> pd.DataFrame:
    """Loads Q and R sessions, calculates best_quali_sec, and merges into the 14-column schema."""
    # 1. Load Qualifying
    try:
        quali = load_session_with_retry(year, round_num, 'Q')
    except Exception as e:
        raise RuntimeError(f"Qualifying session load failed: {type(e).__name__}: {e}") from e

    if quali.results is None or len(quali.results) == 0:
        raise ValueError(f"Qualifying results table is empty for {year} Round {round_num}")

    # Extract Qualifying features
    q_df = quali.results.copy()
    if 'Abbreviation' not in q_df.columns:
        raise ValueError("Column 'Abbreviation' missing from Qualifying results")

    q_sub = pd.DataFrame()
    q_sub['Driver'] = q_df['Abbreviation'].astype(str)
    q_sub['QualiPosition'] = pd.to_numeric(q_df['Position'], errors='coerce')

    # Compute best_quali_sec
    time_cols = [c for c in ['Q1', 'Q2', 'Q3'] if c in q_df.columns]
    if time_cols:
        sec_df = pd.DataFrame(index=q_df.index)
        for col in time_cols:
            td_col = pd.to_timedelta(q_df[col], errors='coerce')
            sec_df[col] = td_col.dt.total_seconds()
        q_sub['best_quali_sec'] = sec_df.min(axis=1)
    else:
        q_sub['best_quali_sec'] = np.nan

    # 2. Load Race
    try:
        race = load_session_with_retry(year, round_num, 'R')
    except Exception as e:
        raise RuntimeError(f"Race session load failed: {type(e).__name__}: {e}") from e

    if race.results is None or len(race.results) == 0:
        raise ValueError(f"Race results table is empty for {year} Round {round_num}")

    r_df = race.results.copy()
    r_sub = pd.DataFrame()
    r_sub['Driver'] = r_df['Abbreviation'].astype(str)
    r_sub['TeamName'] = r_df['TeamName'].astype(str)
    r_sub['GridPosition'] = pd.to_numeric(r_df['GridPosition'], errors='coerce')
    r_sub['FinishPosition'] = pd.to_numeric(r_df['Position'], errors='coerce')
    r_sub['Status'] = r_df['Status'].astype(str)

    # 3. Merge: Race primary, Quali left-joined
    merged = pd.merge(r_sub, q_sub, on='Driver', how='left')
    merged['Year'] = int(year)
    merged['Round'] = int(round_num)
    merged['EventName'] = str(event_name)
    merged['Location'] = str(location)
    merged['EventDate'] = str(event_date)
    merged['EventFormat'] = str(event_format)
    merged['FetchedAtUTC'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    # Return exactly ordered 14 columns
    return merged[REQUIRED_COLUMNS]


def fetch_target_mexico_2026() -> pd.DataFrame:
    """Fetches the 2026 Mexico City GP (Round 19) and saves to data/target_mexico_2026.csv."""
    print("=================================================================")
    print("FETCHING TARGET: Mexico City Grand Prix 2026 (Round 19)")
    print("=================================================================")
    
    # Get official 2026 event schedule info for Round 19
    sched = fastf1.get_event_schedule(2026, include_testing=False)
    r19 = sched[sched['RoundNumber'] == 19]
    if len(r19) == 0:
        event_name = "Mexico City Grand Prix"
        location = "Mexico City"
        event_date = "2026-11-01"
        event_format = "conventional"
    else:
        row = r19.iloc[0]
        event_name = str(row['EventName'])
        location = str(row['Location'])
        event_date = pd.to_datetime(row['EventDate']).strftime('%Y-%m-%d')
        event_format = str(row['EventFormat'])

    df = process_round_data(2026, 19, event_name, location, event_date, event_format)
    df.to_csv(TARGET_MEXICO_CSV, index=False)
    print(f" Successfully saved {len(df)} records to {TARGET_MEXICO_CSV}")
    print("\nTop 3 Finishers in Mexico City GP 2026:")
    top3 = df.sort_values('FinishPosition').head(3)
    for _, r in top3.iterrows():
        print(f"  P{int(r['FinishPosition'])}: {r['Driver']} ({r['TeamName']}) - Started Grid: {r['GridPosition']}")
    return df


def get_prioritized_rounds() -> List[Dict[str, Any]]:
    """Builds the prioritized list of championship rounds to ingest.
    Priority:
      1. 2026 rounds with 0 < RoundNumber < 19 (pre-Mexico)
      2. 2025 full season
      3. 2024 full season
      4. 2023 season rounds (8 to 22)
      5. 2022 and 2021 (for verification/resume completeness)
    """
    prioritized: List[Dict[str, Any]] = []

    # 1. 2026 (Pre-Mexico)
    sched_2026 = fastf1.get_event_schedule(2026, include_testing=False)
    for _, r in sched_2026.iterrows():
        rnd = int(r['RoundNumber'])
        if 0 < rnd < 19:
            prioritized.append({
                'year': 2026,
                'round': rnd,
                'name': str(r['EventName']),
                'location': str(r['Location']),
                'date': pd.to_datetime(r['EventDate']).strftime('%Y-%m-%d'),
                'format': str(r['EventFormat'])
            })

    # 2. 2025 full
    sched_2025 = fastf1.get_event_schedule(2025, include_testing=False)
    for _, r in sched_2025.iterrows():
        rnd = int(r['RoundNumber'])
        if rnd > 0:
            prioritized.append({
                'year': 2025,
                'round': rnd,
                'name': str(r['EventName']),
                'location': str(r['Location']),
                'date': pd.to_datetime(r['EventDate']).strftime('%Y-%m-%d'),
                'format': str(r['EventFormat'])
            })

    # 3. 2024 full
    sched_2024 = fastf1.get_event_schedule(2024, include_testing=False)
    for _, r in sched_2024.iterrows():
        rnd = int(r['RoundNumber'])
        if rnd > 0:
            prioritized.append({
                'year': 2024,
                'round': rnd,
                'name': str(r['EventName']),
                'location': str(r['Location']),
                'date': pd.to_datetime(r['EventDate']).strftime('%Y-%m-%d'),
                'format': str(r['EventFormat'])
            })

    # 4. 2023 remaining
    sched_2023 = fastf1.get_event_schedule(2023, include_testing=False)
    for _, r in sched_2023.iterrows():
        rnd = int(r['RoundNumber'])
        if rnd >= 8:
            prioritized.append({
                'year': 2023,
                'round': rnd,
                'name': str(r['EventName']),
                'location': str(r['Location']),
                'date': pd.to_datetime(r['EventDate']).strftime('%Y-%m-%d'),
                'format': str(r['EventFormat'])
            })

    # 5. 2022 and 2021 (already ingested, will be skipped)
    for y in [2022, 2021]:
        sched = fastf1.get_event_schedule(y, include_testing=False)
        for _, r in sched.iterrows():
            rnd = int(r['RoundNumber'])
            if rnd > 0:
                prioritized.append({
                    'year': y,
                    'round': rnd,
                    'name': str(r['EventName']),
                    'location': str(r['Location']),
                    'date': pd.to_datetime(r['EventDate']).strftime('%Y-%m-%d'),
                    'format': str(r['EventFormat'])
                })

    return prioritized


def save_atomic_csv(df: pd.DataFrame, output_path: str, temp_path: str):
    """Atomically saves dataframe to prevent partial file corruptions."""
    df[REQUIRED_COLUMNS].to_csv(temp_path, index=False)
    os.replace(temp_path, output_path)


def write_pending_rounds(pending_list: List[Dict[str, Any]]):
    """Writes the remaining uncollected rounds to pending_rounds.txt."""
    with open(PENDING_ROUNDS_FILE, 'w', encoding='utf-8') as f:
        for item in pending_list:
            f.write(f"Year: {item['year']} | Round: {item['round']} | Event: {item['name']} | Location: {item['location']}\n")


def run_pipeline(auto_wait: bool = False):
    """Executes the resumable data ingestion pipeline."""
    print("=================================================================")
    print("STARTING F1 ALL-RACES DATA INGESTION PIPELINE (2021–2026)")
    print(f"Auto-wait mode: {'ENABLED (61 min sleep on rate limit)' if auto_wait else 'DISABLED (Exit gracefully)'}")
    print("=================================================================")

    # 1. Load existing CSV and find ingested rounds
    existing_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    ingested_rounds: Set[Tuple[int, int]] = set()

    if os.path.exists(OUTPUT_CSV):
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            for _, r in existing_df.iterrows():
                ingested_rounds.add((int(r['Year']), int(r['Round'])))
            print(f"Loaded existing dataset: {len(existing_df)} rows ({len(ingested_rounds)} unique rounds).")
        except Exception as e:
            print(f"Warning: Could not parse existing CSV: {e}")

    # Clear old rate-limit entries from failed_rounds.txt
    open(FAILED_ROUNDS_FILE, 'w', encoding='utf-8').close()

    # Fast-path idempotency check: if all 132 championship rounds are ingested, exit immediately
    TOTAL_EXPECTED_ROUNDS = 132
    if len(ingested_rounds) >= TOTAL_EXPECTED_ROUNDS:
        print(f"All {len(ingested_rounds)} expected championship rounds are already ingested! Remaining to fetch: 0")
        open(PENDING_ROUNDS_FILE, 'w', encoding='utf-8').close()
        print("\n=================================================================")
        print("INGESTION RUN SUMMARY")
        print("=================================================================")
        print(f"Final dataset rows: {len(existing_df)}")
        print("Total execution time: 0.00 seconds")
        print("Stop reason: Completed all queued rounds (idempotency verified).")
        return

    # 2. Build prioritized queue
    all_queue = get_prioritized_rounds()
    active_queue = [q for q in all_queue if (q['year'], q['round']) not in ingested_rounds]
    print(f"Total prioritized rounds: {len(all_queue)} | Remaining to fetch: {len(active_queue)}")

    current_df = existing_df.copy()
    start_time = time.perf_counter()
    stop_reason = "Completed all queued rounds."

    idx = 0
    while idx < len(active_queue):
        item = active_queue[idx]
        year = item['year']
        rnd = item['round']
        name = item['name']
        loc = item['location']
        dt = item['date']
        fmt = item['format']

        print(f"\nProcessing [{idx+1}/{len(active_queue)}] Season {year} - Round {rnd}: {name} ({loc})")
        
        try:
            round_df = process_round_data(year, rnd, name, loc, dt, fmt)
            current_df = pd.concat([current_df, round_df], ignore_index=True)
            save_atomic_csv(current_df, OUTPUT_CSV, TEMP_CSV)
            print(f"   Success: {len(round_df)} drivers collected. Total dataset rows: {len(current_df)}")
            idx += 1

        except Exception as e:
            if is_rate_limit(e):
                print(f"\n   RATE LIMIT ENCOUNTERED: {type(e).__name__}: {e}")
                remaining_rounds = active_queue[idx:]
                write_pending_rounds(remaining_rounds)
                print(f"   Saved {len(remaining_rounds)} pending rounds to {PENDING_ROUNDS_FILE}")
                
                # Save metadata
                meta = {
                    'last_attempted_year': year,
                    'last_attempted_round': rnd,
                    'last_attempted_event': name,
                    'stop_reason': f"RateLimitExceeded ({type(e).__name__})",
                    'total_collected_rows': len(current_df),
                    'total_rounds_collected': len(current_df.groupby(['Year', 'Round'])),
                    'pending_rounds_count': len(remaining_rounds),
                    'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()
                }
                with open(METADATA_FILE, 'w', encoding='utf-8') as mf:
                    json.dump(meta, mf, indent=2)

                if auto_wait:
                    print("   [--auto-wait active] Sleeping for 61 minutes (3660s) before resuming...")
                    time.sleep(3660)
                    print("   Waking up! Resuming data collection...")
                    continue
                else:
                    stop_reason = f"RateLimitExceeded on Season {year} Round {rnd} ({name})"
                    print(f"   Exiting gracefully. Run again later or use --auto-wait to resume automatically.")
                    break
            else:
                # True failure
                print(f"   TRUE FAILURE: {type(e).__name__}: {e}")
                with open(FAILED_ROUNDS_FILE, 'a', encoding='utf-8') as ff:
                    ff.write(f"Year: {year} | Round: {rnd} | Event: {name} | Location: {loc} | Error: {type(e).__name__}: {e}\n")
                idx += 1

    total_time = time.perf_counter() - start_time
    if idx == len(active_queue):
        # All queued rounds successfully fetched: empty pending_rounds.txt
        open(PENDING_ROUNDS_FILE, 'w', encoding='utf-8').close()
        meta = {
            'status': 'ALL_ROUNDS_COMPLETED',
            'stop_reason': 'Completed all queued rounds.',
            'total_collected_rows': len(current_df),
            'total_rounds_collected': len(current_df.groupby(['Year', 'Round'])),
            'pending_rounds_count': 0,
            'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        with open(METADATA_FILE, 'w', encoding='utf-8') as mf:
            json.dump(meta, mf, indent=2)

    print("\n=================================================================")
    print("INGESTION RUN SUMMARY")
    print("=================================================================")
    print(f"Final dataset rows: {len(current_df)}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Stop reason: {stop_reason}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="F1 Ingestion Pipeline")
    parser.add_argument('--auto-wait', action='store_true', help="Wait 61 minutes and resume when rate limited")
    parser.add_argument('--fetch-target-mexico', action='store_true', help="Fetch only the Mexico 2026 target dataset")
    args = parser.parse_args()

    if args.fetch_target_mexico:
        fetch_target_mexico_2026()
    else:
        run_pipeline(auto_wait=args.auto_wait)

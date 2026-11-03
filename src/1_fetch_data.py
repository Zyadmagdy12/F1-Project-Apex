"""
src/1_fetch_data.py
-------------------
Data Ingestion module for F1 Mexico GP Predictor.
Fetches session data (FP2, Qualifying, Race) for Mexico City Grand Prix
across seasons 2022, 2023, 2024, and 2025 using FastF1 and caches requests.

Output:
    data/raw_mexico_data.csv
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import fastf1
from fastf1.core import Session

# Configure logging for clear and structured engineering diagnostics
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("F1DataIngestion")

# Resolve absolute project directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "f1_cache")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "raw_mexico_data.csv")

TARGET_YEARS = [2022, 2023, 2024, 2025]
EVENT_NAME = "Mexico City Grand Prix"


def setup_environment() -> None:
    """
    Initializes cache and output directories.
    Enables FastF1 disk caching to optimize re-runs and prevent rate limits.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        fastf1.Cache.configure(cache_dir=CACHE_DIR)
        logger.info(f"FastF1 cache enabled at: {CACHE_DIR}")
    except Exception as exc:
        logger.warning(f"Could not use Cache.configure ({exc}), falling back to enable_cache.")
        fastf1.Cache.enable_cache(CACHE_DIR)


def extract_fp2_data(session: Session) -> pd.DataFrame:
    """
    Extracts representative practice data from FP2.
    
    Metrics extracted:
        - Driver: Driver code abbreviation (e.g. 'VER', 'HAM')
        - BestFP2Lap: Driver's fastest competitive lap in FP2 (seconds)
        - PracticeTotalLaps: Total laps logged by driver in FP2
        - FastestFP2Lap: Absolute fastest competitive lap of the entire FP2 session (seconds)
    
    Utilizes pick_quicklaps() to filter out in/out laps and installation laps,
    with defensive checks for empty sessions or missing lap data.
    """
    columns = ["Driver", "BestFP2Lap", "PracticeTotalLaps", "FastestFP2Lap"]

    if session is None:
        logger.warning("FP2 session object is None.")
        return pd.DataFrame(columns=columns)

    # Safely inspect session laps
    laps = getattr(session, "laps", None)
    if laps is None or laps.empty:
        logger.warning("FP2 laps dataset is empty or unpopulated.")
        return pd.DataFrame(columns=columns)

    driver_records: List[Dict[str, Any]] = []

    # Attempt to isolate quick laps across entire session
    try:
        quick_laps_all = laps.pick_quicklaps()
    except Exception as err:
        logger.warning(f"pick_quicklaps failed at session level ({err}), using valid laps.")
        quick_laps_all = laps[laps["LapTime"].notna()]

    overall_fastest_sec: float = np.nan
    if not quick_laps_all.empty and quick_laps_all["LapTime"].notna().any():
        min_lap = quick_laps_all["LapTime"].min()
        if hasattr(min_lap, "total_seconds"):
            overall_fastest_sec = min_lap.total_seconds()

    for driver_code, group in laps.groupby("Driver"):
        total_laps_count = int(len(group))
        driver_best_sec: float = np.nan

        try:
            d_quick = group.pick_quicklaps()
        except Exception:
            d_quick = group[group["LapTime"].notna()]

        if not d_quick.empty and d_quick["LapTime"].notna().any():
            driver_min_lap = d_quick["LapTime"].min()
            if hasattr(driver_min_lap, "total_seconds"):
                driver_best_sec = driver_min_lap.total_seconds()

        driver_records.append({
            "Driver": str(driver_code).strip(),
            "BestFP2Lap": driver_best_sec,
            "PracticeTotalLaps": total_laps_count,
            "FastestFP2Lap": overall_fastest_sec
        })

    df_fp2 = pd.DataFrame(driver_records)
    if df_fp2.empty:
        return pd.DataFrame(columns=columns)

    return df_fp2[columns].drop_duplicates(subset=["Driver"])


def extract_qualifying_data(session: Session) -> pd.DataFrame:
    """
    Extracts Qualifying position and lap time metrics.
    
    Metrics extracted:
        - Driver: Driver code abbreviation
        - QualifyingPosition: Driver's official qualifying rank
        - BestQualifyingLap: Minimum qualifying lap time across Q1, Q2, Q3 (seconds)
        - FastestQualifyingLap: Pole / fastest session lap time overall (seconds)
    """
    columns = ["Driver", "QualifyingPosition", "BestQualifyingLap", "FastestQualifyingLap"]

    if session is None:
        logger.warning("Qualifying session object is None.")
        return pd.DataFrame(columns=columns)

    results = getattr(session, "results", None)
    if results is None or results.empty:
        logger.warning("Qualifying results dataset is empty.")
        return pd.DataFrame(columns=columns)

    driver_records: List[Dict[str, Any]] = []

    for _, row in results.iterrows():
        driver = row.get("Abbreviation")
        if pd.isna(driver) or not str(driver).strip():
            continue

        driver = str(driver).strip()
        q_pos = row.get("Position")
        try:
            q_pos_val = float(q_pos) if pd.notna(q_pos) else np.nan
        except (ValueError, TypeError):
            q_pos_val = np.nan

        # Collect lap times recorded across qualifying rounds Q1, Q2, Q3
        valid_seconds = []
        for q_stage in ["Q1", "Q2", "Q3"]:
            stage_time = row.get(q_stage)
            if pd.notna(stage_time) and hasattr(stage_time, "total_seconds"):
                sec = stage_time.total_seconds()
                if sec > 0:
                    valid_seconds.append(sec)

        best_q_sec = min(valid_seconds) if valid_seconds else np.nan

        driver_records.append({
            "Driver": driver,
            "QualifyingPosition": q_pos_val,
            "BestQualifyingLap": best_q_sec,
        })

    df_q = pd.DataFrame(driver_records)
    if df_q.empty:
        return pd.DataFrame(columns=columns)

    valid_best_series = df_q["BestQualifyingLap"].dropna()
    fastest_q_sec = valid_best_series.min() if not valid_best_series.empty else np.nan
    df_q["FastestQualifyingLap"] = fastest_q_sec

    return df_q[columns].drop_duplicates(subset=["Driver"])


def extract_race_data(session: Session) -> pd.DataFrame:
    """
    Extracts the ground-truth race classification and starting grid positions.
    
    Metrics extracted:
        - Driver: Driver abbreviation (primary key)
        - TeamName: Constructor name
        - GridPosition: Starting grid slot
        - Position: Official finishing position (Target candidate)
        - Status: Finishing status description (e.g., 'Finished', 'Retired', 'Collision')
    """
    columns = ["Driver", "TeamName", "GridPosition", "Position", "Status"]

    if session is None:
        logger.warning("Race session object is None.")
        return pd.DataFrame(columns=columns)

    results = getattr(session, "results", None)
    if results is None or results.empty:
        logger.warning("Race results dataset is empty.")
        return pd.DataFrame(columns=columns)

    records: List[Dict[str, Any]] = []

    for _, row in results.iterrows():
        driver = row.get("Abbreviation")
        if pd.isna(driver) or not str(driver).strip():
            continue

        team = row.get("TeamName")
        grid_pos = row.get("GridPosition")
        fin_pos = row.get("Position")
        status = row.get("Status")

        try:
            grid_val = float(grid_pos) if pd.notna(grid_pos) else np.nan
        except (ValueError, TypeError):
            grid_val = np.nan

        try:
            fin_val = float(fin_pos) if pd.notna(fin_pos) else np.nan
        except (ValueError, TypeError):
            fin_val = np.nan

        records.append({
            "Driver": str(driver).strip(),
            "TeamName": str(team).strip() if pd.notna(team) else "Unknown",
            "GridPosition": grid_val,
            "Position": fin_val,
            "Status": str(status).strip() if pd.notna(status) else "Unknown",
        })

    df_race = pd.DataFrame(records)
    if df_race.empty:
        return pd.DataFrame(columns=columns)

    return df_race[columns].drop_duplicates(subset=["Driver"])


def fetch_season_mexico_data(year: int) -> Optional[pd.DataFrame]:
    """
    Fetches and integrates FP2, Qualifying, and Race sessions for a single season.
    Applies defensive exception handling per session so failures in one session
    do not compromise the entire pipeline.
    """
    logger.info(f"--- Fetching Mexico City GP data for Season {year} ---")

    # 1. Fetch Race session (serves as the master driver participant roster)
    race_df = pd.DataFrame()
    try:
        race_session = fastf1.get_session(year, EVENT_NAME, "R")
        # Load laps and results without heavy telemetry
        race_session.load(laps=False, telemetry=False, weather=False, messages=False)
        race_df = extract_race_data(race_session)
        logger.info(f"Season {year} Race: Extracted {len(race_df)} driver records.")
    except Exception as exc:
        logger.error(f"Season {year} Race loading failed: {exc}")

    if race_df.empty:
        logger.warning(f"No race data available for Season {year}. Skipping season.")
        return None

    # 2. Fetch Qualifying session
    quali_df = pd.DataFrame()
    try:
        q_session = fastf1.get_session(year, EVENT_NAME, "Q")
        q_session.load(laps=False, telemetry=False, weather=False, messages=False)
        quali_df = extract_qualifying_data(q_session)
        logger.info(f"Season {year} Qualifying: Extracted {len(quali_df)} driver records.")
    except Exception as exc:
        logger.warning(f"Season {year} Qualifying loading failed: {exc}")

    # 3. Fetch FP2 session
    fp2_df = pd.DataFrame()
    try:
        fp2_session = fastf1.get_session(year, EVENT_NAME, "FP2")
        fp2_session.load(laps=True, telemetry=False, weather=False, messages=False)
        fp2_df = extract_fp2_data(fp2_session)
        logger.info(f"Season {year} FP2: Extracted {len(fp2_df)} driver records.")
    except Exception as exc:
        logger.warning(f"Season {year} FP2 loading failed / telemetry unavailable: {exc}")

    # 4. Perform Left Merge starting from Race roster
    merged_df = race_df.copy()

    if not quali_df.empty:
        merged_df = pd.merge(merged_df, quali_df, on="Driver", how="left")
    else:
        for col in ["QualifyingPosition", "BestQualifyingLap", "FastestQualifyingLap"]:
            merged_df[col] = np.nan

    if not fp2_df.empty:
        merged_df = pd.merge(merged_df, fp2_df, on="Driver", how="left")
    else:
        for col in ["BestFP2Lap", "PracticeTotalLaps", "FastestFP2Lap"]:
            merged_df[col] = np.nan

    # 5. Attach Year column
    merged_df["Year"] = int(year)

    # Order columns consistently
    standard_columns = [
        "Year",
        "Driver",
        "TeamName",
        "GridPosition",
        "Position",
        "Status",
        "QualifyingPosition",
        "BestQualifyingLap",
        "FastestQualifyingLap",
        "BestFP2Lap",
        "PracticeTotalLaps",
        "FastestFP2Lap"
    ]

    return merged_df[standard_columns]


def main() -> None:
    """
    Main execution pipeline for raw data ingestion.
    """
    setup_environment()

    season_dfs: List[pd.DataFrame] = []

    for year in TARGET_YEARS:
        try:
            df_season = fetch_season_mexico_data(year)
            if df_season is not None and not df_season.empty:
                season_dfs.append(df_season)
                logger.info(f"Successfully processed {year}: {len(df_season)} drivers.")
            else:
                logger.warning(f"No valid data consolidated for {year}.")
        except Exception as exc:
            logger.error(f"Unhandled error processing season {year}: {exc}", exc_info=True)

    if not season_dfs:
        logger.critical("Extraction failed for all target seasons. No CSV will be saved.")
        sys.exit(1)

    raw_mexico_df = pd.concat(season_dfs, ignore_index=True)

    # Verification: check for duplicate driver entries per season
    duplicates = raw_mexico_df.duplicated(subset=["Year", "Driver"]).sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate driver records! Dropping duplicates.")
        raw_mexico_df = raw_mexico_df.drop_duplicates(subset=["Year", "Driver"])

    # Persist raw dataset
    raw_mexico_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Raw data successfully saved to: {OUTPUT_FILE}")
    logger.info(f"Total rows: {len(raw_mexico_df)}, Total columns: {len(raw_mexico_df.columns)}")
    logger.info(f"Seasons covered: {raw_mexico_df['Year'].unique().tolist()}")
    logger.info(f"Unique drivers: {raw_mexico_df['Driver'].nunique()}")


if __name__ == "__main__":
    main()

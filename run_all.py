"""
run_all.py
----------
Stage 8: One-Command End-to-End Offline Reproducibility Runner.

Sequentially executes and verifies all offline pipeline stages:
  1. src/2_clean_data.py              (Stage 2: Data Cleaning & Schema Standardization)
  2. src/3_features.py                (Stage 3: Feature Engineering & Automated Leakage Verification)
  3. src/4_evaluate.py                (Stages 4 & 5: Walk-Forward Model Evaluation & Probability Calibration)
  4. src/5_backtest_mexico.py         (Stage 6: Out-of-Sample Backtest on 2026 Mexico City Grand Prix)
  5. src/6_predict_brazil_live.py --dry-run (Stage 7: Pre-Weekend Readiness & Power Rankings for 2026 Brazil GP)
"""

import os
import sys
import time
import subprocess

# Configure UTF-8 encoding for console output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PIPELINE_STEPS = [
    ("Stage 2: Data Cleaning & Schema Standardization", ["src/2_clean_data.py"]),
    ("Stage 3: Feature Engineering & Leakage Verification", ["src/3_features.py"]),
    ("Stages 4 & 5: Walk-Forward Evaluation & Calibration", ["src/4_evaluate.py"]),
    ("Stage 6: Out-of-Sample Backtest (Mexico GP 2026)", ["src/5_backtest_mexico.py"]),
    ("Stage 7: Pre-Weekend Readiness & Power Rankings (Brazil GP 2026)", ["src/6_predict_brazil_live.py", "--dry-run"]),
]


def main():
    print("=" * 90)
    print("F1 MEXICO & BRAZIL 2026 ML PREDICTOR — END-TO-END REPRODUCIBILITY PIPELINE")
    print("=" * 90)

    total_start = time.time()
    step_results = []

    for idx, (title, cmd_args) in enumerate(PIPELINE_STEPS, start=1):
        script_path = os.path.join(BASE_DIR, cmd_args[0])
        full_cmd = [sys.executable, script_path] + cmd_args[1:]
        cmd_display = "python " + " ".join(cmd_args)

        print(f"\n[{idx}/{len(PIPELINE_STEPS)}] Running {title} -> `{cmd_display}`")
        print("-" * 90)

        t0 = time.time()
        proc = subprocess.run(full_cmd, cwd=BASE_DIR)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"\n[ERROR] Step {idx} (`{cmd_display}`) failed with exit code {proc.returncode}!")
            sys.exit(proc.returncode)

        step_results.append((title, cmd_display, elapsed))
        print(f"\n[OK] Completed {title} in {elapsed:.2f}s.")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 90)
    print("ALL 5 OFFLINE PIPELINE STAGES COMPLETED & VERIFIED SUCCESSFULLY")
    print("=" * 90)
    for idx, (title, cmd_display, elapsed) in enumerate(step_results, start=1):
        print(f"  {idx}. {title:<64} | {elapsed:6.2f}s | `{cmd_display}`")
    print("-" * 90)
    print(f"Total End-to-End Execution Time: {total_elapsed:.2f}s")
    print("=" * 90)


if __name__ == '__main__':
    main()

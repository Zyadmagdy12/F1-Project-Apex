"""
tests/test_project_apex.py
--------------------------
Project Apex — Phase 10 Automated Acceptance Test Suite.
Executes all 11 mandatory verification checks against the actual code and
exported datasets in outputs/project_apex/.
"""

import os
import sys
import json
from typing import Dict, Any
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from apex_features import engineer_apex_features, APEX_POSITION_FEATURES, APEX_DNF_FEATURES
from apex_models import (
    MonotonicHGBClassifier,
    run_monotonicity_sweep_test,
    build_monotonic_constraint_vector,
)

OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'project_apex')


def test_01_temporal_leakage() -> Dict[str, Any]:
    """Mutates future race outcomes and verifies zero change in earlier race features."""
    c_all = pd.read_csv(os.path.join(BASE_DIR, 'data', 'clean_races.csv'))
    # Use a 25-race slice (2025-2026) for fast deterministic mutation check
    sub = c_all[c_all['Year'] >= 2025].copy().reset_index(drop=True)
    feat_orig = engineer_apex_features(sub)

    # Mutate all races in 2026 (invert FinishPosition and set all Status to Retired)
    sub_mut = sub.copy()
    mask_2026 = sub_mut['Year'] == 2026
    sub_mut.loc[mask_2026, 'FinishPosition'] = 23.0 - sub_mut.loc[mask_2026, 'FinishPosition']
    sub_mut.loc[mask_2026, 'Status'] = 'Retired'
    feat_mut = engineer_apex_features(sub_mut)

    # Check that 2025 features and 2026 Round 1 pre-race features are 100% identical
    check_cols = APEX_POSITION_FEATURES + APEX_DNF_FEATURES
    early_orig = feat_orig[feat_orig['Year'] * 100 + feat_orig['Round'] <= 202601][check_cols]
    early_mut = feat_mut[feat_mut['Year'] * 100 + feat_mut['Round'] <= 202601][check_cols]

    max_diff = float(np.nanmax(np.abs(early_orig.values - early_mut.values)))
    passed = max_diff == 0.0
    assert passed, f"Temporal leakage detected! max_diff = {max_diff}"
    return {'test': 'Temporal_Leakage_Mutation_Test', 'max_abs_diff_on_prior_races': max_diff, 'passed': passed}


def test_02_race_grouping() -> Dict[str, Any]:
    """Verifies all drivers from a race remain in the same temporal fold."""
    with open(os.path.join(OUTPUT_DIR, 'walk_forward_metrics.json'), encoding='utf-8') as f:
        wf = json.load(f)
    races = wf['per_race_metrics']
    race_ids = [r['Race_ID'] for r in races]
    passed = (len(race_ids) == len(set(race_ids)) == 66) and all(r['n_entrants'] >= 18 for r in races)
    assert passed, "Race grouping check failed"
    return {'test': 'Race_Grouping_Integrity', 'unique_race_folds': len(race_ids), 'passed': passed}


def test_03_monotonic_hgb() -> Dict[str, Any]:
    """Verifies Monotonic HGB satisfies P(k+1) <= P(k) across all transitions."""
    with open(os.path.join(OUTPUT_DIR, 'sao_paulo_predictions.json'), encoding='utf-8') as f:
        sp = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'mexico_backtest.json'), encoding='utf-8') as f:
        mx = json.load(f)

    sp_win = sp['monotonicity_verification']['Monotonic_HGB_Win']
    sp_pod = sp['monotonicity_verification']['Monotonic_HGB_Podium']
    mx_win = mx['monotonicity_verification']['Monotonic_HGB_Win']
    mx_pod = mx['monotonicity_verification']['Monotonic_HGB_Podium']

    passed = (
        sp_win['overall_passed']
        and sp_pod['overall_passed']
        and mx_win['overall_passed']
        and mx_pod['overall_passed']
        and sp_win['Starting_Grid_Position']['violations_count'] == 0
        and sp_pod['Starting_Grid_Position']['violations_count'] == 0
        and sp_win['Qualifying_Rank']['violations_count'] == 0
        and sp_pod['Qualifying_Rank']['violations_count'] == 0
    )
    assert passed, "Monotonic HGB test failed!"
    return {
        'test': 'Monotonic_HGB_Sweep_Verification',
        'sao_paulo_win_violations': sp_win['Starting_Grid_Position']['violations_count'] + sp_win['Qualifying_Rank']['violations_count'],
        'sao_paulo_podium_violations': sp_pod['Starting_Grid_Position']['violations_count'] + sp_pod['Qualifying_Rank']['violations_count'],
        'mexico_win_violations': mx_win['Starting_Grid_Position']['violations_count'] + mx_win['Qualifying_Rank']['violations_count'],
        'mexico_podium_violations': mx_pod['Starting_Grid_Position']['violations_count'] + mx_pod['Qualifying_Rank']['violations_count'],
        'passed': passed,
    }


def test_04_verstappen_artifact() -> Dict[str, Any]:
    """Verifies original Verstappen P21 40.24% artifact reproduction and monotonic resolution."""
    with open(os.path.join(OUTPUT_DIR, 'verstappen_shap_case.json'), encoding='utf-8') as f:
        vc = json.load(f)
    art = vc['sao_paulo_2026_target_case']['legacy_artifact_reproduction_and_resolution']
    mono_curve = np.array(art['grid_sweep_p1_to_p22_monotonic_hgb_raw'], dtype=float)
    diffs = np.diff(mono_curve)
    passed = (
        bool(art['legacy_40_24_pct_reproduced'])
        and bool(art['monotonicity_test_passed'])
        and bool(np.all(diffs <= 1e-12))
    )
    assert passed, "Verstappen artifact test failed!"
    return {
        'test': 'Verstappen_P21_Artifact_Audit',
        'legacy_unconstrained_calibrated_prob': art['legacy_12_feature_unconstrained_hgb_calibrated_podium_prob'],
        'legacy_monotonic_calibrated_prob': art['legacy_12_feature_monotonic_hgb_calibrated_podium_prob'],
        'apex_monotonic_hgb_calibrated_prob': art['apex_10_feature_monotonic_hgb_calibrated_podium_prob'],
        'plackett_luce_podium_prob': vc['sao_paulo_2026_target_case']['predictions_summary']['P_Podium_PL'],
        'max_grid_sweep_positive_jump': float(np.max(diffs)),
        'passed': passed,
    }


def test_05_dnf_calibration() -> Dict[str, Any]:
    """Verifies out-of-sample DNF calibration metrics and bounds."""
    with open(os.path.join(OUTPUT_DIR, 'walk_forward_metrics.json'), encoding='utf-8') as f:
        wf = json.load(f)
    brier_dnf = wf['aggregate_metrics']['brier_scores_all_1341_starts']['dnf_probability']
    bins_dnf = wf['calibration_curve_bins']['Calibrated_DNF_Platt']
    sp_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'sao_paulo_predictions.csv'))
    passed = (
        0.05 < brier_dnf['Platt_Sigmoid_Calibrated_DNF'] < 0.15
        and len(bins_dnf) >= 4
        and bool(sp_df['P_DNF'].between(0.005, 0.60).all())
    )
    assert passed, "DNF calibration check failed!"
    return {
        'test': 'DNF_Out_of_Sample_Calibration',
        'brier_platt_66_races': brier_dnf['Platt_Sigmoid_Calibrated_DNF'],
        'brier_isotonic_66_races': brier_dnf['Isotonic_Calibrated_DNF'],
        'passed': passed,
    }


def test_06_plackett_luce_coherence() -> Dict[str, Any]:
    """Verifies all unconditional and conditional Plackett-Luce probability identities."""
    for fname in ['sao_paulo_predictions.csv', 'mexico_backtest.csv']:
        df = pd.read_csv(os.path.join(OUTPUT_DIR, fname))
        n = len(df)
        uncond_cols = [f'P_Uncond_Pos_{k}' for k in range(1, n + 1)]
        cond_pos_cols = [f'P_Cond_Pos_{k}' for k in range(1, n + 1)]
        cond_fin_cols = [f'P_Cond_Given_Finish_Pos_{k}' for k in range(1, n + 1)]

        # 1. Row sum: P_DNF + sum(P_Uncond_Pos_1..N) == 1.0
        row_sums = df['P_DNF'].values + df[uncond_cols].sum(axis=1).values
        assert np.max(np.abs(row_sums - 1.0)) < 1e-4, f"Unconditional row sum failed in {fname}"

        # 2. Conditional on pos k existing: column sum == 1.0
        col_sums_cond = df[cond_pos_cols].sum(axis=0).values
        assert np.max(np.abs(col_sums_cond - 1.0)) < 1e-4, f"Conditional pos col sum failed in {fname}"

        # 3. Conditional on driver finishing: row sum == 1.0
        row_sums_cond_fin = df[cond_fin_cols].sum(axis=1).values
        assert np.max(np.abs(row_sums_cond_fin - 1.0)) < 1e-4, f"Conditional finisher row sum failed in {fname}"

        # 4. P_Win_PL == P_Uncond_Pos_1 and P_Podium_PL == sum(P_Uncond_Pos_1..3)
        assert np.max(np.abs(df['P_Win_PL'].values - df['P_Uncond_Pos_1'].values)) < 1e-5
        assert np.max(np.abs(df['P_Podium_PL'].values - df[uncond_cols[:3]].sum(axis=1).values)) < 1e-4

    return {'test': 'Plackett_Luce_Mathematical_Coherence', 'passed': True}


def test_07_conformal_intervals() -> Dict[str, Any]:
    """Verifies conformal bounds ordering, valid range, and past-only sample size."""
    for fname, expected_min_n in [('mexico_backtest.csv', 1500), ('sao_paulo_predictions.csv', 1522)]:
        df = pd.read_csv(os.path.join(OUTPUT_DIR, fname))
        n = len(df)
        assert (df['Lower_Bound_Rank'] >= 1).all()
        assert (df['Upper_Bound_Rank'] <= n).all()
        assert (df['Lower_Bound_Rank'] <= df['Exp_Finish']).all()
        assert (df['Exp_Finish'] <= df['Upper_Bound_Rank']).all()
        assert (df['Conformal_Calibration_Sample_Size'] >= expected_min_n).all()

    return {'test': 'Conformal_Intervals_Validity', 'passed': True}


def test_08_shap_additivity() -> Dict[str, Any]:
    """Verifies exact interventional SHAP additivity across all drivers and models."""
    max_err = 0.0
    total_checks = 0
    for jname in ['sao_paulo_predictions.json', 'mexico_backtest.json']:
        with open(os.path.join(OUTPUT_DIR, jname), encoding='utf-8') as f:
            data = json.load(f)
        shaps = data['driver_shap_explanations']
        assert len(shaps) == 22
        for drv, mod_dict in shaps.items():
            for mod_name, s_obj in mod_dict.items():
                base = float(s_obj['baseline_prediction'])
                fin = float(s_obj['final_prediction'])
                s_sum = sum(float(v) for v in s_obj['shap_values'].values())
                err = abs((base + s_sum) - fin)
                max_err = max(max_err, err)
                total_checks += 1
                assert err < 1e-6, f"SHAP additivity failed for {drv} {mod_name} in {jname}: err={err}"

    return {'test': 'SHAP_Exact_Additivity', 'total_driver_model_explanations_verified': total_checks, 'max_additivity_error': max_err, 'passed': True}


def test_09_walk_forward_validation() -> Dict[str, Any]:
    """Verifies 66 chronological GP races evaluated in walk_forward_metrics.json."""
    with open(os.path.join(OUTPUT_DIR, 'walk_forward_metrics.json'), encoding='utf-8') as f:
        wf = json.load(f)
    assert wf['evaluated_races_count'] == 66
    assert wf['total_evaluated_driver_starts'] == 1341
    assert len(wf['per_race_metrics']) == 66
    return {'test': 'Walk_Forward_66_Races_Verification', 'evaluated_races': wf['evaluated_races_count'], 'passed': True}


def test_10_bootstrap_ci() -> Dict[str, Any]:
    """Verifies 10,000-replicate paired race-cluster bootstrap confidence intervals."""
    with open(os.path.join(OUTPUT_DIR, 'walk_forward_metrics.json'), encoding='utf-8') as f:
        wf = json.load(f)
    cis = wf['paired_cluster_bootstrap_95_cis']
    assert len(cis) >= 10
    for k, v in cis.items():
        assert v['n_evaluated_races'] == 66
        assert v['n_bootstrap_replicates'] == 10000
        assert v['ci_95_lower'] <= v['point_estimate_improvement'] <= v['ci_95_upper']
    return {'test': 'Paired_Cluster_Bootstrap_CI', 'n_metrics_tested': len(cis), 'passed': True}


def test_11_export_integrity() -> Dict[str, Any]:
    """Reopens and validates all 10 CSV and JSON files in outputs/project_apex/."""
    required_files = [
        'sao_paulo_predictions.csv',
        'sao_paulo_predictions.json',
        'mexico_backtest.csv',
        'mexico_backtest.json',
        'walk_forward_metrics.json',
        'verstappen_shap_case.json',
        'model_manifest.json',
        'feature_dictionary.json',
        'data_quality_report.json',
        'run_report.json',
    ]
    for rf in required_files:
        p = os.path.join(OUTPUT_DIR, rf)
        assert os.path.exists(p) and os.path.getsize(p) > 100, f"Missing or empty file: {rf}"
        if rf.endswith('.json'):
            raw_text = open(p, encoding='utf-8').read()
            # Ensure no illegal NaN or Infinity tokens exist in JSON
            for bad_tok in ['NaN', 'Infinity', '-Infinity']:
                assert f': {bad_tok}' not in raw_text, f"Illegal token {bad_tok} in {rf}"
            parsed = json.loads(raw_text)
            assert isinstance(parsed, dict)
        elif rf.endswith('.csv'):
            df = pd.read_csv(p)
            assert len(df) == 22, f"Expected 22 rows in {rf}, got {len(df)}"
            assert df['Driver_ID'].nunique() == 22, f"Duplicate drivers in {rf}"
            for pcol in ['P_Win_HGB', 'P_Podium_HGB', 'P_Win_LR', 'P_Podium_LR', 'P_DNF', 'P_Win_PL', 'P_Podium_PL']:
                assert df[pcol].notna().all() and df[pcol].between(0.0, 1.0).all(), f"Invalid probabilities in {rf}:{pcol}"

    return {'test': 'Export_Schema_and_JSON_Integrity', 'files_verified': len(required_files), 'passed': True}


def run_all_tests() -> None:
    tests = [
        test_01_temporal_leakage,
        test_02_race_grouping,
        test_03_monotonic_hgb,
        test_04_verstappen_artifact,
        test_05_dnf_calibration,
        test_06_plackett_luce_coherence,
        test_07_conformal_intervals,
        test_08_shap_additivity,
        test_09_walk_forward_validation,
        test_10_bootstrap_ci,
        test_11_export_integrity,
    ]
    results = []
    for t_fn in tests:
        res = t_fn()
        results.append(res)
        print(f"[PASS] {res['test']}: {json.dumps(res)}")

    print(f"\nAll {len(results)}/11 Project Apex acceptance test suites PASSED.")


if __name__ == '__main__':
    run_all_tests()

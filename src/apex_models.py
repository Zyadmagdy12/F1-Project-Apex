"""
src/apex_models.py
------------------
Project Apex — Statistical & Machine Learning Models:
  - Phase 1: Monotonic HistGradientBoostingClassifier (Win & Podium) + Monotonicity Audit
  - Phase 3: Calibrated DNF Hazard Classifier (Platt Sigmoid vs. Isotonic out-of-fold calibration)
  - Phase 4: Listwise Plackett-Luce Ranking Model + Coherent DNF-Integrated Position Matrices
  - Phase 5: Leakage-Safe RidgeCV + Out-of-Sample Walk-Forward Conformal Prediction Intervals
  - Phase 6: Exact Interventional Shapley Value Explainer (Full 2^M Coalition Enumeration)
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import pandas as pd
from scipy.special import logit, expit, logsumexp, comb
from scipy.optimize import minimize, brentq
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss


def build_monotonic_constraint_vector(feature_names: List[str]) -> List[int]:
    """
    Phase 1.1: Dynamically constructs the monotonic constraint vector from the
    actual feature column order.
      - Qualifying_Rank ('quali_pos'): -1
      - Starting_Grid_Position ('grid_pos'): -1
      - All other features: 0
    Larger numerical rank/grid values represent worse positions and must never
    increase predicted Win or Podium probability when other features are held constant.
    """
    neg_features = {'Qualifying_Rank', 'Starting_Grid_Position', 'quali_pos', 'grid_pos'}
    return [-1 if col in neg_features else 0 for col in feature_names]


def normalize_race_probabilities(raw_probs: np.ndarray, target_sum: float = 1.0) -> np.ndarray:
    """
    Normalizes race-level probabilities so sum(p) == target_sum while keeping p_i in (0, 1).
    Uses proportional normalization for target_sum == 1.0 and log-odds shift for target_sum > 1.0.
    Both transformations are strictly increasing functions of each driver's raw probability.
    """
    p = np.clip(np.asarray(raw_probs, dtype=float), 1e-6, 1.0 - 1e-6)
    s = float(np.sum(p))
    if s <= 0.0:
        return np.full_like(p, target_sum / len(p))
    if abs(target_sum - 1.0) < 1e-9:
        return p / s
    logits = logit(p)
    try:
        delta = brentq(lambda d: float(np.sum(expit(logits + d))) - target_sum, -25.0, 25.0)
        return expit(logits + delta)
    except ValueError:
        return np.clip(p * (target_sum / s), 1e-6, 1.0 - 1e-6)


class MonotonicHGBClassifier:
    """
    Phase 1: Dedicated HistGradientBoostingClassifier wrapper enforcing monotonic
    constraints on Qualifying_Rank and Starting_Grid_Position, with positive-class
    label verification and optional out-of-fold Platt/Sigmoid calibration.
    """

    def __init__(
        self,
        feature_names: List[str],
        enforce_monotonic: bool = True,
        max_depth: int = 3,
        min_samples_leaf: int = 20,
        l2_regularization: float = 1.0,
        learning_rate: float = 0.05,
        max_iter: int = 100,
        random_state: int = 42,
    ):
        self.feature_names = list(feature_names)
        self.enforce_monotonic = enforce_monotonic
        self.monotonic_cst = build_monotonic_constraint_vector(self.feature_names) if enforce_monotonic else None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state

        self.imputer = SimpleImputer(strategy='median')
        self.model = HistGradientBoostingClassifier(
            monotonic_cst=self.monotonic_cst,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.pos_class_idx: int = 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MonotonicHGBClassifier':
        X_imp = self.imputer.fit_transform(np.asarray(X, dtype=float))
        y_arr = np.asarray(y, dtype=int)
        self.model.fit(X_imp, y_arr)
        classes = list(self.model.classes_)
        if 1 not in classes:
            raise ValueError(f"Positive class label 1 not found in fitted classes: {classes}")
        self.pos_class_idx = int(classes.index(1))
        return self

    def predict_proba_positive(self, X: np.ndarray) -> np.ndarray:
        X_imp = self.imputer.transform(np.asarray(X, dtype=float))
        probs = self.model.predict_proba(X_imp)[:, self.pos_class_idx]
        return np.clip(probs, 1e-6, 1.0 - 1e-6)


def run_monotonicity_sweep_test(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    feature_names: List[str],
    X_reference: np.ndarray,
    max_position: int = 22,
    tol: float = 1e-12,
) -> Dict[str, Any]:
    """
    Phase 1.3: Automated monotonicity verification test.
    Varies Qualifying_Rank across 1..max_position while holding all other features constant,
    and independently varies Starting_Grid_Position across 1..max_position while holding
    all other features constant.
    Checks P(pos = k + 1) <= P(pos = k) + tol for every row in X_reference and every k.
    """
    X_ref = np.asarray(X_reference, dtype=float)
    q_idx = feature_names.index('Qualifying_Rank') if 'Qualifying_Rank' in feature_names else feature_names.index('quali_pos')
    g_idx = feature_names.index('Starting_Grid_Position') if 'Starting_Grid_Position' in feature_names else feature_names.index('grid_pos')

    results = {}
    for col_name, col_idx in [('Qualifying_Rank', q_idx), ('Starting_Grid_Position', g_idx)]:
        violations = 0
        max_violation = 0.0
        total_transitions = 0
        for row_i in range(len(X_ref)):
            grid_X = np.repeat(X_ref[row_i : row_i + 1], max_position, axis=0)
            grid_X[:, col_idx] = np.arange(1, max_position + 1, dtype=float)
            p_seq = predict_fn(grid_X)
            diffs = np.diff(p_seq)  # p(k+1) - p(k); must be <= tol
            total_transitions += len(diffs)
            bad = diffs[diffs > tol]
            if len(bad) > 0:
                violations += int(len(bad))
                max_violation = max(max_violation, float(np.max(bad)))
        results[col_name] = {
            'passed': violations == 0,
            'violations_count': int(violations),
            'total_transitions_tested': int(total_transitions),
            'max_positive_jump': float(max_violation),
            'tolerance': float(tol),
        }

    results['overall_passed'] = bool(results['Qualifying_Rank']['passed'] and results['Starting_Grid_Position']['passed'])
    return results


class CalibratedBinaryLogisticModel:
    """
    Leakage-safe Imputer + StandardScaler + LogisticRegression binary probability model.
    """

    def __init__(self, C: float = 0.5, random_state: int = 42):
        self.C = C
        self.random_state = random_state
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.lr = LogisticRegression(C=self.C, max_iter=1000, random_state=self.random_state)
        self.pos_class_idx: int = 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CalibratedBinaryLogisticModel':
        X_imp = self.imputer.fit_transform(np.asarray(X, dtype=float))
        X_sc = self.scaler.fit_transform(X_imp)
        self.lr.fit(X_sc, np.asarray(y, dtype=int))
        classes = list(self.lr.classes_)
        self.pos_class_idx = int(classes.index(1))
        return self

    def predict_proba_positive(self, X: np.ndarray) -> np.ndarray:
        X_imp = self.imputer.transform(np.asarray(X, dtype=float))
        X_sc = self.scaler.transform(X_imp)
        return np.clip(self.lr.predict_proba(X_sc)[:, self.pos_class_idx], 1e-6, 1.0 - 1e-6)


class CalibratedDNFHazardModel:
    """
    Phase 3: Dedicated DNF Hazard Classifier with strictly past out-of-fold
    comparison between Platt (Sigmoid) calibration and Isotonic calibration.
    """

    def __init__(self, feature_names: List[str], random_state: int = 42):
        self.feature_names = list(feature_names)
        self.random_state = random_state
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.base_lr = LogisticRegression(C=0.15, max_iter=1000, random_state=self.random_state)
        self.base_hgb = HistGradientBoostingClassifier(
            max_depth=2,
            min_samples_leaf=35,
            l2_regularization=5.0,
            learning_rate=0.03,
            max_iter=80,
            random_state=self.random_state,
        )
        self.platt_calibrator = LogisticRegression(C=10.0, max_iter=500, random_state=self.random_state)
        self.iso_calibrator = IsotonicRegression(y_min=0.01, y_max=0.55, increasing=True, out_of_bounds='clip')
        self.selected_method: str = 'platt_sigmoid'
        self.oof_diagnostics: Dict[str, Any] = {}

    def _raw_blend(self, X_imp: np.ndarray, X_sc: np.ndarray, lr_mod: LogisticRegression, hgb_mod: HistGradientBoostingClassifier) -> np.ndarray:
        idx_lr = list(lr_mod.classes_).index(1)
        idx_hgb = list(hgb_mod.classes_).index(1)
        p_lr = lr_mod.predict_proba(X_sc)[:, idx_lr]
        p_hgb = hgb_mod.predict_proba(X_imp)[:, idx_hgb]
        return np.clip(0.65 * p_lr + 0.35 * p_hgb, 1e-5, 1.0 - 1e-5)

    def fit(self, X: np.ndarray, y: np.ndarray, race_groups: np.ndarray) -> 'CalibratedDNFHazardModel':
        """
        Fits the base DNF hazard estimator and selects Platt vs. Isotonic calibration
        using strictly past-only expanding temporal folds (grouped by race_groups).
        Never calibrates on in-sample predictions.
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        groups = np.asarray(race_groups)
        unique_groups = sorted(np.unique(groups))

        # Generate temporally valid out-of-fold raw predictions across past training races
        oof_raw = np.full(len(y_arr), np.nan, dtype=float)
        warmup_races = max(20, len(unique_groups) // 3)
        step = max(1, (len(unique_groups) - warmup_races) // 12)

        for k in range(warmup_races, len(unique_groups), step):
            val_races = unique_groups[k : k + step]
            train_mask = groups < val_races[0]
            val_mask = np.isin(groups, val_races)
            if np.sum(train_mask) < 100 or len(np.unique(y_arr[train_mask])) < 2:
                continue
            imp_k = SimpleImputer(strategy='median').fit(X_arr[train_mask])
            X_tr_imp = imp_k.transform(X_arr[train_mask])
            sc_k = StandardScaler().fit(X_tr_imp)
            X_tr_sc = sc_k.transform(X_tr_imp)

            lr_k = LogisticRegression(C=0.15, max_iter=1000, random_state=self.random_state).fit(X_tr_sc, y_arr[train_mask])
            hgb_k = HistGradientBoostingClassifier(
                max_depth=2, min_samples_leaf=35, l2_regularization=5.0, learning_rate=0.03, max_iter=80, random_state=self.random_state
            ).fit(X_tr_imp, y_arr[train_mask])

            X_val_imp = imp_k.transform(X_arr[val_mask])
            X_val_sc = sc_k.transform(X_val_imp)
            oof_raw[val_mask] = self._raw_blend(X_val_imp, X_val_sc, lr_k, hgb_k)

        valid_oof = ~np.isnan(oof_raw)
        p_oof = oof_raw[valid_oof]
        y_oof = y_arr[valid_oof]

        # Evaluate Platt vs Isotonic using chronological split of the OOF predictions (first 60% fit calibrator, next 40% score)
        n_oof = len(p_oof)
        split_idx = int(0.6 * n_oof)
        p_cal_tr, y_cal_tr = p_oof[:split_idx], y_oof[:split_idx]
        p_cal_te, y_cal_te = p_oof[split_idx:], y_oof[split_idx:]

        platt_tmp = LogisticRegression(C=10.0, max_iter=500, random_state=self.random_state)
        platt_tmp.fit(logit(p_cal_tr).reshape(-1, 1), y_cal_tr)
        p_platt_te = np.clip(platt_tmp.predict_proba(logit(p_cal_te).reshape(-1, 1))[:, 1], 0.01, 0.55)

        iso_tmp = IsotonicRegression(y_min=0.01, y_max=0.55, increasing=True, out_of_bounds='clip')
        iso_tmp.fit(p_cal_tr, y_cal_tr)
        p_iso_te = np.clip(iso_tmp.predict(p_cal_te), 0.01, 0.55)

        brier_raw = float(brier_score_loss(y_cal_te, p_cal_te))
        brier_platt = float(brier_score_loss(y_cal_te, p_platt_te))
        brier_iso = float(brier_score_loss(y_cal_te, p_iso_te))
        ll_platt = float(log_loss(y_cal_te, p_platt_te))
        ll_iso = float(log_loss(y_cal_te, p_iso_te))

        self.selected_method = 'platt_sigmoid' if brier_platt <= brier_iso else 'isotonic'
        self.oof_diagnostics = {
            'oof_sample_size': int(n_oof),
            'brier_raw_uncalibrated': round(brier_raw, 6),
            'brier_platt_sigmoid': round(brier_platt, 6),
            'brier_isotonic': round(brier_iso, 6),
            'logloss_platt_sigmoid': round(ll_platt, 6),
            'logloss_isotonic': round(ll_iso, 6),
            'selected_calibration_method': self.selected_method,
        }

        # Fit final calibrators on all out-of-fold predictions (strictly out-of-sample relative to base learners)
        self.platt_calibrator.fit(logit(p_oof).reshape(-1, 1), y_oof)
        self.iso_calibrator.fit(p_oof, y_oof)

        # Fit final base learners on full training set
        X_full_imp = self.imputer.fit_transform(X_arr)
        X_full_sc = self.scaler.fit_transform(X_full_imp)
        self.base_lr.fit(X_full_sc, y_arr)
        self.base_hgb.fit(X_full_imp, y_arr)
        return self

    def predict_proba_calibrated(self, X: np.ndarray, method: Optional[str] = None) -> np.ndarray:
        X_imp = self.imputer.transform(np.asarray(X, dtype=float))
        X_sc = self.scaler.transform(X_imp)
        raw_p = self._raw_blend(X_imp, X_sc, self.base_lr, self.base_hgb)
        m = method or self.selected_method
        if m == 'isotonic':
            cal_p = self.iso_calibrator.predict(raw_p)
        else:
            cal_p = self.platt_calibrator.predict_proba(logit(raw_p).reshape(-1, 1))[:, 1]
        return np.clip(cal_p, 0.01, 0.55)


class PlackettLuceRankingModel:
    """
    Phase 4: L2-regularized Listwise Plackett-Luce Ranking Model trained on
    historical classified race finishing orders, integrated with calibrated DNF
    probabilities to produce coherent Unconditional and Conditional position distributions.
    """

    def __init__(self, feature_names: List[str], l2_lambda: float = 0.2, random_state: int = 42):
        self.feature_names = list(feature_names)
        self.l2_lambda = l2_lambda
        self.random_state = random_state
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.weights: np.ndarray = np.zeros(len(feature_names), dtype=float)

    def fit(
        self,
        X: np.ndarray,
        finish_positions: np.ndarray,
        is_classified: np.ndarray,
        race_groups: np.ndarray,
    ) -> 'PlackettLuceRankingModel':
        """
        Maximizes the regularized Plackett-Luce listwise log-likelihood across
        training races. Only classified finishers ('is_classified == 1') form the
        ordered choice sequence, while all starters enter the initial risk set
        (avoiding inventing finishing orders among unclassified DNFs).
        """
        X_imp = self.imputer.fit_transform(np.asarray(X, dtype=float))
        Z = self.scaler.fit_transform(X_imp)
        fin_pos = np.asarray(finish_positions, dtype=float)
        classified = np.asarray(is_classified, dtype=int)
        groups = np.asarray(race_groups)

        # Pre-index races for fast vectorized gradient computation
        race_slices: List[Tuple[np.ndarray, int]] = []
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            # Order classified finishers first (by FinishPosition ascending), then non-classified starters
            c_mask = classified[idx] == 1
            c_idx = idx[c_mask][np.argsort(fin_pos[idx[c_mask]])]
            nc_idx = idx[~c_mask]
            ordered_idx = np.concatenate([c_idx, nc_idx])
            m_classified = len(c_idx)
            if m_classified >= 2:
                race_slices.append((ordered_idx, m_classified))

        n_races = max(1, len(race_slices))
        n_features = Z.shape[1]

        def nll_and_grad(w: np.ndarray) -> Tuple[float, np.ndarray]:
            nll = 0.5 * self.l2_lambda * float(np.dot(w, w))
            grad = self.l2_lambda * w.copy()
            eta_all = Z @ w

            for ordered_idx, m_c in race_slices:
                eta = eta_all[ordered_idx]
                Z_r = Z[ordered_idx]
                n_r = len(ordered_idx)
                # For each stage k = 0 .. m_c - 1, choice is k from pool k .. n_r - 1
                # Compute log-sum-exp and softmax weights efficiently
                for k in range(m_c):
                    pool_eta = eta[k:]
                    lse = float(logsumexp(pool_eta))
                    nll -= (eta[k] - lse) / n_races
                    probs_k = np.exp(pool_eta - lse)
                    grad -= (Z_r[k] - (probs_k @ Z_r[k:])) / n_races

            return float(nll), grad

        w0 = np.zeros(n_features, dtype=float)
        res = minimize(nll_and_grad, w0, jac=True, method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-9})
        self.weights = res.x.copy()
        return self

    def predict_utility(self, X: np.ndarray) -> np.ndarray:
        """Returns driver Plackett-Luce utility score eta_i = w^T z_i (higher = stronger finisher)."""
        X_imp = self.imputer.transform(np.asarray(X, dtype=float))
        Z = self.scaler.transform(X_imp)
        return Z @ self.weights

    def simulate_race_distribution(
        self,
        X_race: np.ndarray,
        p_dnf_race: np.ndarray,
        n_simulations: int = 25000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Phase 4.2 & 4.3: Combines calibrated driver DNF probabilities with the
        Plackett-Luce ranking distribution via exact Gumbel-Max sampling over
        stratified survival realizations.
        Guarantees:
          1. Unconditional: P_DNF_i + sum_{k=1..N} P_uncond(i, k) == 1.0 for every driver i
          2. Column identity: sum_{i=1..N} P_uncond(i, k) == P(N_finishers >= k) for every k
          3. Conditional on pos k existing: sum_{i=1..N} P_cond_pos(i | k) == 1.0
          4. Conditional on driver i finishing: sum_{k=1..N} P_cond_drv(k | i finishes) == 1.0
        """
        eta = self.predict_utility(X_race)
        p_dnf = np.clip(np.asarray(p_dnf_race, dtype=float), 0.005, 0.95)
        n_drivers = len(eta)
        S = int(n_simulations)
        rng = np.random.default_rng(seed)

        # Stratified survival realizations so empirical DNF rate equals calibrated p_dnf to within 1/(2S)
        surv_matrix = np.ones((S, n_drivers), dtype=bool)
        for i in range(n_drivers):
            n_retire = int(np.round(S * p_dnf[i]))
            retire_sims = rng.choice(S, size=n_retire, replace=False)
            surv_matrix[retire_sims, i] = False

        # Draw i.i.d. Gumbel(0, 1) perturbations: eta_i + Gumbel(0,1) generates exact Plackett-Luce orderings
        gumbels = rng.gumbel(loc=0.0, scale=1.0, size=(S, n_drivers))
        perturbed_utilities = eta[None, :] + gumbels
        # Mask retired drivers with -inf so they sort after all surviving finishers
        masked_utilities = np.where(surv_matrix, perturbed_utilities, -np.inf)

        # Sort descending across drivers for each simulation
        orderings = np.argsort(-masked_utilities, axis=1)  # shape (S, n_drivers)
        n_finishers_per_sim = np.sum(surv_matrix, axis=1)  # shape (S,)

        uncond_counts = np.zeros((n_drivers, n_drivers), dtype=np.int64)
        for k in range(n_drivers):
            valid_sim_mask = n_finishers_per_sim > k
            drivers_at_k = orderings[valid_sim_mask, k]
            np.add.at(uncond_counts[:, k], drivers_at_k, 1)

        p_uncond = uncond_counts.astype(float) / float(S)
        p_dnf_actual = 1.0 - (np.sum(surv_matrix, axis=0).astype(float) / float(S))
        p_pos_exists = np.sum(p_uncond, axis=0)  # Equals P(N_finishers >= k)

        # Monte Carlo standard errors: sqrt(p*(1-p)/S)
        mc_se_uncond = np.sqrt(np.maximum(0.0, p_uncond * (1.0 - p_uncond)) / float(S))

        # Conditional Matrix 1: P(Driver i occupies position k | position k exists)
        p_cond_given_pos = np.zeros_like(p_uncond)
        for k in range(n_drivers):
            if p_pos_exists[k] > 0:
                p_cond_given_pos[:, k] = p_uncond[:, k] / p_pos_exists[k]

        # Conditional Matrix 2: P(Driver i finishes in position k | Driver i finishes)
        p_cond_given_fin = np.zeros_like(p_uncond)
        for i in range(n_drivers):
            surv_prob_i = 1.0 - p_dnf_actual[i]
            if surv_prob_i > 0:
                p_cond_given_fin[i, :] = p_uncond[i, :] / surv_prob_i

        # Expected finishing position from Plackett-Luce (conditional on finishing, and overall including DNF rank)
        positions_1_to_n = np.arange(1, n_drivers + 1, dtype=float)
        exp_pos_cond_fin = p_cond_given_fin @ positions_1_to_n
        # For overall expected rank (used in Rank MAE), retired drivers place behind finishers (~ (N_fin + 1 + N)/2)
        sim_ranks = np.empty((S, n_drivers), dtype=float)
        row_idx = np.arange(S)[:, None]
        sim_ranks[row_idx, orderings] = np.arange(1, n_drivers + 1, dtype=float)[None, :]
        exp_pos_overall = np.mean(sim_ranks, axis=0)

        p_win_pl = p_uncond[:, 0].copy()
        p_podium_pl = np.sum(p_uncond[:, : min(3, n_drivers)], axis=1)

        return {
            'n_entrants': int(n_drivers),
            'n_simulations': int(S),
            'seed': int(seed),
            'utility_scores': eta,
            'p_dnf': p_dnf_actual,
            'p_unconditional': p_uncond,
            'p_conditional_given_pos_exists': p_cond_given_pos,
            'p_conditional_given_driver_finishes': p_cond_given_fin,
            'p_pos_exists': p_pos_exists,
            'mc_se_unconditional': mc_se_uncond,
            'max_mc_se': float(np.max(mc_se_uncond)),
            'p_win_pl': p_win_pl,
            'p_podium_pl': p_podium_pl,
            'exp_pos_pl_conditional': exp_pos_cond_fin,
            'exp_pos_pl_overall': exp_pos_overall,
            'pred_rank_pl': pd.Series(exp_pos_overall).rank(method='first').values.astype(int),
        }


class LeakageSafeRidgeConformalModel:
    """
    Phase 5: Leakage-safe Imputer + StandardScaler + RidgeCV regression pipeline
    wrapped with a 90% finite-sample split/walk-forward conformal prediction interval
    calibrated strictly on past out-of-sample walk-forward residuals.
    """

    def __init__(self, alphas: Optional[np.ndarray] = None, confidence_level: float = 0.90):
        self.alphas = alphas if alphas is not None else np.logspace(-1, 3, 25)
        self.confidence_level = float(confidence_level)
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.ridge = RidgeCV(alphas=self.alphas)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LeakageSafeRidgeConformalModel':
        X_imp = self.imputer.fit_transform(np.asarray(X, dtype=float))
        X_sc = self.scaler.fit_transform(X_imp)
        self.ridge.fit(X_sc, np.asarray(y, dtype=float))
        return self

    def predict(self, X: np.ndarray, n_entrants: int = 22) -> np.ndarray:
        X_imp = self.imputer.transform(np.asarray(X, dtype=float))
        X_sc = self.scaler.transform(X_imp)
        raw_pred = self.ridge.predict(X_sc)
        # Clip continuous expected finish to valid [1.0, float(n_entrants)] range
        return np.clip(raw_pred, 1.0, float(n_entrants))

    @staticmethod
    def compute_conformal_quantile(past_abs_residuals: np.ndarray, confidence_level: float = 0.90) -> float:
        """
        Phase 5.2: Finite-sample conformal quantile:
          q_level = min(1.0, ceil((n + 1) * confidence_level) / n)
        computed using method='higher' on strictly prior out-of-sample residuals.
        """
        res = np.asarray(past_abs_residuals, dtype=float)
        res = res[~np.isnan(res)]
        n = len(res)
        if n == 0:
            return 6.0
        q_level = min(1.0, float(np.ceil((n + 1) * confidence_level)) / float(n))
        return float(np.quantile(res, q_level, method='higher'))

    def predict_with_conformal_interval(
        self,
        X: np.ndarray,
        past_abs_residuals: np.ndarray,
        n_entrants: int = 22,
    ) -> Dict[str, Any]:
        exp_finish = self.predict(X, n_entrants=n_entrants)
        q_hat = self.compute_conformal_quantile(past_abs_residuals, self.confidence_level)
        lower_cont = np.clip(exp_finish - q_hat, 1.0, float(n_entrants))
        upper_cont = np.clip(exp_finish + q_hat, 1.0, float(n_entrants))
        # Integer rank bounds guaranteed to satisfy Lower_Bound_Rank <= Exp_Finish <= Upper_Bound_Rank
        lower_rank = np.clip(np.floor(lower_cont), 1.0, np.floor(exp_finish)).astype(int)
        upper_rank = np.clip(np.ceil(upper_cont), np.ceil(exp_finish), float(n_entrants)).astype(int)

        return {
            'Exp_Finish': exp_finish,
            'Lower_Bound_Cont': lower_cont,
            'Upper_Bound_Cont': upper_cont,
            'Lower_Bound_Rank': lower_rank,
            'Upper_Bound_Rank': upper_rank,
            'Conformal_Quantile_Width': float(q_hat),
            'Conformal_Confidence_Level': float(self.confidence_level),
            'Conformal_Calibration_Sample_Size': int(len(past_abs_residuals)),
        }


def compute_exact_interventional_shap(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x_instance: np.ndarray,
    X_background: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Phase 6: Computes EXACT Interventional Shapley Values by enumerating the full
    2^M coalition space over the fixed historical background dataset X_background.
    For M = 10 features, 2^10 = 1,024 coalitions.
    Guarantees exact additivity:
      baseline_value + sum(shap_values) == model_prediction(x_instance)
    within machine floating-point precision (~1e-14).
    """
    x = np.asarray(x_instance, dtype=float).reshape(1, -1)
    bg = np.asarray(X_background, dtype=float)
    K, M = bg.shape
    assert x.shape[1] == M == len(feature_names), "Feature dimension mismatch in exact SHAP"

    n_coalitions = 1 << M  # 2^M
    # Build binary mask matrix of shape (2^M, M)
    coalition_indices = np.arange(n_coalitions, dtype=np.int32)[:, None]
    bit_positions = np.arange(M, dtype=np.int32)[None, :]
    masks = ((coalition_indices >> bit_positions) & 1).astype(bool)  # (2^M, M)

    # Construct evaluation array of shape (2^M * K, M)
    # Where mask is True, take x; where False, take background
    bg_expanded = np.broadcast_to(bg[None, :, :], (n_coalitions, K, M))
    x_expanded = np.broadcast_to(x[:, None, :], (n_coalitions, K, M))
    mask_expanded = np.broadcast_to(masks[:, None, :], (n_coalitions, K, M))
    eval_matrix = np.where(mask_expanded, x_expanded, bg_expanded).reshape(n_coalitions * K, M)

    # Single vectorized batch call to predict_fn
    preds_all = predict_fn(eval_matrix).reshape(n_coalitions, K)
    v_coalitions = np.mean(preds_all, axis=1)  # shape (2^M,)

    baseline_val = float(v_coalitions[0])  # All bits 0 -> E[f(X_bg)]
    final_pred = float(v_coalitions[-1])   # All bits 1 -> f(x)

    # Precompute Shapley combinatorial weights: |S|!(M - |S| - 1)! / M! = 1 / (M * comb(M - 1, |S|))
    subset_sizes = np.sum(masks, axis=1)
    shap_values = np.zeros(M, dtype=float)

    for j in range(M):
        bit_j = 1 << j
        # Coalitions S that do NOT contain feature j
        without_j = np.where(~masks[:, j])[0]
        with_j = without_j | bit_j
        s_sizes = subset_sizes[without_j]
        weights = 1.0 / (float(M) * comb(M - 1, s_sizes))
        marginal_diffs = v_coalitions[with_j] - v_coalitions[without_j]
        shap_values[j] = float(np.sum(weights * marginal_diffs))

    reconstructed = baseline_val + float(np.sum(shap_values))
    additivity_error = abs(reconstructed - final_pred)

    return {
        'baseline_prediction': baseline_val,
        'final_prediction': final_pred,
        'reconstructed_prediction': reconstructed,
        'additivity_abs_error': additivity_error,
        'additivity_passed': bool(additivity_error < 1e-6),
        'shap_values': {feat: float(shap_values[i]) for i, feat in enumerate(feature_names)},
        'method': f'Exact Interventional Shapley (Full 2^{M} = {n_coalitions} Coalition Enumeration over K={K} Pre-Race Background Rows)',
        'is_exact': True,
    }

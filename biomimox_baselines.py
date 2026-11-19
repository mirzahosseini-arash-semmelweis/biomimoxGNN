# biomimox_baselines.py
# Ablation arms 0-4: the bar the GNN has to clear.
#
#   0  class prior              stratified dummy
#   1  HOT_SMARTS flags only    L2 logistic regression
#   2a ECFP4 (2048 bit)         L2 logistic regression
#   2b ECFP4 (2048 bit)         random forest
#   3  ECFP4 + 9 RDKit + 4 Percepta        random forest
#   4  arm 3 + HOT_SMARTS flags            random forest
#
# Design points that make this a fair test rather than a straw man:
#   * identical molecules and identical train/test split as the GNN, read from
#     splits/<assay>_split_manifest.csv (written by biomimox_prepare.py)
#   * the same repeated stratified CV protocol, so every number has a spread
#   * each arm gets its own nested hyper-parameter selection on the inner folds
#   * each arm gets an F1-optimal decision threshold chosen on out-of-fold
#     training predictions only, exactly like the GNN
#
# Outputs (ablation_results/):
#   baselines_cv_<assay>.csv          per-repeat CV metrics, per arm
#   baselines_oof_<assay>_<arm>.csv   out-of-fold probabilities
#   baselines_test_<assay>.csv        held-out metrics, per arm per repeat
#   baselines_summary.csv             mean +/- sd across repeats, every assay
#
# Usage:
#   python biomimox_prepare.py && python biomimox_baselines.py

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import biomimox_common as C

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Parallelism lives in the grid search, not inside the forest: nesting
# n_jobs=-1 inside n_jobs=-1 oversubscribes the CPU and runs slower than
# serial on a small machine.
N_JOBS = int(os.environ.get("BIOMIMOX_N_JOBS", "-1"))
RF_TREES = int(os.environ.get("BIOMIMOX_RF_TREES", "500"))
INNER_CV_MAX = 3


# -----------------------------------------------------------------------------
# Arm definitions
# -----------------------------------------------------------------------------
def _logreg_grid() -> Dict[str, List]:
    return {"clf__C": [0.01, 0.1, 1.0, 10.0]}


def _rf_grid() -> Dict[str, List]:
    return {
        "clf__max_features": ["sqrt", 0.1],
        "clf__min_samples_leaf": [1, 3],
    }


def make_logreg(seed: int) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2", solver="liblinear", class_weight="balanced",
            max_iter=5000, random_state=seed)),
    ])


def make_rf(seed: int) -> Pipeline:
    # n_jobs=1 on purpose: the grid search above it does the parallelising.
    return Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=RF_TREES, class_weight="balanced_subsample",
            n_jobs=1, random_state=seed)),
    ])


def make_dummy(seed: int) -> Pipeline:
    return Pipeline([("clf", DummyClassifier(strategy="stratified", random_state=seed))])


# arm id -> (label, feature blocks, estimator factory, param grid)
ARMS: Dict[str, Tuple[str, Tuple[str, ...], Callable[[int], Pipeline], Optional[Dict]]] = {
    "0_prior":          ("0  class prior (dummy)",            ("hot",),                                   make_dummy,  None),
    "1_hot_logreg":     ("1  HOT_SMARTS + logistic",          ("hot",),                                   make_logreg, _logreg_grid()),
    "2a_ecfp4_logreg":  ("2a ECFP4 + logistic",               ("ecfp4",),                                 make_logreg, _logreg_grid()),
    "2b_ecfp4_rf":      ("2b ECFP4 + random forest",          ("ecfp4",),                                 make_rf,     _rf_grid()),
    "3_ecfp4_desc_rf":  ("3  ECFP4 + RDKit + Percepta + RF",  ("ecfp4", "rdkit", "percepta"),             make_rf,     _rf_grid()),
    "4_all_rf":         ("4  arm 3 + HOT_SMARTS + RF",        ("ecfp4", "rdkit", "percepta", "hot"),      make_rf,     _rf_grid()),
}


# -----------------------------------------------------------------------------
# Feature assembly
# -----------------------------------------------------------------------------
def build_feature_blocks(manifest: pd.DataFrame,
                         mol_lookup: pd.DataFrame) -> Dict[str, np.ndarray]:
    """One dict of feature blocks for the molecules in this assay's manifest."""
    smiles = manifest["canonical_smiles"].astype(str).tolist()
    sub = mol_lookup.set_index("orig_idx").loc[manifest["orig_idx"].values]
    return {
        "ecfp4": C.ecfp_matrix(smiles),
        "hot": C.hot_flag_matrix(smiles),
        "rdkit": C.rdkit_descriptor_matrix(smiles),
        "percepta": sub[C.PERCEPTA_COLS].values.astype(np.float32),
    }


def assemble(blocks: Dict[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    return np.hstack([blocks[n] for n in names]).astype(np.float64)


# -----------------------------------------------------------------------------
# Fit / predict helpers
# -----------------------------------------------------------------------------
def fit_predict(estimator_factory: Callable[[int], Pipeline],
                grid: Optional[Dict],
                X_tr: np.ndarray,
                y_tr: np.ndarray,
                X_te: np.ndarray,
                seed: int) -> np.ndarray:
    """Nested selection on the inner folds, then probabilities for X_te."""
    est = estimator_factory(seed)
    counts = np.bincount(np.asarray(y_tr).astype(int), minlength=2)
    n_inner = int(min(INNER_CV_MAX, counts.min()))

    if grid and n_inner >= 2:
        inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
        search = GridSearchCV(est, grid, scoring="average_precision",
                              cv=inner, n_jobs=N_JOBS, refit=True,
                              error_score=np.nan, pre_dispatch="2*n_jobs")
        try:
            search.fit(X_tr, y_tr)
            model = search.best_estimator_
        except Exception:
            model = estimator_factory(seed).fit(X_tr, y_tr)
    else:
        model = est.fit(X_tr, y_tr)

    try:
        proba = model.predict_proba(X_te)
        if proba.shape[1] == 2:
            return proba[:, 1].astype(float)
        return proba.ravel().astype(float)
    except Exception:
        scores = model.decision_function(X_te).astype(float)
        return 1.0 / (1.0 + np.exp(-scores))


def run_arm(arm_id: str,
            X: np.ndarray,
            y: np.ndarray,
            groups: Sequence[str],
            train_pos: np.ndarray,
            test_pos: np.ndarray,
            n_repeats: int = C.CV_REPEATS) -> Tuple[List[Dict], List[Dict], pd.DataFrame]:
    """Repeated CV on the training pool + one held-out evaluation per repeat."""
    label, blocks, factory, grid = ARMS[arm_id]
    X_train_pool, y_train_pool = X[train_pos], y[train_pos]
    X_test, y_test = X[test_pos], y[test_pos]
    groups_pool = [groups[i] for i in train_pos]

    folds = C.repeated_cv_indices(
        y_train_pool, groups=groups_pool,
        n_splits=C.CV_SPLITS, n_repeats=n_repeats,
        seed=C.SEED, mode=C.SPLIT_MODE,
    )

    cv_rows: List[Dict] = []
    test_rows: List[Dict] = []
    oof_records: List[Dict] = []

    by_repeat: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for rep, fold, tr, va in folds:
        by_repeat.setdefault(rep, []).append((tr, va))

    for rep, fold_list in sorted(by_repeat.items()):
        seed = C.SEED + 1000 * rep
        oof = np.full(len(y_train_pool), np.nan, dtype=float)

        for fold_i, (tr, va) in enumerate(fold_list):
            if np.unique(y_train_pool[tr]).size < 2:
                continue
            oof[va] = fit_predict(factory, grid, X_train_pool[tr], y_train_pool[tr],
                                  X_train_pool[va], seed=seed + fold_i)

        threshold = C.optimal_threshold(y_train_pool, oof)
        cv_met = C.threshold_metrics(y_train_pool, oof, threshold)
        cv_met.update({"arm": arm_id, "arm_label": label, "repeat": rep,
                       "stage": "cv_oof", "n_features": int(X.shape[1])})
        cv_rows.append(cv_met)

        for pos_i, (yy, pp) in enumerate(zip(y_train_pool, oof)):
            oof_records.append({"repeat": rep, "train_pos": int(train_pos[pos_i]),
                                "y": int(yy), "prob": float(pp)})

        # Held-out set: refit on the whole training pool, apply the OOF threshold.
        test_prob = fit_predict(factory, grid, X_train_pool, y_train_pool, X_test, seed=seed)
        test_met = C.threshold_metrics(y_test, test_prob, threshold)
        test_met.update({"arm": arm_id, "arm_label": label, "repeat": rep,
                         "stage": "test", "n_features": int(X.shape[1])})
        test_rows.append(test_met)

    return cv_rows, test_rows, pd.DataFrame(oof_records)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    C.set_workdir()
    C.ensure_dirs()
    C.set_seeds(C.SEED)

    if not (C.SPLIT_DIR / "valid_molecules.csv").exists():
        raise RuntimeError("Run biomimox_prepare.py first to write splits/")

    mol_lookup = pd.read_csv(C.SPLIT_DIR / "valid_molecules.csv")

    all_cv: List[pd.DataFrame] = []
    all_test: List[pd.DataFrame] = []
    summary_rows: List[Dict] = []

    for target in C.TARGET_COLUMNS:
        manifest_path = C.SPLIT_DIR / f"{target}_split_manifest.csv"
        if not manifest_path.exists():
            print(f"\n=== {target}: no split manifest; skipped ===")
            continue

        manifest = pd.read_csv(manifest_path)
        y = manifest["y"].values.astype(int)
        groups = manifest["scaffold"].astype(str).tolist()
        train_pos = np.where(manifest["split"].values == "train")[0]
        test_pos = np.where(manifest["split"].values == "test")[0]

        print(f"\n=== {target}: n={len(manifest)} "
              f"(train {len(train_pos)} / test {len(test_pos)}), "
              f"prevalence={y.mean():.3f}, mode={manifest['split_mode'].iloc[0]} ===")

        blocks = build_feature_blocks(manifest, mol_lookup)

        target_cv: List[Dict] = []
        target_test: List[Dict] = []

        for arm_id in ARMS:
            label, block_names, _, _ = ARMS[arm_id]
            X = assemble(blocks, block_names)
            cv_rows, test_rows, oof_df = run_arm(
                arm_id, X, y, groups, train_pos, test_pos, n_repeats=C.CV_REPEATS
            )
            target_cv.extend(cv_rows)
            target_test.extend(test_rows)

            if not oof_df.empty:
                oof_df.insert(0, "target", target)
                oof_df.to_csv(C.RESULTS_DIR / f"baselines_oof_{target}_{arm_id}.csv", index=False)

            agg_cv = C.aggregate_repeat_metrics(cv_rows)
            agg_test = C.aggregate_repeat_metrics(test_rows)
            summary_rows.append({
                "target": target, "arm": arm_id, "arm_label": label,
                "family": "baseline", "n_features": int(X.shape[1]),
                "n_train": int(len(train_pos)), "n_test": int(len(test_pos)),
                "prevalence": float(y.mean()),
                **{f"cv_{k}": v for k, v in agg_cv.items()},
                **{f"test_{k}": v for k, v in agg_test.items()},
            })
            print(f"  {label:<38s} CV PR-AUC {agg_cv['pr_auc_mean']:.3f} "
                  f"+/- {agg_cv['pr_auc_sd']:.3f} | CV ROC {agg_cv['roc_auc_mean']:.3f} "
                  f"| CV MCC {agg_cv['mcc_mean']:.3f}")

        cv_df = pd.DataFrame(target_cv)
        test_df = pd.DataFrame(target_test)
        cv_df.insert(0, "target", target)
        test_df.insert(0, "target", target)
        cv_df.to_csv(C.RESULTS_DIR / f"baselines_cv_{target}.csv", index=False)
        test_df.to_csv(C.RESULTS_DIR / f"baselines_test_{target}.csv", index=False)
        all_cv.append(cv_df)
        all_test.append(test_df)

    if all_cv:
        pd.concat(all_cv, ignore_index=True).to_csv(
            C.RESULTS_DIR / "baselines_cv_all.csv", index=False)
    if all_test:
        pd.concat(all_test, ignore_index=True).to_csv(
            C.RESULTS_DIR / "baselines_test_all.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(C.RESULTS_DIR / "baselines_summary.csv", index=False)
    print(f"\nWrote {C.RESULTS_DIR / 'baselines_summary.csv'}")

    if not summary.empty:
        print("\nBest baseline per assay (by CV PR-AUC):")
        for target, grp in summary.groupby("target", sort=False):
            best = grp.loc[grp["cv_pr_auc_mean"].idxmax()]
            print(f"  {target:<10s} {best['arm_label']:<38s} "
                  f"PR-AUC {best['cv_pr_auc_mean']:.3f} +/- {best['cv_pr_auc_sd']:.3f}")


if __name__ == "__main__":
    main()

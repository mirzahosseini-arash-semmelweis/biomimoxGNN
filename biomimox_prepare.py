# biomimox_prepare.py
# Step 0 of the pipeline. Run this ONCE before the baselines, the single-task
# GNN or the multitask GNN.
#
# It does three things:
#   1. Builds (and caches) the conformer graph dataset, so the expensive RDKit
#      embedding work happens once instead of once per assay per arm.
#   2. Writes ONE global molecule-level train/test partition. Every arm --
#      baselines, single-task GNN, multitask GNN -- inherits it, so a molecule
#      is on the same side of the split everywhere. That is what makes the
#      paired comparison in biomimox_compare.py meaningful.
#   3. Writes a per-assay manifest: the global partition restricted to the
#      molecules that carry a label for that assay.
#
# A per-assay fallback split is used only when the global split would leave an
# assay with no positives on one side; such assays are marked in the manifest
# and excluded from the paired statistics.
#
# Usage:
#   python biomimox_prepare.py
#   BIOMIMOX_SPLIT=random python biomimox_prepare.py     # original-style split
#   BIOMIMOX_NCONF=1 python biomimox_prepare.py          # single conformer

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

import biomimox_common as C

MIN_TEST_POS = 1          # below this, fall back to a per-assay split
MIN_TRAIN_POS = 2


def positive_count_bins(y_wide: pd.DataFrame, targets: List[str]) -> np.ndarray:
    """Stratification label for the global split: how many reaction types fire
    on this molecule, capped at 3. Keeps promiscuous and inert compounds from
    piling up on one side."""
    sub = y_wide[targets].values.astype(float)
    n_pos = np.nansum(sub == 1.0, axis=1)
    return np.minimum(n_pos, 3).astype(int)


def main() -> None:
    C.set_workdir()
    C.ensure_dirs()
    C.set_seeds(C.SEED)

    print("\n=== Loading input tables ===")
    mol_df = C.load_molecule_table()
    print(f"{len(mol_df)} input molecules")
    print(f"{mol_df['scaffold'].nunique()} distinct generic Murcko scaffolds")

    targets = C.load_targets(C.TARGET_COLUMNS, n_rows=len(mol_df))

    print(f"\n=== Building conformer graphs (n_conformers={C.N_CONFORMERS}) ===")
    import biomimox_graphs as G          # imported here: needs torch
    graphs, valid_mask = G.build_graph_dataset(
        mol_df, n_conformers=C.N_CONFORMERS, seed=C.SEED, use_cache=True
    )
    n_valid = int(valid_mask.sum())
    print(f"{n_valid}/{len(mol_df)} molecules produced at least one conformer")
    if n_valid == 0:
        raise RuntimeError("No molecules could be embedded; check the SMILES input")

    conf_counts = np.asarray([len(g) for g in graphs], dtype=int)
    print(f"conformers per molecule: mean={conf_counts.mean():.2f}, "
          f"min={conf_counts.min()}, max={conf_counts.max()}")

    valid_df = mol_df.loc[valid_mask].reset_index(drop=True).copy()
    valid_df["dataset_idx"] = np.arange(len(valid_df), dtype=int)
    valid_df["n_conformers"] = conf_counts
    valid_df.to_csv(C.SPLIT_DIR / "valid_molecules.csv", index=False)
    print(f"Wrote {C.SPLIT_DIR / 'valid_molecules.csv'}")

    y_valid = targets.loc[valid_mask].reset_index(drop=True)
    labels_out = y_valid.copy()
    labels_out.insert(0, "dataset_idx", valid_df["dataset_idx"].values)
    labels_out.insert(1, "orig_idx", valid_df["orig_idx"].values)
    labels_out.to_csv(C.SPLIT_DIR / "labels_valid.csv", index=False)
    print(f"Wrote {C.SPLIT_DIR / 'labels_valid.csv'}")

    # -------------------------------------------------------------------------
    # Global molecule-level split, shared by every arm
    # -------------------------------------------------------------------------
    print(f"\n=== Global train/test split (mode={C.SPLIT_MODE}) ===")
    mt_targets = [t for t in C.MULTITASK_TARGETS if t in y_valid.columns]
    strat_bins = positive_count_bins(y_valid, mt_targets)
    y_any = (strat_bins > 0).astype(int)

    g_train, g_test, g_mode = C.make_split(
        valid_df, y_any, mode=C.SPLIT_MODE, test_size=C.TEST_SIZE, seed=C.SEED
    )
    global_manifest = valid_df.copy()
    global_manifest["n_pos_bin"] = strat_bins
    global_manifest["split"] = "train"
    global_manifest.loc[g_test, "split"] = "test"
    global_manifest["split_mode"] = g_mode
    global_cols = ["dataset_idx", "orig_idx", "SMILES", "canonical_smiles",
                   "scaffold", "n_pos_bin", "split", "split_mode"]
    global_manifest[global_cols].to_csv(C.SPLIT_DIR / "global_split_manifest.csv", index=False)
    print(f"global split: train {len(g_train)} / test {len(g_test)} (mode {g_mode})")
    print(f"Wrote {C.SPLIT_DIR / 'global_split_manifest.csv'}")

    is_test_global = np.zeros(len(valid_df), dtype=bool)
    is_test_global[g_test] = True

    # -------------------------------------------------------------------------
    # Per-assay manifests derived from the global split
    # -------------------------------------------------------------------------
    print("\n=== Per-assay manifests ===")
    summary_rows: List[Dict] = []
    for target in C.TARGET_COLUMNS:
        if target not in y_valid.columns:
            continue
        y_raw = y_valid[target].values.astype(float)
        labelled = np.isfinite(y_raw)
        if labelled.sum() == 0:
            print(f"  {target:<10s} no labelled valid molecules; skipped")
            continue

        pos_labelled = np.where(labelled)[0]
        y_sub = y_raw[pos_labelled].astype(int)
        counts = np.bincount(y_sub, minlength=2)
        if counts.min() < 2:
            print(f"  {target:<10s} class counts {counts.tolist()} too small; skipped")
            continue

        sub_df = valid_df.iloc[pos_labelled].reset_index(drop=True)
        test_flag = is_test_global[pos_labelled]

        n_test_pos = int(y_sub[test_flag].sum())
        n_train_pos = int(y_sub[~test_flag].sum())
        mode_used = f"global_{g_mode}"

        if (n_test_pos < MIN_TEST_POS or n_train_pos < MIN_TRAIN_POS
                or test_flag.all() or (~test_flag).all()):
            # Rare label landed badly under the shared split: give this assay
            # its own split so it stays usable, and mark it.
            tr_local, te_local, m = C.make_split(
                sub_df, y_sub, mode=C.SPLIT_MODE, test_size=C.TEST_SIZE, seed=C.SEED
            )
            test_flag = np.zeros(len(y_sub), dtype=bool)
            test_flag[te_local] = True
            mode_used = f"per_assay_{m}"
            n_test_pos = int(y_sub[test_flag].sum())
            n_train_pos = int(y_sub[~test_flag].sum())

        manifest = sub_df.copy()
        manifest["y"] = y_sub
        manifest["split"] = np.where(test_flag, "test", "train")
        manifest["split_mode"] = mode_used
        out_cols = ["dataset_idx", "orig_idx", "SMILES", "canonical_smiles",
                    "scaffold", "y", "split", "split_mode"]
        manifest[out_cols].to_csv(C.SPLIT_DIR / f"{target}_split_manifest.csv", index=False)

        y_tr = y_sub[~test_flag]
        n_folds = max(C.MIN_CV_SPLITS,
                      min(C.CV_SPLITS, int(np.bincount(y_tr, minlength=2).min())))
        summary_rows.append({
            "target": target,
            "n_labelled": int(len(y_sub)),
            "n_pos": int(counts[1]), "n_neg": int(counts[0]),
            "prevalence": float(y_sub.mean()),
            "n_train": int((~test_flag).sum()), "n_test": int(test_flag.sum()),
            "train_pos": n_train_pos, "test_pos": n_test_pos,
            "cv_folds": int(n_folds), "split_mode": mode_used,
            "shares_global_split": bool(mode_used.startswith("global")),
        })
        print(f"  {target:<10s} n={len(y_sub):>4d} pos={int(counts[1]):>3d} "
              f"train={int((~test_flag).sum()):>3d}/test={int(test_flag.sum()):>3d} "
              f"test_pos={n_test_pos:>2d} folds={n_folds} mode={mode_used}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(C.SPLIT_DIR / "split_summary.csv", index=False)
    print(f"\nWrote {C.SPLIT_DIR / 'split_summary.csv'}")

    C.write_json(C.SPLIT_DIR / "prepare_config.json", C.describe_config())

    if not summary.empty:
        tiny = summary[summary["test_pos"] < 5]
        if len(tiny):
            print("\nNOTE: these targets have fewer than 5 positives in the held-out set:")
            for _, r in tiny.iterrows():
                print(f"  {r['target']}: {int(r['test_pos'])} test positives")
            print("Treat their test metrics as a sanity check only; the repeated-CV"
                  " columns are the comparison metric.")
        odd = summary[~summary["shares_global_split"]]
        if len(odd):
            print("\nNOTE: these targets needed their own split and are excluded"
                  " from the paired statistics:")
            print("  " + ", ".join(odd["target"].tolist()))


if __name__ == "__main__":
    main()

# biomimox_predict.py
# Predict biomimetic-oxidation reaction types for new molecules, using either
# the single-task bundles from biomimox_gnn.py or the multitask bundle from
# biomimox_multitask.py (or both).
#
# Inputs (unchanged from the original predict script):
#   predict_new/SMILES_new.csv      column: SMILES
#   predict_new/percepta_new.csv    columns: refractivity, logD, neg, pos
#                                   (optionally SMILES, for merge-by-key)
#
# Model bundles:
#   gnn_models/<assay>/{gnn.pth, meta.pt}       single-task, one per assay
#   gnn_models_multitask/{gnn.pth, meta.pt}     multitask, all heads at once
#
# Outputs (binary_predictions/):
#   predictions_<assay>.csv                 single-task, per assay
#   predictions_singletask_long.csv
#   predictions_multitask_long.csv
#   predictions_all_wide.csv                one row per molecule
#   prediction_manifest.csv                 which inputs were usable
#   prediction_summary.json
#
# Every probability is averaged over the conformer ensemble, matching how the
# models were trained and evaluated. Each row carries the applicability-domain
# distance to the nearest training compound and the OOD flag from that model's
# own cutoff; a flagged prediction should be read as "outside the domain the
# model was fitted on", not as a confident call.
#
# Usage:
#   python biomimox_predict.py
#   BIOMIMOX_PREDICT_ARMS=single python biomimox_predict.py
#   BIOMIMOX_PREDICT_ARMS=multi  python biomimox_predict.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

import biomimox_common as C
import biomimox_graphs as G
import biomimox_models as M

PREDICT_DIR = Path("predict_new")
SMILES_NEW_CSV = PREDICT_DIR / "SMILES_new.csv"
PERCEPTA_NEW_CSV = PREDICT_DIR / "percepta_new.csv"
OUTPUT_DIR = Path("binary_predictions")

SINGLE_MODEL_DIR = Path("gnn_models")
MULTI_MODEL_DIR = Path("gnn_models_multitask")

WHICH = os.environ.get("BIOMIMOX_PREDICT_ARMS", "both").lower()
DEFAULT_BATCH_SIZE = 16


# -----------------------------------------------------------------------------
# Input handling
# -----------------------------------------------------------------------------
def load_new_molecules() -> pd.DataFrame:
    if not SMILES_NEW_CSV.exists():
        raise FileNotFoundError(f"{SMILES_NEW_CSV} not found")
    if not PERCEPTA_NEW_CSV.exists():
        raise FileNotFoundError(f"{PERCEPTA_NEW_CSV} not found")

    df_smi = pd.read_csv(SMILES_NEW_CSV)
    if "SMILES" not in df_smi.columns:
        raise ValueError(f"{SMILES_NEW_CSV} must have a 'SMILES' column header")

    df_p = C.load_percepta_table(SMILES_NEW_CSV, PERCEPTA_NEW_CSV)
    if len(df_smi) != len(df_p):
        raise ValueError("Row count mismatch between the SMILES and Percepta inputs")

    df = pd.DataFrame({
        "orig_idx": np.arange(len(df_smi), dtype=int),
        "SMILES": df_smi["SMILES"].astype(str).tolist(),
    })
    df["canonical_smiles"] = [C.canonicalize_smiles(s) or s for s in df["SMILES"]]
    df["scaffold"] = [C.murcko_scaffold(s) for s in df["SMILES"]]
    for c in C.PERCEPTA_COLS:
        df[c] = df_p[c].astype(float).values
    return df


# -----------------------------------------------------------------------------
# Bundle loading
# -----------------------------------------------------------------------------
def load_bundle(bundle_dir: Path, device: torch.device) -> Tuple[Dict, M.BiomimoxNet]:
    meta_path = bundle_dir / "meta.pt"
    weights_path = bundle_dir / "gnn.pth"
    if not meta_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Missing meta.pt or gnn.pth in {bundle_dir}")

    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    cfg = dict(meta.get("best_config", {}))
    g_mean = meta.get("G_mean")
    g_std = meta.get("G_std")
    if g_mean is None or g_std is None:
        raise ValueError(f"Missing G_mean/G_std in {meta_path}")

    model = M.BiomimoxNet(
        global_in=int(g_mean.numel()),
        n_tasks=int(meta.get("n_tasks", 1)),
        backbone=str(meta.get("backbone", "dimenet++")),
        hidden=int(cfg.get("hidden", 64)),
        activation=str(cfg.get("activation", "relu")),
        dropout=float(cfg.get("dropout", 0.1)),
        cutoff=float(meta.get("cutoff", 5.0)),
        max_neighbors=int(meta.get("max_neighbors", 32)),
        dpp_blocks=int(meta.get("dpp_blocks", 4)),
        schnet_interactions=int(meta.get("schnet_interactions", 4)),
        use_2d_branch=bool(meta.get("use_2d_branch", True)),
        gine_layers=int(cfg.get("gine_layers", 3)),
    ).to(device)

    state = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return meta, model


@torch.no_grad()
def predict_bundle(model: M.BiomimoxNet,
                   graphs: List[List],
                   meta: Dict,
                   batch_size: int,
                   device: torch.device) -> np.ndarray:
    """Conformer-averaged probabilities, shape [n_mol, n_tasks]."""
    g_mean = meta["G_mean"].view(-1).cpu()
    g_std = meta["G_std"].view(-1).cpu()
    g_std = torch.where(g_std < 1e-8, torch.ones_like(g_std), g_std)

    mol_indices = np.arange(len(graphs), dtype=int)
    ds = G.MolGraphDataset(graphs, mol_indices, y=None, g_mean=g_mean, g_std=g_std,
                           conf_mode="all", seed=C.SEED)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False)

    probs: List[np.ndarray] = []
    positions: List[int] = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        positions.extend(batch.mol_pos.view(-1).cpu().numpy().tolist())

    n_tasks = int(meta.get("n_tasks", 1))
    if not probs:
        return np.full((len(graphs), n_tasks), np.nan)
    P = np.vstack(probs)
    uniq, mean_p = G.average_over_conformers_2d(np.asarray(positions), P)
    out = np.full((len(graphs), P.shape[1]), np.nan, dtype=float)
    out[uniq] = mean_p
    return out


def ood_for_bundle(meta: Dict, query_smiles: Sequence[str]) -> pd.DataFrame:
    return C.score_ood_smiles(
        query_smiles=list(query_smiles),
        ref_smiles=list(meta.get("ood_reference_smiles", [])),
        cutoff=float(meta.get("ood_nn_distance_cutoff", float("nan"))),
        radius=int(meta.get("ood_morgan_radius", C.MORGAN_RADIUS)),
        nbits=int(meta.get("ood_morgan_nbits", C.MORGAN_NBITS)),
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    C.set_workdir()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    C.set_seeds(C.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | arms: {WHICH}")

    input_df = load_new_molecules()
    print(f"\n{len(input_df)} input molecules")

    # Conformers for the new molecules, with the same settings as training.
    n_conf = C.N_CONFORMERS
    for probe in (SINGLE_MODEL_DIR, MULTI_MODEL_DIR):
        meta_files = sorted(probe.glob("**/meta.pt"))
        if meta_files:
            try:
                m = torch.load(meta_files[0], map_location="cpu", weights_only=False)
                n_conf = int(m.get("n_conformers", C.N_CONFORMERS))
            except Exception:
                pass
            break
    print(f"Building conformers (n_conformers={n_conf})")

    graphs, valid_mask = G.build_graph_dataset(
        input_df, n_conformers=n_conf, seed=C.SEED, use_cache=True,
        cache_dir=C.CACHE_DIR, verbose=True,
    )
    kept = input_df.loc[valid_mask].reset_index(drop=True).copy()
    kept["dataset_idx"] = np.arange(len(kept), dtype=int)

    manifest = input_df[["orig_idx", "SMILES", "canonical_smiles"]].copy()
    manifest["valid_for_gnn"] = valid_mask.astype(int)
    manifest.to_csv(OUTPUT_DIR / "prediction_manifest.csv", index=False)
    print(f"{int(valid_mask.sum())}/{len(input_df)} molecules embedded; "
          f"wrote {OUTPUT_DIR / 'prediction_manifest.csv'}")

    if len(graphs) == 0:
        raise RuntimeError("No input molecule could be embedded into a 3-D conformer")

    query_smiles = kept["canonical_smiles"].astype(str).tolist()
    wide = kept[["orig_idx", "SMILES", "canonical_smiles"]].copy()
    long_single: List[pd.DataFrame] = []
    long_multi: List[pd.DataFrame] = []

    # ---- single-task bundles ----
    if WHICH in ("both", "single") and SINGLE_MODEL_DIR.exists():
        assay_dirs = sorted([p for p in SINGLE_MODEL_DIR.iterdir()
                             if p.is_dir() and (p / "meta.pt").exists()
                             and (p / "gnn.pth").exists()])
        print(f"\n=== Single-task bundles: {len(assay_dirs)} found ===")
        for assay_dir in assay_dirs:
            assay = assay_dir.name
            try:
                meta, model = load_bundle(assay_dir, device)
            except Exception as exc:
                print(f"  {assay}: could not load ({exc}); skipped")
                continue

            bs = int(meta.get("best_config", {}).get("batch_size", DEFAULT_BATCH_SIZE))
            P = predict_bundle(model, graphs, meta, bs, device)
            prob = P[:, 0]
            th = float(meta.get("threshold", 0.5))
            ood = ood_for_bundle(meta, query_smiles)

            frame = kept[["orig_idx", "SMILES", "canonical_smiles"]].copy()
            frame["assay"] = assay
            frame["prob"] = prob
            frame["pred"] = (prob >= th).astype(int)
            frame["threshold"] = th
            frame = pd.concat([frame.reset_index(drop=True),
                               ood.reset_index(drop=True)], axis=1)
            frame.to_csv(OUTPUT_DIR / f"predictions_{assay}.csv", index=False)
            long_single.append(frame)

            wide[f"prob_{assay}"] = prob
            wide[f"pred_{assay}"] = frame["pred"].values
            wide[f"ood_flag_{assay}"] = frame["ood_flag"].values
            print(f"  {assay:<10s} mean prob={np.nanmean(prob):.3f} "
                  f"positives={int(frame['pred'].sum())}/{len(frame)} "
                  f"OOD-flagged={int(frame['ood_flag'].sum())}")

    # ---- multitask bundle ----
    if WHICH in ("both", "multi") and (MULTI_MODEL_DIR / "meta.pt").exists():
        print("\n=== Multitask bundle ===")
        meta, model = load_bundle(MULTI_MODEL_DIR, device)
        task_names = list(meta.get("task_names", []))
        thresholds = list(meta.get("thresholds", [0.5] * len(task_names)))
        bs = int(meta.get("best_config", {}).get("batch_size", DEFAULT_BATCH_SIZE))
        P = predict_bundle(model, graphs, meta, bs, device)
        ood = ood_for_bundle(meta, query_smiles)

        for t, name in enumerate(task_names):
            prob = P[:, t]
            th = float(thresholds[t]) if t < len(thresholds) else 0.5
            frame = kept[["orig_idx", "SMILES", "canonical_smiles"]].copy()
            frame["task"] = name
            frame["prob"] = prob
            frame["pred"] = (prob >= th).astype(int)
            frame["threshold"] = th
            frame = pd.concat([frame.reset_index(drop=True),
                               ood.reset_index(drop=True)], axis=1)
            long_multi.append(frame)
            wide[f"mt_prob_{name}"] = prob
            wide[f"mt_pred_{name}"] = frame["pred"].values
            print(f"  {name:<10s} mean prob={np.nanmean(prob):.3f} "
                  f"positives={int(frame['pred'].sum())}/{len(frame)}")
        wide["mt_ood_flag"] = ood["ood_flag"].values
        wide["mt_ood_nn_distance"] = ood["ood_nn_distance"].values

    if long_single:
        pd.concat(long_single, ignore_index=True).to_csv(
            OUTPUT_DIR / "predictions_singletask_long.csv", index=False)
    if long_multi:
        pd.concat(long_multi, ignore_index=True).to_csv(
            OUTPUT_DIR / "predictions_multitask_long.csv", index=False)
    if not long_single and not long_multi:
        raise RuntimeError(
            "No model bundles found. Train with biomimox_gnn.py and/or "
            "biomimox_multitask.py first."
        )

    wide.to_csv(OUTPUT_DIR / "predictions_all_wide.csv", index=False)
    C.write_json(OUTPUT_DIR / "prediction_summary.json", {
        "n_input_rows": int(len(input_df)),
        "n_valid_molecules": int(len(kept)),
        "n_conformers": int(n_conf),
        "arms": WHICH,
        "single_task_assays": sorted([p.name for p in SINGLE_MODEL_DIR.iterdir()
                                      if p.is_dir() and (p / "meta.pt").exists()])
        if SINGLE_MODEL_DIR.exists() else [],
        "multitask": bool((MULTI_MODEL_DIR / "meta.pt").exists()),
        "smiles_csv": str(SMILES_NEW_CSV),
        "percepta_csv": str(PERCEPTA_NEW_CSV),
    })
    print(f"\nWrote predictions under {OUTPUT_DIR}")
    print("Rows with ood_flag=1 fall outside the training domain of that model;"
          " treat those probabilities as unreliable rather than as calls.")


if __name__ == "__main__":
    main()

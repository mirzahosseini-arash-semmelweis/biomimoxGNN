# biomimox_multitask.py
# Ablation arm 6: one shared encoder, one head per reaction type.
#
# Implementation notes:
#   * masked BCE -- a molecule with no label for S_ox contributes nothing to the
#     S_ox head rather than a spurious zero
#   * per-head pos_weight, computed on fold-training labels only
#   * per-head decision thresholds, each chosen on out-of-fold predictions
#   * `percent` is deliberately excluded: it is overall turnover, not one
#     reaction type, and mixing it in muddies the interpretation of the shared
#     representation. Set BIOMIMOX_MT_INCLUDE_PERCENT=1 to include it anyway.
#   * results are reported PER TASK, never as a single average. With 11 heads
#     and ~100 compounds a better mean can easily hide two tasks getting worse;
#     negative transfer is the thing to look for, and biomimox_compare.py counts
#     it explicitly.
#
# Backbone selection is the same as the single-task script:
#   BIOMIMOX_BACKBONE=dimenet++ (default) | schnet | none
#
# Outputs:
#   ablation_results/multitask_{cv,test}_all.csv, multitask_summary.csv
#   gnn_models_multitask/{gnn.pth, meta.pt, summary.json, *.svg, predictions.csv}
#
# Usage:
#   python biomimox_prepare.py && python biomimox_multitask.py

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

import biomimox_common as C
import biomimox_graphs as G
import biomimox_models as M

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BACKBONE = os.environ.get("BIOMIMOX_BACKBONE", "dimenet++")
USE_2D_BRANCH = os.environ.get("BIOMIMOX_USE_2D", "1") not in ("0", "false", "False")
LOSS_NAME = os.environ.get("BIOMIMOX_LOSS", "bce")
INCLUDE_PERCENT = os.environ.get("BIOMIMOX_MT_INCLUDE_PERCENT", "0") in ("1", "true", "True")

CUTOFF = float(os.environ.get("BIOMIMOX_CUTOFF", "5.0"))
MAX_NEIGHBORS = int(os.environ.get("BIOMIMOX_MAX_NEIGHBORS", "32"))
DPP_BLOCKS = 4
SCHNET_INTERACTIONS = 4

MAX_EPOCHS = int(os.environ.get("BIOMIMOX_EPOCHS", "150"))
EARLY_STOPPING_PATIENCE = 25
EARLY_STOPPING_MIN_DELTA = 1e-4
GRAD_CLIP = 5.0

MODEL_DIR = Path("gnn_models_multitask")
SELECT_REPEAT = 0

_BASE_GRID: List[Dict[str, object]] = [
    {"hidden": 64,  "batch_size": 16, "lr": 1e-3, "weight_decay": 1e-5, "dropout": 0.10, "activation": "relu", "gine_layers": 3},
    {"hidden": 128, "batch_size": 16, "lr": 1e-3, "weight_decay": 1e-5, "dropout": 0.20, "activation": "relu", "gine_layers": 3},
    {"hidden": 64,  "batch_size": 8,  "lr": 5e-4, "weight_decay": 1e-4, "dropout": 0.20, "activation": "relu", "gine_layers": 4},
    {"hidden": 128, "batch_size": 32, "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.30, "activation": "relu", "gine_layers": 3},
]


def build_config_grid(backbone: str) -> List[Dict[str, object]]:
    """A shared encoder can justify more capacity than a single-task model, so
    hidden=128 is kept here even for DimeNet++ -- but only one such config, and
    with the larger batch, so the run stays affordable. See the note in
    biomimox_models.build_dimenetpp on why hidden no longer drags the
    per-triplet embedding sizes up with it."""
    grid = list(_BASE_GRID)
    if str(backbone).lower().startswith("dimenet"):
        grid = [grid[0], grid[1], grid[2]]
    if os.environ.get("BIOMIMOX_FULL_GRID", "0") in ("1", "true", "True"):
        grid = list(_BASE_GRID)
    max_cfg = os.environ.get("BIOMIMOX_MAX_CONFIGS")
    if max_cfg:
        grid = grid[: int(max_cfg)]
    return grid


CONFIG_GRID: List[Dict[str, object]] = build_config_grid(BACKBONE)

ARM_ID = "6_multitask_" + {"dimenet++": "dimenetpp", "schnet": "schnet",
                           "none": "gine2d"}.get(BACKBONE, BACKBONE)
ARM_LABEL = "6  multitask " + {"dimenet++": "DimeNet++", "schnet": "SchNet",
                               "none": "GINE (2-D only)"}.get(BACKBONE, BACKBONE)


# -----------------------------------------------------------------------------
# Data assembly
# -----------------------------------------------------------------------------
def load_multitask_frame() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """Returns (manifest, Y, mask, task_names) over the globally split molecules."""
    gm_path = C.SPLIT_DIR / "global_split_manifest.csv"
    labels_path = C.SPLIT_DIR / "labels_valid.csv"
    if not gm_path.exists() or not labels_path.exists():
        raise RuntimeError("Run biomimox_prepare.py first to write splits/")

    manifest = pd.read_csv(gm_path)
    labels = pd.read_csv(labels_path)
    labels = labels.set_index("dataset_idx").loc[manifest["dataset_idx"].values]

    task_names = [t for t in C.MULTITASK_TARGETS if t in labels.columns]
    if INCLUDE_PERCENT and "percent" in labels.columns:
        task_names = ["percent"] + task_names

    Y = labels[task_names].values.astype(float)
    mask = np.isfinite(Y).astype(float)

    # Drop any task with too few positives to be learnable at all.
    keep: List[int] = []
    for t in range(Y.shape[1]):
        col = Y[:, t]
        col = col[np.isfinite(col)]
        if col.size >= 10 and int((col == 1).sum()) >= 3 and int((col == 0).sum()) >= 3:
            keep.append(t)
        else:
            print(f"  dropping task '{task_names[t]}': "
                  f"{int((col == 1).sum())} pos / {int((col == 0).sum())} neg")
    task_names = [task_names[i] for i in keep]
    return manifest, Y[:, keep], mask[:, keep], task_names


def build_model(config: Dict[str, object],
                global_in: int,
                n_tasks: int,
                device: torch.device) -> M.BiomimoxNet:
    return M.BiomimoxNet(
        global_in=global_in, n_tasks=n_tasks, backbone=BACKBONE,
        hidden=int(config["hidden"]), activation=str(config["activation"]),
        dropout=float(config["dropout"]), cutoff=CUTOFF,
        max_neighbors=MAX_NEIGHBORS, dpp_blocks=DPP_BLOCKS,
        schnet_interactions=SCHNET_INTERACTIONS,
        use_2d_branch=USE_2D_BRANCH, gine_layers=int(config.get("gine_layers", 3)),
    ).to(device)


# -----------------------------------------------------------------------------
# Train / predict
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, batch.y, getattr(batch, "y_mask", None))
        if not torch.isfinite(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total += float(loss.detach()) * batch.num_graphs
        n += batch.num_graphs
    return total / max(n, 1)


@torch.no_grad()
def predict_multitask(model,
                      graphs,
                      mol_indices: Sequence[int],
                      Y: np.ndarray,
                      mask: np.ndarray,
                      g_mean: torch.Tensor,
                      g_std: torch.Tensor,
                      batch_size: int,
                      device: torch.device,
                      conf_mode: str = "all") -> np.ndarray:
    """Returns a [n_mol, n_tasks] probability matrix, conformer-averaged."""
    model.eval()
    ds = G.MolGraphDataset(graphs, mol_indices, y=Y, y_mask=mask,
                           g_mean=g_mean, g_std=g_std, conf_mode=conf_mode, seed=C.SEED)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False)

    probs: List[np.ndarray] = []
    positions: List[int] = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        positions.extend(batch.mol_pos.view(-1).cpu().numpy().tolist())

    if not probs:
        return np.full((len(mol_indices), Y.shape[1]), np.nan)

    P = np.vstack(probs)
    uniq, mean_p = G.average_over_conformers_2d(np.asarray(positions), P)
    out = np.full((len(mol_indices), Y.shape[1]), np.nan, dtype=float)
    out[uniq] = mean_p
    return out


def macro_pr_auc(Y: np.ndarray, P: np.ndarray, mask: np.ndarray) -> float:
    vals = []
    for t in range(Y.shape[1]):
        m = mask[:, t] > 0
        if m.sum() < 4:
            continue
        met = C.ranking_metrics(Y[m, t], P[m, t])
        if np.isfinite(met["pr_auc"]):
            vals.append(met["pr_auc"])
    return float(np.mean(vals)) if vals else float("nan")


def fit_fold(graphs,
             train_mols: Sequence[int],
             val_mols: Sequence[int],
             Y_tr: np.ndarray, M_tr: np.ndarray,
             Y_va: np.ndarray, M_va: np.ndarray,
             config: Dict[str, object],
             device: torch.device,
             seed: int,
             verbose_prefix: str = "",
             fixed_epochs: Optional[int] = None) -> Dict[str, object]:
    """fixed_epochs=None early-stops on the validation fold and is used only
    for configuration selection. fixed_epochs=N trains exactly N epochs without
    consulting the validation fold, which is what the reported evaluation
    repeats use; otherwise the out-of-fold metrics are inflated by a stopping
    rule tuned on the very fold being scored."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    g_mean, g_std = G.global_scaler(graphs, train_mols)
    n_tasks = Y_tr.shape[1]

    train_ds = G.MolGraphDataset(graphs, train_mols, y=Y_tr, y_mask=M_tr,
                                 g_mean=g_mean, g_std=g_std,
                                 conf_mode="random", seed=seed)
    # No class sampler here: with 11 heads there is no single class to balance,
    # so imbalance is handled per head by pos_weight inside the masked loss.
    train_loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True)

    model = build_model(config, G.global_feature_dim(graphs), n_tasks, device)
    criterion = M.make_loss(LOSS_NAME, np.where(M_tr > 0, Y_tr, np.nan), n_tasks, device)
    optimizer = AdamW(model.parameters(), lr=float(config["lr"]),
                      weight_decay=float(config["weight_decay"]))

    early_stop = fixed_epochs is None
    n_epochs = MAX_EPOCHS if early_stop else int(fixed_epochs)

    best_metric, best_epoch, best_state = -np.inf, n_epochs, None
    patience = 0
    losses: List[float] = []
    macro_curve: List[float] = []

    for epoch in range(1, n_epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        losses.append(loss)

        if early_stop:
            P_va = predict_multitask(model, graphs, val_mols, Y_va, M_va, g_mean, g_std,
                                     int(config["batch_size"]), device, conf_mode="first")
            macro = macro_pr_auc(Y_va, P_va, M_va)
            macro_curve.append(macro)

            if np.isfinite(macro) and macro > best_metric + EARLY_STOPPING_MIN_DELTA:
                best_metric, best_epoch = macro, epoch
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

        if verbose_prefix and (epoch == 1 or epoch % 25 == 0):
            extra = f" macroPR={macro_curve[-1]:.3f}" if early_stop else ""
            print(f"{verbose_prefix} epoch {epoch:03d} loss={loss:.4f}{extra}")

        if early_stop and patience >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    P_final = predict_multitask(model, graphs, val_mols, Y_va, M_va, g_mean, g_std,
                                int(config["batch_size"]), device, conf_mode="all")
    return {
        "best_epoch": int(best_epoch), "val_probs": P_final,
        "train_losses": losses, "macro_curve": macro_curve,
        "macro_pr_auc": macro_pr_auc(Y_va, P_final, M_va),
        "n_params": M.count_parameters(model),
    }


def run_repeat(graphs,
               pool_mols: np.ndarray,
               Y_pool: np.ndarray, M_pool: np.ndarray,
               groups_pool: Sequence[str],
               strat_pool: np.ndarray,
               config: Dict[str, object],
               repeat: int,
               device: torch.device,
               verbose: bool = False,
               timer: Optional[Dict[str, float]] = None,
               fixed_epochs: Optional[int] = None) -> Dict[str, object]:
    folds = [f for f in C.repeated_cv_indices(
        strat_pool, groups=groups_pool, n_splits=C.CV_SPLITS,
        n_repeats=repeat + 1, seed=C.SEED, mode=C.SPLIT_MODE) if f[0] == repeat]

    oof = np.full_like(Y_pool, np.nan, dtype=float)
    best_epochs: List[int] = []
    curves: Dict[str, List[List[float]]] = {"train": [], "macro": []}
    n_params = 0

    pass_start = time.time()
    for _, fold_i, tr, va in folds:
        t0 = time.time()
        res = fit_fold(graphs, pool_mols[tr], pool_mols[va],
                       Y_pool[tr], M_pool[tr], Y_pool[va], M_pool[va],
                       config, device, seed=C.SEED + 1000 * repeat + fold_i,
                       verbose_prefix=(f"  r{repeat}f{fold_i} h={config['hidden']}"
                                       if verbose else ""),
                       fixed_epochs=fixed_epochs)
        dt = time.time() - t0
        if timer is not None and timer.get("per_fold") is None:
            timer["per_fold"] = dt
        print(f"    fold {fold_i} done in {dt:6.1f}s | "
              f"{len(res['train_losses']):>3d} epochs | "
              f"fold macro PR-AUC={res['macro_pr_auc']:.3f}", flush=True)
        oof[va] = res["val_probs"]
        best_epochs.append(res["best_epoch"])
        curves["train"].append(res["train_losses"])
        curves["macro"].append(res["macro_curve"])
        n_params = res["n_params"]

    return {
        "repeat": repeat, "oof": oof,
        "epoch_rule": int(np.clip(round(float(np.median(best_epochs))) if best_epochs else 20,
                                  1, MAX_EPOCHS)),
        "curves": curves, "n_params": n_params,
        "pass_seconds": time.time() - pass_start,
    }


def per_task_metrics(Y: np.ndarray, P: np.ndarray, mask: np.ndarray,
                     task_names: Sequence[str],
                     thresholds: Optional[Sequence[float]] = None,
                     stage: str = "cv_oof",
                     repeat: int = -1) -> Tuple[List[Dict], List[float]]:
    rows: List[Dict] = []
    ths: List[float] = []
    for t, name in enumerate(task_names):
        m = mask[:, t] > 0
        y_t, p_t = Y[m, t], P[m, t]
        th = (C.optimal_threshold(y_t, p_t) if thresholds is None
              else float(thresholds[t]))
        ths.append(th)
        met = C.threshold_metrics(y_t, p_t, th)
        met.update({"target": name, "arm": ARM_ID, "arm_label": ARM_LABEL,
                    "repeat": repeat, "stage": stage})
        rows.append(met)
    return rows, ths


def fit_final(graphs, pool_mols, Y_pool, M_pool, config, epochs, device):
    torch.manual_seed(C.SEED)
    g_mean, g_std = G.global_scaler(graphs, pool_mols)
    ds = G.MolGraphDataset(graphs, pool_mols, y=Y_pool, y_mask=M_pool,
                           g_mean=g_mean, g_std=g_std, conf_mode="random", seed=C.SEED)
    loader = DataLoader(ds, batch_size=int(config["batch_size"]), shuffle=True)

    model = build_model(config, G.global_feature_dim(graphs), Y_pool.shape[1], device)
    criterion = M.make_loss(LOSS_NAME, np.where(M_pool > 0, Y_pool, np.nan),
                            Y_pool.shape[1], device)
    optimizer = AdamW(model.parameters(), lr=float(config["lr"]),
                      weight_decay=float(config["weight_decay"]))
    losses = []
    for epoch in range(1, int(epochs) + 1):
        losses.append(train_one_epoch(model, loader, criterion, optimizer, device))
        if epoch == 1 or epoch % 25 == 0 or epoch == int(epochs):
            print(f"  final epoch {epoch:03d}/{int(epochs):03d} loss={losses[-1]:.4f}")
    return {"model": model, "g_mean": g_mean, "g_std": g_std, "train_losses": losses}


def plot_curves(curves: Dict[str, List[List[float]]], out_dir: Path) -> None:
    for key, ylabel, fname in (("train", "Training loss", "cv_loss"),
                               ("macro", "Macro PR-AUC", "cv_macro_pr")):
        arrs = curves.get(key, [])
        if not arrs:
            continue
        n = max(len(a) for a in arrs)
        mat = np.full((len(arrs), n), np.nan)
        for i, a in enumerate(arrs):
            mat[i, : len(a)] = a
        plt.figure(figsize=(5, 4), dpi=300)
        for row in mat:
            plt.plot(np.arange(1, n + 1), row, alpha=0.25)
        plt.plot(np.arange(1, n + 1), np.nanmean(mat, axis=0), linewidth=2,
                 label=f"{ylabel} (mean)")
        plt.xlabel("Epoch"); plt.ylabel(ylabel)
        if key == "macro":
            plt.ylim(0.0, 1.0)
        plt.legend(frameon=False); plt.tight_layout()
        plt.savefig(out_dir / f"multitask_{fname}.svg", format="svg"); plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    C.set_workdir()
    C.ensure_dirs()
    C.set_seeds(C.SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | backbone={BACKBONE} | 2-D branch={USE_2D_BRANCH} "
          f"| loss={LOSS_NAME} | arm={ARM_ID}")

    print("\n=== Assembling multitask labels ===")
    manifest, Y, mask, task_names = load_multitask_frame()
    print(f"{len(manifest)} molecules, {len(task_names)} tasks: {', '.join(task_names)}")
    per_task_pos = [(n, int(np.nansum(Y[mask[:, t] > 0, t] == 1)))
                    for t, n in enumerate(task_names)]
    print("positives per task: " + ", ".join(f"{n}={k}" for n, k in per_task_pos))

    mol_df = C.load_molecule_table()
    graphs, _ = G.build_graph_dataset(mol_df, n_conformers=C.N_CONFORMERS,
                                      seed=C.SEED, use_cache=True, verbose=True)

    mol_idx = manifest["dataset_idx"].values.astype(int)
    groups = manifest["scaffold"].astype(str).tolist()
    strat = manifest["n_pos_bin"].values.astype(int)
    is_train = manifest["split"].values == "train"

    pool_mols, test_mols = mol_idx[is_train], mol_idx[~is_train]
    Y_pool, M_pool = Y[is_train], mask[is_train]
    Y_test, M_test = Y[~is_train], mask[~is_train]
    groups_pool = [g for g, t in zip(groups, is_train) if t]
    strat_pool = strat[is_train]

    print(f"train {len(pool_mols)} / test {len(test_mols)}")

    # ---- configuration selection ----
    timer: Dict[str, object] = {"per_fold": None}
    pass_times: List[float] = []
    select: List[Tuple[Dict, float, Dict]] = []
    for cfg_i, config in enumerate(CONFIG_GRID, start=1):
        print(f"\n-- config {cfg_i}/{len(CONFIG_GRID)}: {config}", flush=True)
        res = run_repeat(graphs, pool_mols, Y_pool, M_pool, groups_pool, strat_pool,
                         config, SELECT_REPEAT, device, verbose=(cfg_i == 1), timer=timer)
        macro = macro_pr_auc(Y_pool, res["oof"], M_pool)
        print(f"   selection macro PR-AUC={macro:.3f} epochs~{res['epoch_rule']} "
              f"params={res['n_params']:,}", flush=True)
        select.append((config, macro, res))
        pass_times.append(float(res["pass_seconds"]))

        passes_total = len(CONFIG_GRID) + max(C.CV_REPEATS - 1, 1)
        mean_pass = float(np.mean(pass_times))
        print(f"   [timing] this CV pass {res['pass_seconds'] / 60.0:.1f} min | "
              f"mean {mean_pass / 60.0:.1f} min | {len(pass_times)}/{passes_total} "
              f"passes done | ~{mean_pass * max(passes_total - len(pass_times), 0) / 60.0:.1f} "
              f"min remaining", flush=True)

    best_config, best_macro, best_select = max(
        select, key=lambda t: (-np.inf if not np.isfinite(t[1]) else t[1]))
    select_epochs = int(best_select["epoch_rule"])
    print(f"\nSelected config: {best_config} (macro PR-AUC {best_macro:.3f}, "
          f"epoch budget {select_epochs})")
    pd.DataFrame([{**cfg, "select_macro_pr_auc": m} for cfg, m, _ in select]).to_csv(
        MODEL_DIR / "config_selection.csv", index=False)

    # ---- reported evaluation ----
    eval_repeats = [r for r in range(C.CV_REPEATS) if r != SELECT_REPEAT] or [SELECT_REPEAT]
    if len(eval_repeats) < 2:
        print("[warning] fewer than 2 evaluation repeats: the SD column will be "
              "empty. Raise BIOMIMOX_REPEATS (default 5) to get a spread.")
    cv_rows: List[Dict] = []
    epoch_rules: List[int] = []
    threshold_sets: List[List[float]] = []
    curves_for_plot: Optional[Dict[str, List[List[float]]]] = None

    for rep in eval_repeats:
        res = run_repeat(graphs, pool_mols, Y_pool, M_pool, groups_pool, strat_pool,
                         best_config, rep, device, verbose=False, timer=timer,
                         fixed_epochs=select_epochs)
        pass_times.append(float(res["pass_seconds"]))
        rows, ths = per_task_metrics(Y_pool, res["oof"], M_pool, task_names,
                                     thresholds=None, stage="cv_oof", repeat=rep)
        cv_rows.extend(rows)
        epoch_rules.append(res["epoch_rule"])
        threshold_sets.append(ths)
        if curves_for_plot is None:
            curves_for_plot = res["curves"]
        macro = macro_pr_auc(Y_pool, res["oof"], M_pool)
        print(f"  repeat {rep}: macro PR-AUC={macro:.3f}")
        pd.DataFrame(res["oof"], columns=task_names).assign(
            repeat=rep, dataset_idx=pool_mols).to_csv(
            C.RESULTS_DIR / f"multitask_oof_repeat{rep}.csv", index=False)

    plot_curves(best_select["curves"], MODEL_DIR)

    thresholds = np.median(np.asarray(threshold_sets, dtype=float), axis=0).tolist()
    epoch_rule = int(np.clip(round(float(np.median(epoch_rules)))
                             if epoch_rules else select_epochs, 1, MAX_EPOCHS))

    # ---- final refit and held-out evaluation ----
    print(f"\nRefitting on the full training pool: {epoch_rule} epochs")
    final = fit_final(graphs, pool_mols, Y_pool, M_pool, best_config, epoch_rule, device)
    model = final["model"]

    P_test = predict_multitask(model, graphs, test_mols, Y_test, M_test,
                               final["g_mean"], final["g_std"],
                               int(best_config["batch_size"]), device, "all")
    P_train = predict_multitask(model, graphs, pool_mols, Y_pool, M_pool,
                                final["g_mean"], final["g_std"],
                                int(best_config["batch_size"]), device, "all")

    test_rows, _ = per_task_metrics(Y_test, P_test, M_test, task_names,
                                    thresholds=thresholds, stage="test", repeat=-1)

    # Predictions in long form, with applicability-domain distance.
    train_smiles = manifest.loc[is_train, "canonical_smiles"].astype(str).tolist()
    test_smiles = manifest.loc[~is_train, "canonical_smiles"].astype(str).tolist()
    ood_ref = C.build_ood_reference(train_smiles, quantile=C.OOD_DISTANCE_QUANTILE)
    ood_test = C.score_ood_smiles(test_smiles, ood_ref["reference_smiles"], ood_ref["cutoff"])

    long_rows: List[Dict] = []
    for split_name, mols, smis, Yv, Mv, Pv in (
        ("train", pool_mols, train_smiles, Y_pool, M_pool, P_train),
        ("test", test_mols, test_smiles, Y_test, M_test, P_test),
    ):
        for i, (ds_idx, smi) in enumerate(zip(mols, smis)):
            for t, name in enumerate(task_names):
                long_rows.append({
                    "dataset_idx": int(ds_idx), "SMILES": smi, "split": split_name,
                    "task": name,
                    "y": (float(Yv[i, t]) if Mv[i, t] > 0 else np.nan),
                    "prob": float(Pv[i, t]),
                    "pred": int(Pv[i, t] >= thresholds[t]) if np.isfinite(Pv[i, t]) else -1,
                    "threshold": float(thresholds[t]),
                    "ood_nn_distance": (float(ood_test["ood_nn_distance"].iloc[i])
                                        if split_name == "test" else np.nan),
                    "ood_flag": (int(ood_test["ood_flag"].iloc[i])
                                 if split_name == "test" else -1),
                })
    pd.DataFrame(long_rows).to_csv(MODEL_DIR / "multitask_predictions_long.csv", index=False)

    # ---- summaries ----
    cv_df = pd.DataFrame(cv_rows)
    test_df = pd.DataFrame(test_rows)
    cv_df.to_csv(C.RESULTS_DIR / "multitask_cv_all.csv", index=False)
    test_df.to_csv(C.RESULTS_DIR / "multitask_test_all.csv", index=False)

    summary_rows: List[Dict] = []
    for name in task_names:
        cv_sub = cv_df[cv_df["target"] == name].to_dict("records")
        agg = C.aggregate_repeat_metrics(cv_sub)
        te = test_df[test_df["target"] == name]
        summary_rows.append({
            "target": name, "arm": ARM_ID, "arm_label": ARM_LABEL,
            "family": "gnn_multitask", "backbone": BACKBONE,
            "use_2d_branch": bool(USE_2D_BRANCH),
            "n_tasks": len(task_names),
            "n_train": int(len(pool_mols)), "n_test": int(len(test_mols)),
            "prevalence": float(np.nanmean(Y[mask[:, task_names.index(name)] > 0,
                                             task_names.index(name)])),
            **{f"cv_{k}": v for k, v in agg.items()},
            **{f"test_{k}": (float(te[k].iloc[0]) if len(te) and k in te.columns else np.nan)
               for k in ("roc_auc", "pr_auc", "f1", "mcc", "balanced_accuracy", "Se", "Sp")},
            **{f"cfg_{k}": v for k, v in best_config.items()},
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(C.RESULTS_DIR / "multitask_summary.csv", index=False)

    torch.save(model.state_dict(), MODEL_DIR / "gnn.pth")
    torch.save({
        "arm": ARM_ID, "model_type": "BiomimoxNet_multitask",
        "backbone": BACKBONE, "use_2d_branch": bool(USE_2D_BRANCH),
        "n_tasks": len(task_names), "task_names": task_names,
        "best_config": best_config, "epoch_rule": epoch_rule,
        "thresholds": [float(t) for t in thresholds],
        "threshold": float(np.median(thresholds)),
        "g_feature_names": C.GLOBAL_FEATURE_NAMES,
        "G_mean": final["g_mean"].cpu(), "G_std": final["g_std"].cpu(),
        "cutoff": float(CUTOFF), "max_neighbors": int(MAX_NEIGHBORS),
        "dpp_blocks": int(DPP_BLOCKS), "schnet_interactions": int(SCHNET_INTERACTIONS),
        "n_conformers": int(C.N_CONFORMERS), "loss_name": LOSS_NAME,
        "seed": int(C.SEED), "cv_splits": int(C.CV_SPLITS),
        "cv_repeats": int(C.CV_REPEATS),
        "split_mode": str(manifest["split_mode"].iloc[0]),
        "ood_method": "morgan_nn_distance",
        "ood_reference_smiles": ood_ref["reference_smiles"],
        "ood_nn_distance_cutoff": float(ood_ref["cutoff"]),
        "ood_quantile": float(ood_ref["quantile"]),
        "ood_morgan_radius": int(ood_ref["radius"]),
        "ood_morgan_nbits": int(ood_ref["nbits"]),
        "train_dataset_idx": pool_mols.tolist(),
        "test_dataset_idx": test_mols.tolist(),
    }, MODEL_DIR / "meta.pt")

    C.write_json(MODEL_DIR / "summary.json", {
        "arm": ARM_ID, "task_names": task_names, "best_config": best_config,
        "epoch_rule": epoch_rule, "thresholds": thresholds,
        "cv_macro_pr_auc": float(np.nanmean(summary["cv_pr_auc_mean"].values)),
    })

    print(f"\nWrote {C.RESULTS_DIR / 'multitask_summary.csv'} and {MODEL_DIR}")
    print(summary[["target", "cv_pr_auc_mean", "cv_pr_auc_sd",
                   "cv_roc_auc_mean", "test_pr_auc"]].to_string(index=False))
    print("\nRun biomimox_compare.py to put this next to the single-task and"
          " baseline arms and to count negative transfer per task.")


if __name__ == "__main__":
    main()

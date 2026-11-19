# biomimox_gnn.py
# Ablation arm 5: single-task 3-D GNN, with the five information losses fixed.
#
#   fix 1  conformer ensemble (random conformer per epoch during training,
#          averaged probabilities at evaluation) instead of one ETKDG/UFF pose
#   fix 2  atom features are consumed by the 2-D branch instead of discarded
#   fix 3  the real bond graph carries bond order / conjugation / ring / stereo;
#          the 3-D backbone builds its own neighbour list from `pos`
#   fix 4  scaffold split by default; the OOD cutoff is reported, never used
#          to select anything
#   fix 5  a 4-point grid plus repeated CV, so everything carries a spread
#
# Backbone: DimeNet++ by default. If it is too slow, switch with
#     BIOMIMOX_BACKBONE=schnet python biomimox_gnn.py
#     BIOMIMOX_BACKBONE=none   python biomimox_gnn.py     # 2-D only, seconds
# The script also prints a projected wall-clock after the first fold and tells
# you when switching is worth it.
#
# Outputs (ablation_results/ and gnn_models/<assay>/):
#   gnn_summary.csv, gnn_cv_<assay>.csv, gnn_test_<assay>.csv
#   gnn_models/<assay>/{gnn.pth, meta.pt, summary.json, *.svg}
#
# Usage:
#   python biomimox_prepare.py && python biomimox_gnn.py

from __future__ import annotations

import math
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
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

import biomimox_common as C
import biomimox_graphs as G
import biomimox_models as M

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BACKBONE = os.environ.get("BIOMIMOX_BACKBONE", "dimenet++")
USE_2D_BRANCH = os.environ.get("BIOMIMOX_USE_2D", "1") not in ("0", "false", "False")
LOSS_NAME = os.environ.get("BIOMIMOX_LOSS", "bce")        # "bce" or "focal"

CUTOFF = float(os.environ.get("BIOMIMOX_CUTOFF", "5.0"))
# The triplet count inside DimeNet++ scales steeply with this: a 45-atom
# molecule gives ~24k triplets at 32 neighbours but ~10k at 16. Lowering it to
# 16 (or the cutoff to 4.0 A) roughly halves DimeNet++ cost on the larger
# compounds. Both change the geometry the model sees, so treat a change as an
# ablation to report, not a free speed-up.
MAX_NEIGHBORS = int(os.environ.get("BIOMIMOX_MAX_NEIGHBORS", "32"))
DPP_BLOCKS = 4
SCHNET_INTERACTIONS = 4

MAX_EPOCHS = int(os.environ.get("BIOMIMOX_EPOCHS", "120"))
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 1e-4
GRAD_CLIP = 5.0

MODEL_DIR = Path("gnn_models")

# Repeat 0 selects the configuration; the remaining repeats are the reported
# evaluation, so selection and reporting do not share fold assignments.
SELECT_REPEAT = 0

_BASE_GRID: List[Dict[str, object]] = [
    {"hidden": 64,  "batch_size": 16, "lr": 1e-3, "weight_decay": 1e-5, "dropout": 0.10, "activation": "relu", "gine_layers": 3},
    {"hidden": 32,  "batch_size": 16, "lr": 1e-3, "weight_decay": 1e-5, "dropout": 0.20, "activation": "relu", "gine_layers": 3},
    {"hidden": 64,  "batch_size": 8,  "lr": 5e-4, "weight_decay": 1e-5, "dropout": 0.10, "activation": "relu", "gine_layers": 4},
    {"hidden": 128, "batch_size": 16, "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.30, "activation": "relu", "gine_layers": 3},
]


def build_config_grid(backbone: str) -> List[Dict[str, object]]:
    """Grid sized to the backbone.

    hidden=128 is dropped for DimeNet++: it is 1.28M parameters against ~100
    training molecules, it costs several times a hidden=64 fold, and higher
    dropout plus weight decay make it the slowest config to early-stop. It is
    kept for SchNet and the 2-D model, where a fold is cheap.
    """
    grid = list(_BASE_GRID)
    if str(backbone).lower().startswith("dimenet"):
        grid = [c for c in grid if int(c["hidden"]) <= 64]
    if os.environ.get("BIOMIMOX_FULL_GRID", "0") in ("1", "true", "True"):
        grid = list(_BASE_GRID)
    max_cfg = os.environ.get("BIOMIMOX_MAX_CONFIGS")
    if max_cfg:
        grid = grid[: int(max_cfg)]
    return grid


CONFIG_GRID: List[Dict[str, object]] = build_config_grid(BACKBONE)

TIME_BUDGET_HOURS = float(os.environ.get("BIOMIMOX_TIME_BUDGET_H", "8"))


def arm_id() -> str:
    tag = {"dimenet++": "dimenetpp", "schnet": "schnet", "none": "gine2d"}.get(BACKBONE, BACKBONE)
    return f"5_gnn_{tag}" + ("" if USE_2D_BRANCH else "_no2d")


def arm_label() -> str:
    base = {"dimenet++": "DimeNet++", "schnet": "SchNet", "none": "GINE (2-D only)"}.get(
        BACKBONE, BACKBONE)
    suffix = " + bond graph" if (USE_2D_BRANCH and BACKBONE != "none") else ""
    return f"5  single-task {base}{suffix}"


# -----------------------------------------------------------------------------
# Train / evaluate one model
# -----------------------------------------------------------------------------
def build_model(config: Dict[str, object],
                global_in: int,
                n_tasks: int,
                device: torch.device) -> M.BiomimoxNet:
    return M.BiomimoxNet(
        global_in=global_in,
        n_tasks=n_tasks,
        backbone=BACKBONE,
        hidden=int(config["hidden"]),
        activation=str(config["activation"]),
        dropout=float(config["dropout"]),
        cutoff=CUTOFF,
        max_neighbors=MAX_NEIGHBORS,
        dpp_blocks=DPP_BLOCKS,
        schnet_interactions=SCHNET_INTERACTIONS,
        use_2d_branch=USE_2D_BRANCH,
        gine_layers=int(config.get("gine_layers", 3)),
    ).to(device)


def make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    y = np.asarray(y).astype(int).reshape(-1)
    counts = np.bincount(y, minlength=2).astype(float)
    counts[counts == 0] = 1.0
    w = (1.0 / counts)[y]
    return WeightedRandomSampler(
        weights=torch.tensor(w, dtype=torch.double),
        num_samples=len(w), replacement=True,
    )


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        mask = getattr(batch, "y_mask", None)
        loss = criterion(logits, batch.y, mask)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total += float(loss.detach()) * batch.num_graphs
        n += batch.num_graphs
    return total / max(n, 1)


@torch.no_grad()
def predict_molecules(model,
                      graphs,
                      mol_indices: Sequence[int],
                      y: Optional[np.ndarray],
                      g_mean: torch.Tensor,
                      g_std: torch.Tensor,
                      batch_size: int,
                      device: torch.device,
                      conf_mode: str = "all") -> Tuple[np.ndarray, np.ndarray]:
    """At inference: average the probability over the conformer ensemble.

    Returns (y_vec, prob_vec) aligned with the order of `mol_indices`.
    """
    model.eval()
    ds = G.MolGraphDataset(graphs, mol_indices, y=y, g_mean=g_mean, g_std=g_std,
                           conf_mode=conf_mode, seed=C.SEED)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False)

    probs: List[float] = []
    positions: List[int] = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs.extend(torch.sigmoid(logits).view(-1).cpu().numpy().tolist())
        positions.extend(batch.mol_pos.view(-1).cpu().numpy().tolist())

    uniq_pos, mean_prob = G.average_over_conformers(np.asarray(positions), np.asarray(probs))
    out = np.full(len(mol_indices), np.nan, dtype=float)
    out[uniq_pos] = mean_prob
    y_vec = (np.asarray(y, dtype=float).reshape(-1)
             if y is not None else np.full(len(mol_indices), np.nan))
    return y_vec, out


def fit_fold(graphs,
             train_mols: Sequence[int],
             val_mols: Sequence[int],
             y_train: np.ndarray,
             y_val: np.ndarray,
             config: Dict[str, object],
             device: torch.device,
             seed: int,
             verbose_prefix: str = "",
             fixed_epochs: Optional[int] = None) -> Dict[str, object]:
    """Fit one fold.

    Two modes, and the difference matters for the honesty of the reported CV:

      fixed_epochs=None  -- early stopping on the validation fold. Used only in
                            the configuration-selection pass.
      fixed_epochs=N     -- train exactly N epochs and never look at the
                            validation fold during training. Used for every
                            reported evaluation repeat, so the out-of-fold
                            predictions are not the product of a stopping rule
                            tuned on the fold being scored. Without this, CV
                            PR-AUC is optimistically biased -- on random labels
                            the early-stopped version scores well above
                            prevalence, which is exactly the inflation the
                            baselines cannot benefit from.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    g_mean, g_std = G.global_scaler(graphs, train_mols)
    global_in = G.global_feature_dim(graphs)

    train_ds = G.MolGraphDataset(graphs, train_mols, y=y_train,
                                 g_mean=g_mean, g_std=g_std,
                                 conf_mode="random", seed=seed)
    train_loader = DataLoader(train_ds, batch_size=int(config["batch_size"]),
                              sampler=make_weighted_sampler(y_train))

    model = build_model(config, global_in, 1, device)
    criterion = M.make_loss(LOSS_NAME, y_train.reshape(-1, 1), 1, device)
    optimizer = AdamW(model.parameters(), lr=float(config["lr"]),
                      weight_decay=float(config["weight_decay"]))

    early_stop = fixed_epochs is None
    n_epochs = MAX_EPOCHS if early_stop else int(fixed_epochs)

    best_metric, best_epoch, best_state = -np.inf, n_epochs, None
    patience = 0
    train_losses: List[float] = []
    val_pr: List[float] = []
    val_roc: List[float] = []

    for epoch in range(1, n_epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(loss)

        if early_stop:
            # Validation uses the lowest-energy conformer during the epoch loop
            # (cheap) and the full ensemble once at the end.
            _, p_val = predict_molecules(model, graphs, val_mols, y_val, g_mean, g_std,
                                         int(config["batch_size"]), device,
                                         conf_mode="first")
            mets = C.ranking_metrics(y_val, p_val)
            val_pr.append(mets["pr_auc"])
            val_roc.append(mets["roc_auc"])

            key = mets["pr_auc"]
            if np.isfinite(key) and key > best_metric + EARLY_STOPPING_MIN_DELTA:
                best_metric, best_epoch = key, epoch
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

        if verbose_prefix and (epoch == 1 or epoch % 20 == 0):
            extra = (f" PR={val_pr[-1]:.3f} ROC={val_roc[-1]:.3f}" if early_stop else "")
            print(f"{verbose_prefix} epoch {epoch:03d} loss={loss:.4f}{extra}")

        if early_stop and patience >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, p_val_final = predict_molecules(model, graphs, val_mols, y_val, g_mean, g_std,
                                       int(config["batch_size"]), device, conf_mode="all")
    return {
        "best_epoch": int(best_epoch),
        "val_probs": p_val_final,
        "train_losses": train_losses,
        "val_pr": val_pr,
        "val_roc": val_roc,
        "ranking": C.ranking_metrics(y_val, p_val_final),
        "n_params": M.count_parameters(model),
    }


def run_repeat(graphs,
               train_pool_mols: np.ndarray,
               y_pool: np.ndarray,
               groups_pool: Sequence[str],
               config: Dict[str, object],
               repeat: int,
               device: torch.device,
               verbose: bool = False,
               timer: Optional[Dict[str, float]] = None,
               fixed_epochs: Optional[int] = None) -> Dict[str, object]:
    """One full CV pass over the training pool for one configuration.

    Pass fixed_epochs to train a set number of epochs with no reference to the
    validation fold; leave it None for the selection pass, which is allowed to
    early-stop because its numbers are never reported.
    """
    folds = [f for f in C.repeated_cv_indices(
        y_pool, groups=groups_pool, n_splits=C.CV_SPLITS,
        n_repeats=repeat + 1, seed=C.SEED, mode=C.SPLIT_MODE) if f[0] == repeat]

    oof = np.full(len(y_pool), np.nan, dtype=float)
    best_epochs: List[int] = []
    curves: Dict[str, List[List[float]]] = {"train": [], "pr": [], "roc": []}
    n_params = 0

    pass_start = time.time()
    for _, fold_i, tr_local, va_local in folds:
        if np.unique(y_pool[tr_local]).size < 2:
            continue
        t0 = time.time()
        res = fit_fold(
            graphs,
            train_pool_mols[tr_local], train_pool_mols[va_local],
            y_pool[tr_local], y_pool[va_local],
            config, device, seed=C.SEED + 1000 * repeat + fold_i,
            verbose_prefix=(f"  r{repeat}f{fold_i} h={config['hidden']}" if verbose else ""),
            fixed_epochs=fixed_epochs,
        )
        dt = time.time() - t0
        if timer is not None:
            timer.setdefault("fold_times", []).append(dt)
            if timer.get("per_fold") is None:
                timer["per_fold"] = dt

        # Always print one line per fold. Silence for minutes on end is
        # indistinguishable from a hang, and only the first config used to be
        # verbose, so the expensive configs were exactly the quiet ones.
        print(f"    fold {fold_i} done in {dt:6.1f}s | "
              f"{len(res['train_losses']):>3d} epochs | "
              f"fold PR-AUC={res['ranking']['pr_auc']:.3f}", flush=True)

        oof[va_local] = res["val_probs"]
        best_epochs.append(res["best_epoch"])
        curves["train"].append(res["train_losses"])
        curves["pr"].append(res["val_pr"])
        curves["roc"].append(res["val_roc"])
        n_params = res["n_params"]

    threshold = C.optimal_threshold(y_pool, oof)
    metrics = C.threshold_metrics(y_pool, oof, threshold)
    epoch_rule = int(np.clip(round(float(np.median(best_epochs))) if best_epochs else 10,
                             1, MAX_EPOCHS))
    return {
        "repeat": repeat, "oof": oof, "threshold": threshold,
        "metrics": metrics, "epoch_rule": epoch_rule,
        "best_epochs": best_epochs, "curves": curves, "n_params": n_params,
        "pass_seconds": time.time() - pass_start,
    }


def fit_final(graphs,
              train_pool_mols: np.ndarray,
              y_pool: np.ndarray,
              config: Dict[str, object],
              epochs: int,
              device: torch.device) -> Dict[str, object]:
    torch.manual_seed(C.SEED)
    g_mean, g_std = G.global_scaler(graphs, train_pool_mols)
    train_ds = G.MolGraphDataset(graphs, train_pool_mols, y=y_pool,
                                 g_mean=g_mean, g_std=g_std,
                                 conf_mode="random", seed=C.SEED)
    loader = DataLoader(train_ds, batch_size=int(config["batch_size"]),
                        sampler=make_weighted_sampler(y_pool))

    model = build_model(config, G.global_feature_dim(graphs), 1, device)
    criterion = M.make_loss(LOSS_NAME, y_pool.reshape(-1, 1), 1, device)
    optimizer = AdamW(model.parameters(), lr=float(config["lr"]),
                      weight_decay=float(config["weight_decay"]))

    losses = []
    for epoch in range(1, int(epochs) + 1):
        losses.append(train_one_epoch(model, loader, criterion, optimizer, device))
        if epoch == 1 or epoch % 20 == 0 or epoch == int(epochs):
            print(f"  final epoch {epoch:03d}/{int(epochs):03d} loss={losses[-1]:.4f}")
    return {"model": model, "g_mean": g_mean, "g_std": g_std, "train_losses": losses}


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def _pad(arrs: List[List[float]]) -> np.ndarray:
    if not arrs:
        return np.zeros((0, 0))
    n = max(len(a) for a in arrs)
    out = np.full((len(arrs), n), np.nan)
    for i, a in enumerate(arrs):
        out[i, : len(a)] = a
    return out


def plot_cv(curves: Dict[str, List[List[float]]], out_dir: Path, prefix: str) -> None:
    for key, ylabel, fname in (("train", "Training loss", "cv_loss"),
                               ("pr", "PR-AUC", "cv_pr_vs_epoch"),
                               ("roc", "ROC-AUC", "cv_roc_vs_epoch")):
        mat = _pad(curves.get(key, []))
        if mat.size == 0:
            continue
        plt.figure(figsize=(5, 4), dpi=300)
        for row in mat:
            plt.plot(np.arange(1, mat.shape[1] + 1), row, alpha=0.25)
        plt.plot(np.arange(1, mat.shape[1] + 1), np.nanmean(mat, axis=0),
                 linewidth=2, label=f"{ylabel} (mean)")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        if key != "train":
            plt.ylim(0.0, 1.0)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_{fname}.svg", format="svg")
        plt.close()


def plot_test(df: pd.DataFrame, out_dir: Path, prefix: str, threshold: float) -> None:
    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
    y = df["y"].values.astype(int)
    p = df["prob"].values.astype(float)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if y.size == 0:
        return

    if np.unique(y).size == 2:
        fpr, tpr, _ = roc_curve(y, p)
        plt.figure(figsize=(5, 4), dpi=300)
        plt.plot(fpr, tpr, linewidth=2)
        plt.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
        plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
        plt.tight_layout(); plt.savefig(out_dir / f"{prefix}_test_roc.svg", format="svg"); plt.close()

        prec, rec, _ = precision_recall_curve(y, p)
        plt.figure(figsize=(5, 4), dpi=300)
        plt.plot(rec, prec, linewidth=2)
        plt.axhline(y.mean(), linestyle="--", color="grey", linewidth=1)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.tight_layout(); plt.savefig(out_dir / f"{prefix}_test_pr.svg", format="svg"); plt.close()

    cm = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1])
    plt.figure(figsize=(4, 4), dpi=300)
    plt.imshow(cm, interpolation="nearest")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"]); plt.yticks([0, 1], ["True 0", "True 1"])
    plt.tight_layout(); plt.savefig(out_dir / f"{prefix}_test_cm.svg", format="svg"); plt.close()

    if "ood_nn_distance" in df.columns:
        plt.figure(figsize=(5, 4), dpi=300)
        plt.scatter(df["ood_nn_distance"], df["prob"], alpha=0.8)
        plt.xlabel("Tanimoto distance to nearest training compound")
        plt.ylabel("Predicted probability")
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_test_prob_vs_ood.svg", format="svg")
        plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    C.set_workdir()
    C.ensure_dirs()
    C.set_seeds(C.SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not (C.SPLIT_DIR / "valid_molecules.csv").exists():
        raise RuntimeError("Run biomimox_prepare.py first to write splits/")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | backbone={BACKBONE} | 2-D branch={USE_2D_BRANCH} "
          f"| loss={LOSS_NAME} | arm={arm_id()}")

    mol_df = C.load_molecule_table()
    graphs, valid_mask = G.build_graph_dataset(mol_df, n_conformers=C.N_CONFORMERS,
                                               seed=C.SEED, use_cache=True, verbose=True)
    print(f"{len(graphs)} molecules with conformers, "
          f"{G.global_feature_dim(graphs)} global features")

    targets_done: List[str] = []
    summary_rows: List[Dict] = []
    all_cv: List[pd.DataFrame] = []
    all_test: List[pd.DataFrame] = []
    timer: Dict[str, object] = {"per_fold": None}
    warned_slow = False

    n_assays_total = sum(1 for t in C.TARGET_COLUMNS
                         if (C.SPLIT_DIR / f"{t}_split_manifest.csv").exists())
    n_eval_repeats = max(C.CV_REPEATS - 1, 1)
    passes_per_assay = len(CONFIG_GRID) + n_eval_repeats
    pass_times: List[float] = []
    print(f"{len(CONFIG_GRID)} configs x {n_assays_total} assays, plus "
          f"{n_eval_repeats} evaluation repeats each: "
          f"{n_assays_total * passes_per_assay} CV passes total")

    for target in C.TARGET_COLUMNS:
        manifest_path = C.SPLIT_DIR / f"{target}_split_manifest.csv"
        if not manifest_path.exists():
            print(f"\n=== {target}: no split manifest; skipped ===")
            continue

        manifest = pd.read_csv(manifest_path)
        y_all = manifest["y"].values.astype(int)
        mol_idx = manifest["dataset_idx"].values.astype(int)
        groups = manifest["scaffold"].astype(str).tolist()
        is_train = manifest["split"].values == "train"

        train_pool_mols = mol_idx[is_train]
        test_mols = mol_idx[~is_train]
        y_pool = y_all[is_train]
        y_test = y_all[~is_train]
        groups_pool = [g for g, t in zip(groups, is_train) if t]

        assay_dir = MODEL_DIR / target
        assay_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n========== {target} ==========")
        print(f"train {len(train_pool_mols)} (pos {int(y_pool.sum())}) | "
              f"test {len(test_mols)} (pos {int(y_test.sum())}) | "
              f"mode {manifest['split_mode'].iloc[0]}")

        # ---- configuration selection on the selection repeat only ----
        select_results: List[Tuple[Dict, Dict]] = []
        for cfg_i, config in enumerate(CONFIG_GRID, start=1):
            print(f"-- config {cfg_i}/{len(CONFIG_GRID)}: {config}", flush=True)
            res = run_repeat(graphs, train_pool_mols, y_pool, groups_pool,
                             config, SELECT_REPEAT, device, verbose=(cfg_i == 1),
                             timer=timer)
            pr = res["metrics"]["pr_auc"]
            roc = res["metrics"]["roc_auc"]
            print(f"   selection PR-AUC={pr:.3f} ROC-AUC={roc:.3f} "
                  f"epochs~{res['epoch_rule']} params={res['n_params']:,}", flush=True)
            select_results.append((config, res))
            pass_times.append(float(res["pass_seconds"]))

            # Projection from completed CV passes, not from the first fold of
            # the first config. The old probe extrapolated the cheapest fold of
            # the cheapest configuration across the whole run and was therefore
            # optimistic by several times.
            passes_done = len(pass_times)
            passes_total = n_assays_total * passes_per_assay
            mean_pass = float(np.mean(pass_times))
            remaining_h = mean_pass * max(passes_total - passes_done, 0) / 3600.0
            print(f"   [timing] this CV pass {res['pass_seconds'] / 60.0:.1f} min | "
                  f"mean {mean_pass / 60.0:.1f} min | "
                  f"{passes_done}/{passes_total} passes done | "
                  f"~{remaining_h:.1f} h remaining", flush=True)
            if remaining_h > TIME_BUDGET_HOURS and not warned_slow:
                warned_slow = True
                if BACKBONE.startswith("dimenet"):
                    print("   [timing] over budget. Options, cheapest first: "
                          "BIOMIMOX_BACKBONE=schnet (much faster), "
                          "BIOMIMOX_MAX_NEIGHBORS=16 (roughly halves the triplet "
                          "count), BIOMIMOX_MAX_CONFIGS=2, or "
                          "BIOMIMOX_REPEATS=3.", flush=True)
                else:
                    print("   [timing] over budget; consider "
                          "BIOMIMOX_MAX_CONFIGS=2 or BIOMIMOX_REPEATS=3.",
                          flush=True)

        best_config, best_select = max(
            select_results,
            key=lambda t: (-np.inf if not np.isfinite(t[1]["metrics"]["pr_auc"])
                           else t[1]["metrics"]["pr_auc"]),
        )
        print(f"Selected config: {best_config}")

        pd.DataFrame([{**cfg, "select_pr_auc": r["metrics"]["pr_auc"],
                       "select_roc_auc": r["metrics"]["roc_auc"],
                       "epoch_rule": r["epoch_rule"]}
                      for cfg, r in select_results]).to_csv(
            assay_dir / "config_selection.csv", index=False)

        # ---- reported evaluation on the remaining repeats ----
        # The epoch budget comes from the selection pass, and the evaluation
        # folds are then trained for exactly that many epochs. No stopping rule
        # ever sees the fold it is scored on.
        select_epochs = int(best_select["epoch_rule"])
        eval_repeats = [r for r in range(C.CV_REPEATS) if r != SELECT_REPEAT] or [SELECT_REPEAT]
        if len(eval_repeats) < 2:
            print("  [warning] fewer than 2 evaluation repeats: the SD column will "
                  "be empty. Raise BIOMIMOX_REPEATS (default 5) to get a spread.")
        cv_rows: List[Dict] = []
        epoch_rules: List[int] = []
        thresholds: List[float] = []
        curves_for_plot: Optional[Dict[str, List[List[float]]]] = None

        for rep in eval_repeats:
            res = run_repeat(graphs, train_pool_mols, y_pool, groups_pool,
                             best_config, rep, device, verbose=False, timer=timer,
                             fixed_epochs=select_epochs)
            pass_times.append(float(res["pass_seconds"]))
            met = dict(res["metrics"])
            met.update({"arm": arm_id(), "arm_label": arm_label(), "repeat": rep,
                        "stage": "cv_oof", **best_config})
            cv_rows.append(met)
            epoch_rules.append(res["epoch_rule"])
            thresholds.append(res["threshold"])
            if curves_for_plot is None:
                curves_for_plot = res["curves"]
            print(f"  repeat {rep}: CV PR-AUC={met['pr_auc']:.3f} "
                  f"ROC-AUC={met['roc_auc']:.3f} MCC={met['mcc']:.3f}")

        # Learning curves come from the selection pass, which is the pass that
        # tracked validation metrics per epoch.
        plot_cv(best_select["curves"], assay_dir, prefix="selected_config")
        if curves_for_plot and curves_for_plot.get("train"):
            plot_cv({"train": curves_for_plot["train"]}, assay_dir, prefix="eval_repeat")

        # ---- final refit on the whole training pool, one held-out evaluation ----
        epoch_rule = int(np.clip(round(float(np.median(epoch_rules)))
                                 if epoch_rules else select_epochs, 1, MAX_EPOCHS))
        threshold = float(np.median(thresholds))
        print(f"Refitting on the full training pool: {epoch_rule} epochs, "
              f"threshold {threshold:.3f}")
        final = fit_final(graphs, train_pool_mols, y_pool, best_config, epoch_rule, device)
        model = final["model"]

        _, p_train = predict_molecules(model, graphs, train_pool_mols, y_pool,
                                       final["g_mean"], final["g_std"],
                                       int(best_config["batch_size"]), device, "all")
        _, p_test = predict_molecules(model, graphs, test_mols, y_test,
                                      final["g_mean"], final["g_std"],
                                      int(best_config["batch_size"]), device, "all")

        train_smiles = manifest.loc[is_train, "canonical_smiles"].astype(str).tolist()
        test_smiles = manifest.loc[~is_train, "canonical_smiles"].astype(str).tolist()
        ood_ref = C.build_ood_reference(train_smiles, quantile=C.OOD_DISTANCE_QUANTILE)
        ood_df = C.score_ood_smiles(test_smiles, ood_ref["reference_smiles"], ood_ref["cutoff"])

        test_df = pd.DataFrame({
            "orig_idx": manifest.loc[~is_train, "orig_idx"].values,
            "SMILES": test_smiles,
            "y": y_test,
            "prob": p_test,
            "pred": (p_test >= threshold).astype(int),
            "split": "test",
        })
        test_df = pd.concat([test_df.reset_index(drop=True),
                             ood_df.reset_index(drop=True)], axis=1)
        train_df = pd.DataFrame({
            "orig_idx": manifest.loc[is_train, "orig_idx"].values,
            "SMILES": train_smiles, "y": y_pool, "prob": p_train,
            "pred": (p_train >= threshold).astype(int), "split": "train",
        })
        pd.concat([train_df, test_df], ignore_index=True).to_csv(
            assay_dir / "train_test_predictions.csv", index=False)

        test_met = C.threshold_metrics(y_test, p_test, threshold)
        test_met.update({"arm": arm_id(), "arm_label": arm_label(), "repeat": -1,
                         "stage": "test", "ood_cutoff": float(ood_ref["cutoff"]),
                         "ood_flag_rate": float(np.mean(test_df["ood_flag"].values)),
                         **best_config})
        plot_test(test_df, assay_dir, "selected_config", threshold)

        cv_df = pd.DataFrame(cv_rows); cv_df.insert(0, "target", target)
        te_df = pd.DataFrame([test_met]); te_df.insert(0, "target", target)
        cv_df.to_csv(C.RESULTS_DIR / f"gnn_cv_{target}.csv", index=False)
        te_df.to_csv(C.RESULTS_DIR / f"gnn_test_{target}.csv", index=False)
        all_cv.append(cv_df); all_test.append(te_df)

        agg_cv = C.aggregate_repeat_metrics(cv_rows)
        summary_rows.append({
            "target": target, "arm": arm_id(), "arm_label": arm_label(),
            "family": "gnn_single", "backbone": BACKBONE,
            "use_2d_branch": bool(USE_2D_BRANCH),
            "n_train": int(len(train_pool_mols)), "n_test": int(len(test_mols)),
            "prevalence": float(y_all.mean()),
            "n_params": int(best_select["n_params"]),
            **{f"cv_{k}": v for k, v in agg_cv.items()},
            **{f"test_{k}": float(test_met.get(k, np.nan))
               for k in ("roc_auc", "pr_auc", "f1", "mcc", "balanced_accuracy", "Se", "Sp")},
            **{f"cfg_{k}": v for k, v in best_config.items()},
        })

        torch.save(model.state_dict(), assay_dir / "gnn.pth")
        torch.save({
            "target_column": target,
            "arm": arm_id(),
            "model_type": "BiomimoxNet_single",
            "backbone": BACKBONE,
            "use_2d_branch": bool(USE_2D_BRANCH),
            "n_tasks": 1,
            "task_names": [target],
            "best_config": best_config,
            "epoch_rule": epoch_rule,
            "threshold": threshold,
            "thresholds": [threshold],
            "g_feature_names": C.GLOBAL_FEATURE_NAMES,
            "G_mean": final["g_mean"].cpu(),
            "G_std": final["g_std"].cpu(),
            "cutoff": float(CUTOFF),
            "max_neighbors": int(MAX_NEIGHBORS),
            "dpp_blocks": int(DPP_BLOCKS),
            "schnet_interactions": int(SCHNET_INTERACTIONS),
            "n_conformers": int(C.N_CONFORMERS),
            "loss_name": LOSS_NAME,
            "seed": int(C.SEED),
            "split_mode": str(manifest["split_mode"].iloc[0]),
            "cv_splits": int(C.CV_SPLITS),
            "cv_repeats": int(C.CV_REPEATS),
            "ood_method": "morgan_nn_distance",
            "ood_reference_smiles": ood_ref["reference_smiles"],
            "ood_nn_distance_cutoff": float(ood_ref["cutoff"]),
            "ood_quantile": float(ood_ref["quantile"]),
            "ood_morgan_radius": int(ood_ref["radius"]),
            "ood_morgan_nbits": int(ood_ref["nbits"]),
            "train_orig_idx": manifest.loc[is_train, "orig_idx"].tolist(),
            "test_orig_idx": manifest.loc[~is_train, "orig_idx"].tolist(),
        }, assay_dir / "meta.pt")

        C.write_json(assay_dir / "summary.json", {
            "target": target, "arm": arm_id(), "best_config": best_config,
            "epoch_rule": epoch_rule, "threshold": threshold,
            "cv": agg_cv, "test": test_met,
        })
        targets_done.append(target)
        print(f"Saved {assay_dir}")

    if all_cv:
        pd.concat(all_cv, ignore_index=True).to_csv(C.RESULTS_DIR / "gnn_cv_all.csv", index=False)
    if all_test:
        pd.concat(all_test, ignore_index=True).to_csv(C.RESULTS_DIR / "gnn_test_all.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    out = C.RESULTS_DIR / f"gnn_summary_{arm_id()}.csv"
    summary.to_csv(out, index=False)
    summary.to_csv(C.RESULTS_DIR / "gnn_summary.csv", index=False)
    print(f"\nFinished {len(targets_done)} assays. Wrote {out}")
    if not summary.empty:
        print(summary[["target", "cv_pr_auc_mean", "cv_pr_auc_sd",
                       "cv_roc_auc_mean", "test_pr_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()

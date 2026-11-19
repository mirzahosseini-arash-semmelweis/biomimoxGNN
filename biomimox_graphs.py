# biomimox_graphs.py
# PyTorch-Geometric graph construction and datasets for the biomimetic
# oxidation models. Kept separate from biomimox_common.py so the sklearn
# baselines never need to import torch.
#
# Each molecule becomes a list of Data objects, one per conformer:
#   x           [n_atoms, ATOM_FEATURE_DIM]   atom features        (fix 2)
#   z           [n_atoms]                     atomic numbers
#   pos         [n_atoms, 3]                  conformer geometry   (fix 1)
#   edge_index  [2, 2*n_bonds]                REAL bonds           (fix 3)
#   edge_attr   [2*n_bonds, BOND_FEATURE_DIM] bond features        (fix 3)
#   raw_g       [n_global]                    descriptors + hot flags + percepta
#
# The 5 A radius graph is deliberately NOT stored: DimeNet++/SchNet build their
# own neighbour lists from `pos`.

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import biomimox_common as C


# -----------------------------------------------------------------------------
# Single-molecule graph construction
# -----------------------------------------------------------------------------
def mol_conformer_to_data(mol, conf_id: int, raw_g: torch.Tensor) -> Data:
    x = torch.tensor([C.atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)

    conf = mol.GetConformer(conf_id)
    pos = torch.tensor(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())],
        dtype=torch.float,
    )

    src: List[int] = []
    dst: List[int] = []
    eattr: List[List[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = C.bond_features(bond)
        src += [i, j]
        dst += [j, i]
        eattr += [bf, bf]

    if len(src) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, C.BOND_FEATURE_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(eattr, dtype=torch.float)

    return Data(x=x, z=z, pos=pos, edge_index=edge_index,
                edge_attr=edge_attr, raw_g=raw_g.clone())


def smiles_to_conformer_graphs(smiles: str,
                               percepta: Sequence[float],
                               n_conformers: int = C.N_CONFORMERS,
                               seed: int = C.SEED) -> Optional[List[Data]]:
    mol3d = C.embed_conformers(smiles, n_conformers=n_conformers, seed=seed)
    if mol3d is None:
        return None

    heavy = C.largest_fragment(C.parse_smiles(smiles))
    desc = C.rdkit_global_descriptors(heavy)
    raw_g = torch.tensor(list(desc) + [float(v) for v in percepta], dtype=torch.float)

    graphs: List[Data] = []
    for cid in range(mol3d.GetNumConformers()):
        graphs.append(mol_conformer_to_data(mol3d, cid, raw_g))
    return graphs if graphs else None


# -----------------------------------------------------------------------------
# Dataset build + cache (built once, reused by every assay and the multitask run)
# -----------------------------------------------------------------------------
def _cache_key(mol_df: pd.DataFrame, n_conformers: int) -> str:
    h = hashlib.sha1()
    h.update(("|".join(mol_df["canonical_smiles"].astype(str).tolist())).encode("utf-8"))
    h.update(np.ascontiguousarray(mol_df[C.PERCEPTA_COLS].values.astype(np.float64)).tobytes())
    h.update(f"nconf={n_conformers};afd={C.ATOM_FEATURE_DIM};bfd={C.BOND_FEATURE_DIM};v3".encode())
    return h.hexdigest()[:16]


def build_graph_dataset(mol_df: pd.DataFrame,
                        n_conformers: int = C.N_CONFORMERS,
                        seed: int = C.SEED,
                        use_cache: bool = True,
                        cache_dir: Path = C.CACHE_DIR,
                        verbose: bool = True) -> Tuple[List[List[Data]], np.ndarray]:
    """Returns (graphs, valid_mask).

    graphs[i] is the conformer list for the i-th VALID molecule.
    valid_mask is a boolean array over rows of mol_df.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"graphs_{_cache_key(mol_df, n_conformers)}.pt"

    if use_cache and cache_path.exists():
        if verbose:
            print(f"Loading cached graphs from {cache_path}")
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        return blob["graphs"], np.asarray(blob["valid_mask"], dtype=bool)

    graphs: List[List[Data]] = []
    valid: List[bool] = []
    n = len(mol_df)
    for i, row in mol_df.reset_index(drop=True).iterrows():
        g = smiles_to_conformer_graphs(
            str(row["SMILES"]),
            [float(row[c]) for c in C.PERCEPTA_COLS],
            n_conformers=n_conformers,
            seed=seed,
        )
        if g is None:
            valid.append(False)
            if verbose:
                print(f"  [{i+1}/{n}] embedding failed: {row['SMILES']}")
            continue
        for d in g:
            d.mol_idx = torch.tensor([len(graphs)], dtype=torch.long)
            d.orig_idx = torch.tensor([int(row["orig_idx"])], dtype=torch.long)
        graphs.append(g)
        valid.append(True)
        if verbose and ((i + 1) % 25 == 0 or i + 1 == n):
            print(f"  featurised {i+1}/{n} molecules ({len(graphs)} valid)")

    valid_mask = np.asarray(valid, dtype=bool)
    if use_cache:
        torch.save({"graphs": graphs, "valid_mask": valid_mask.tolist()}, cache_path)
        if verbose:
            print(f"Cached graphs to {cache_path}")
    return graphs, valid_mask


def global_feature_dim(graphs: List[List[Data]]) -> int:
    return int(graphs[0][0].raw_g.numel())


def stack_raw_globals(graphs: List[List[Data]], mol_indices: Sequence[int]) -> torch.Tensor:
    return torch.stack([graphs[int(i)][0].raw_g for i in mol_indices], dim=0)


def global_scaler(graphs: List[List[Data]],
                  train_mol_indices: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit on inner-train molecules only -- no leakage into CV folds."""
    g = stack_raw_globals(graphs, train_mol_indices)
    mean = g.mean(dim=0)
    std = g.std(dim=0)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return mean, std


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------
class MolGraphDataset(torch.utils.data.Dataset):
    """Conformer-aware dataset.

    conf_mode:
      "random" -- sample one conformer per access (train-time augmentation)
      "first"  -- lowest-energy conformer only
      "all"    -- every conformer as a separate item (test-time ensembling)

    y / y_mask are [n_mol, n_tasks] float arrays; mask 0 means "label missing",
    which the masked loss ignores (multitask).
    """

    def __init__(self,
                 graphs: List[List[Data]],
                 mol_indices: Sequence[int],
                 y: Optional[np.ndarray] = None,
                 y_mask: Optional[np.ndarray] = None,
                 g_mean: Optional[torch.Tensor] = None,
                 g_std: Optional[torch.Tensor] = None,
                 conf_mode: str = "first",
                 seed: int = C.SEED):
        self.graphs = graphs
        self.mol_indices = np.asarray(list(mol_indices), dtype=int)
        self.conf_mode = str(conf_mode)
        self.g_mean = g_mean
        self.g_std = g_std
        self._rng = np.random.RandomState(int(seed))

        if y is None:
            self.y = None
            self.y_mask = None
        else:
            y_arr = np.asarray(y, dtype=float)
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(-1, 1)
            self.y = y_arr
            if y_mask is None:
                self.y_mask = np.isfinite(y_arr).astype(float)
            else:
                m = np.asarray(y_mask, dtype=float)
                self.y_mask = m.reshape(y_arr.shape)

        items: List[Tuple[int, int]] = []
        for pos, mi in enumerate(self.mol_indices):
            n_conf = len(self.graphs[int(mi)])
            if self.conf_mode == "all":
                items.extend([(pos, c) for c in range(n_conf)])
            else:
                items.append((pos, -1))
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Data:
        pos, conf = self.items[int(i)]
        mi = int(self.mol_indices[pos])
        confs = self.graphs[mi]
        if conf < 0:
            conf = int(self._rng.randint(len(confs))) if self.conf_mode == "random" else 0

        d = confs[conf].clone()
        if self.g_mean is not None and self.g_std is not None:
            d.g = ((d.raw_g - self.g_mean) / self.g_std).view(1, -1)
        else:
            d.g = d.raw_g.view(1, -1)

        d.mol_pos = torch.tensor([pos], dtype=torch.long)
        d.conf_id = torch.tensor([conf], dtype=torch.long)

        if self.y is not None:
            yv = np.nan_to_num(self.y[pos], nan=0.0)
            d.y = torch.tensor(yv, dtype=torch.float).view(1, -1)
            d.y_mask = torch.tensor(self.y_mask[pos], dtype=torch.float).view(1, -1)
        return d


def average_over_conformers(mol_pos: np.ndarray,
                            probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse per-conformer predictions to one probability per molecule.

    Returns (unique_mol_pos_sorted, mean_prob). Works for a single task column.
    """
    mol_pos = np.asarray(mol_pos, dtype=int).reshape(-1)
    probs = np.asarray(probs, dtype=float).reshape(-1)
    order = np.argsort(mol_pos, kind="stable")
    mp, pr = mol_pos[order], probs[order]
    uniq, start = np.unique(mp, return_index=True)
    sums = np.add.reduceat(np.nan_to_num(pr, nan=0.0), start)
    counts = np.add.reduceat(np.isfinite(pr).astype(float), start)
    counts = np.where(counts <= 0, np.nan, counts)
    return uniq, sums / counts


def average_over_conformers_2d(mol_pos: np.ndarray,
                               probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Same, for a [n_items, n_tasks] probability matrix."""
    mol_pos = np.asarray(mol_pos, dtype=int).reshape(-1)
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    order = np.argsort(mol_pos, kind="stable")
    mp, pr = mol_pos[order], probs[order]
    uniq, start = np.unique(mp, return_index=True)
    sums = np.add.reduceat(np.nan_to_num(pr, nan=0.0), start, axis=0)
    counts = np.add.reduceat(np.isfinite(pr).astype(float), start, axis=0)
    counts = np.where(counts <= 0, np.nan, counts)
    return uniq, sums / counts

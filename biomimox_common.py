# biomimox_common.py
# Shared featurisation, splitting, metric and OOD utilities for the
# biomimetic-oxidation ablation suite (baselines 0-4, single-task GNN 5,
# multitask GNN 6).
#
# Reads exactly the same inputs as GNN_binary_train.py:
#   input_files/SMILESforGNN.csv        column: SMILES
#   input_files/percepta_desc.csv       columns: refractivity, logD, neg, pos
#                                       (optionally SMILES for merge-by-key)
#   input_files/<assay>_outcomes.csv    column: <assay>   (0/1)
#
# What this pipeline features:
# -----------------------------------------------
# (1) Conformers: N_CONFORMERS ETKDGv3 embeddings with RMS pruning, MMFF94s
#     optimisation (UFF fallback), energy-sorted. Used for train-time
#     augmentation and test-time ensembling instead of one arbitrary conformer.
# (2) Atom features are never discarded: they are carried on Data.x and
#     consumed by a 2-D message-passing branch (see biomimox_gnn.py).
# (3) The real bond graph is carried on Data.edge_index / Data.edge_attr with
#     bond-order, conjugation, ring and stereo features. The 5 A radius graph
#     is left to the 3-D backbone, which builds it internally.
# (4) Splitting supports Bemis-Murcko scaffold splits so the held-out set is
#     not guaranteed to be a near-neighbour of the training set; OOD stats are
#     reported but never used to select anything.
# (5) Repeated stratified CV replaces most of the hyper-parameter grid, so every
#     reported number carries a spread across repeats.

from __future__ import annotations

import os
import json
import math
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    f1_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

RDLogger.DisableLog("rdApp.*")

# -----------------------------------------------------------------------------
# Working directory
# -----------------------------------------------------------------------------
# Override with:  set BIOMIMOX_DIR=...   (Windows)  /  export BIOMIMOX_DIR=...
DEFAULT_WORKDIR = Path(__file__).resolve().parent
WORKDIR = os.environ.get("BIOMIMOX_DIR", DEFAULT_WORKDIR)


def set_workdir(path: Optional[str] = None) -> Path:
    target = Path(path or WORKDIR)
    if target.exists():
        os.chdir(target)
    else:
        warnings.warn(f"Working directory {target} not found; staying in {os.getcwd()}")
    print("Working directory:", os.getcwd())
    return Path(os.getcwd())


# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20

# "scaffold" gives the honest, pessimistic estimate (fix 4).
# "random" reproduces the original 80/20 stratified split.
SPLIT_MODE = os.environ.get("BIOMIMOX_SPLIT", "scaffold")

# Non-generic Bemis-Murcko scaffolds by default; see murcko_scaffold().
GENERIC_SCAFFOLDS = os.environ.get("BIOMIMOX_GENERIC_SCAFFOLD", "0") in ("1", "true", "True")

# A scaffold split is only attempted when the groups are small enough to build
# a test set of roughly the requested size. If the largest group already
# exceeds this fraction of the dataset, the split falls back to stratified
# random and says so, rather than silently returning a lopsided partition.
MAX_SCAFFOLD_GROUP_FRACTION = 0.5
SCAFFOLD_SIZE_TOLERANCE = 1.15

CV_SPLITS = 5
CV_REPEATS = int(os.environ.get("BIOMIMOX_REPEATS", "5"))
MIN_CV_SPLITS = 2

N_CONFORMERS = int(os.environ.get("BIOMIMOX_NCONF", "5"))
CONF_PRUNE_RMS = 0.5
CONF_MAX_ITERS = 400

MORGAN_RADIUS = 2            # ECFP4
MORGAN_NBITS = 2048
OOD_DISTANCE_QUANTILE = 0.95

INPUT_DIR = Path("input_files")
SMILES_CSV = INPUT_DIR / "SMILESforGNN.csv"
PERCEPTA_CSV = INPUT_DIR / "percepta_desc.csv"

SPLIT_DIR = Path("splits")
RESULTS_DIR = Path("ablation_results")
CACHE_DIR = Path("cache")

PERCEPTA_COLS = ["refractivity", "logD", "neg", "pos"]

# All 12 original targets. `percent` is kept for the single-task runs and the
# baselines, but excluded from the multitask heads because it is a different
# kind of endpoint (overall turnover) rather than one reaction type.
TARGET_COLUMNS = [
    "percent",
    "Ar_ox", "Bn_ox", "deH_ox",
    "N_dealk", "N_dearyl",
    "O_dealk", "O_dearyl", "OH_ox",
    "R_ox", "S_ox", "unk_ox",
]
MULTITASK_TARGETS = [t for t in TARGET_COLUMNS if t != "percent"]

# -----------------------------------------------------------------------------
# RDKit global descriptors (unchanged from the original, same order)
# -----------------------------------------------------------------------------
GLOBAL_DESC_FUNCS = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHAcceptors", rdMolDescriptors.CalcNumHBA),
    ("NumHDonors", rdMolDescriptors.CalcNumHBD),
    ("NumRotatableBonds", rdMolDescriptors.CalcNumRotatableBonds),
    ("RingCount", rdMolDescriptors.CalcNumRings),
    ("HeavyAtomCount", rdMolDescriptors.CalcNumHeavyAtoms),
    ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
]

# -----------------------------------------------------------------------------
# Chemistry-aware hot scaffolds (identical SMARTS and identical ordering to the
# original scripts, so feature vectors stay comparable)
# -----------------------------------------------------------------------------
SmartsLike = Union[str, List[str], Tuple[str, ...]]
_RING_ATOM = "[#6,#7,#8,#16;!a;R]"

HOT_SCAFFOLD_SMARTS: Dict[str, SmartsLike] = {
    "hot_Ar_ox": [
        "[aH]~[a;!#6,$([a]([!#6;!#1]))]",
        "[aH]~[a]~[a]~[a;!#6,$([a]([!#6;!#1]))]",
    ],
    "hot_Bn_ox": "[c]-[CX4H1,CX4H2,CX4H3]",
    "hot_deH_ox": [
        "[NX3,OX2,SX2][CH2,CH][CH2,CH]",
        "[CH2,CH][CH2,CH][NX3,OX2,SX2]",
        "[C;X3;!a;H1,H2]=[C;X3;!a;H1,H2]",
        f"{_RING_ATOM}1={_RING_ATOM}{_RING_ATOM}{_RING_ATOM}{_RING_ATOM}1",
        f"{_RING_ATOM}1{_RING_ATOM}={_RING_ATOM}{_RING_ATOM}{_RING_ATOM}1",
        f"{_RING_ATOM}1{_RING_ATOM}{_RING_ATOM}={_RING_ATOM}{_RING_ATOM}1",
        f"{_RING_ATOM}1{_RING_ATOM}{_RING_ATOM}{_RING_ATOM}={_RING_ATOM}1",
        f"{_RING_ATOM}1{_RING_ATOM}{_RING_ATOM}{_RING_ATOM}{_RING_ATOM}=1",
        f"{_RING_ATOM}1={_RING_ATOM}{_RING_ATOM}{_RING_ATOM}{_RING_ATOM}{_RING_ATOM}1",
        f"{_RING_ATOM}1{_RING_ATOM}={_RING_ATOM}{_RING_ATOM}{_RING_ATOM}{_RING_ATOM}1",
        f"{_RING_ATOM}1{_RING_ATOM}{_RING_ATOM}={_RING_ATOM}{_RING_ATOM}{_RING_ATOM}1",
        f"{_RING_ATOM}1{_RING_ATOM}{_RING_ATOM}{_RING_ATOM}={_RING_ATOM}{_RING_ATOM}1",
        f"{_RING_ATOM}1{_RING_ATOM}{_RING_ATOM}{_RING_ATOM}{_RING_ATOM}={_RING_ATOM}1",
        f"{_RING_ATOM}1{_RING_ATOM}{_RING_ATOM}{_RING_ATOM}{_RING_ATOM}{_RING_ATOM}=1",
        "[c]1[c]([OX2H,NX3H1])cccc1[OX2H,NX3H1]",
        "[c]1cc([OX2H,NX3H1])ccc1[OX2H,NX3H1]",
    ],
    "hot_N_dealk": "[N;!$([N+]);!$([N-]);!$([N]-[CX3](=O))]~[CX4;H1,H2,H3]",
    "hot_N_dearyl": "[N;!$([N+]);!$([N-]);!$([N]-[CX3](=O))]~[c]",
    "hot_N_ox": "[NX3;H0;!$([N+]);!$([N-])](~[#6])(~[#6])",
    "hot_O_dealk": "[O;H0;!$([O-]);!$([O+])]~[CX4;H1,H2,H3]",
    "hot_O_dearyl": "[O;H0;!$([O-]);!$([O+])]~[c]",
    "hot_OH_ox": "[OX2H]-[CX4;H1,H2,H3;!a]",
    "hot_R_ox": [
        "[NX3;!$([N+]);!$([N-]);!$([N]-[CX3](=O))]-[CX4;H1,H2]-[CX4;H1,H2,H3]",
        "[CX4;H1,H2,H3]-[CX4;H1,H2]-[NX3;!$([N+]);!$([N-]);!$([N]-[CX3](=O))]",
        "[O;H0;!$([O-]);!$([O+])]-[CX4;H1,H2]-[CX4;H1,H2,H3]",
        "[CX4;H1,H2,H3]-[CX4;H1,H2]-[O;H0;!$([O-]);!$([O+])]",
        "[CX3](=O)-[CX4;H1,H2]-[CX4;H1,H2,H3]",
        "[CX4;H1,H2,H3]-[CX4;H1,H2]-[CX3](=O)",
        "[CX4H1;!a]([#6;!a])([#6;!a])[#6;!a]",
    ],
    "hot_S_ox": "[S;H0;!$([S+]);!$([S-])](~[#6])(~[#6])",
}


def _as_smarts_list(x: SmartsLike) -> List[str]:
    return list(x) if isinstance(x, (list, tuple)) else [x]


HOT_SCAFFOLD_SMARTS["hot_unk_ox"] = (
    _as_smarts_list(HOT_SCAFFOLD_SMARTS["hot_Ar_ox"])
    + _as_smarts_list(HOT_SCAFFOLD_SMARTS["hot_Bn_ox"])
    + _as_smarts_list(HOT_SCAFFOLD_SMARTS["hot_N_ox"])
    + _as_smarts_list(HOT_SCAFFOLD_SMARTS["hot_R_ox"])
    + _as_smarts_list(HOT_SCAFFOLD_SMARTS["hot_S_ox"])
)


def _compile_hot_scaffolds(hot_smarts: Dict[str, SmartsLike]) -> List[Tuple[str, List[Chem.Mol]]]:
    compiled: List[Tuple[str, List[Chem.Mol]]] = []
    for name, smarts_like in hot_smarts.items():
        pats: List[Chem.Mol] = []
        for sm in _as_smarts_list(smarts_like):
            try:
                m = Chem.MolFromSmarts(sm)
            except Exception:
                m = None
            if m is not None:
                pats.append(m)
        if pats:
            compiled.append((name, pats))
    return compiled


HOT_SCAFFOLDS: List[Tuple[str, List[Chem.Mol]]] = _compile_hot_scaffolds(HOT_SCAFFOLD_SMARTS)
HOT_SCAFFOLD_NAMES = [name for name, _ in HOT_SCAFFOLDS]
GLOBAL_FEATURE_NAMES = (
    [name for name, _ in GLOBAL_DESC_FUNCS] + HOT_SCAFFOLD_NAMES + PERCEPTA_COLS
)

# -----------------------------------------------------------------------------
# Atom / bond feature vocabulary
# -----------------------------------------------------------------------------
COMMON_ELEMENTS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
DEGREE_BUCKETS = [0, 1, 2, 3, 4]
NUM_H_BUCKETS = [0, 1, 2, 3, 4]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


def one_hot(value, allowed: Sequence) -> List[int]:
    if value not in allowed:
        value = allowed[-1]
    return [int(value == a) for a in allowed]


def safe_float(x) -> float:
    if x is None:
        return 0.0
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(xf) or math.isinf(xf):
        return 0.0
    return xf


def atom_features(atom: Chem.rdchem.Atom) -> List[float]:
    """Fix 2: these are now actually consumed by the 2-D branch of the model."""
    feats: List[float] = []
    feats += one_hot(atom.GetAtomicNum(), COMMON_ELEMENTS)
    feats += one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
    feats.append(int(atom.GetIsAromatic()))
    feats += one_hot(min(atom.GetTotalDegree(), DEGREE_BUCKETS[-1]), DEGREE_BUCKETS)
    feats += one_hot(min(atom.GetTotalNumHs(includeNeighbors=True), NUM_H_BUCKETS[-1]), NUM_H_BUCKETS)
    feats.append(int(atom.IsInRing()))
    feats.append(int(atom.IsInRingSize(3)))
    feats.append(int(atom.IsInRingSize(5)))
    feats.append(int(atom.IsInRingSize(6)))
    feats.append(safe_float(atom.GetFormalCharge()))
    feats.append(safe_float(atom.GetNumRadicalElectrons()))
    return feats


def bond_features(bond: Chem.rdchem.Bond) -> List[float]:
    """Fix 3: real bonds, with the information a radius graph throws away."""
    feats: List[float] = []
    feats += one_hot(bond.GetBondType(), BOND_TYPES)
    feats.append(int(bond.GetIsConjugated()))
    feats.append(int(bond.IsInRing()))
    feats += one_hot(bond.GetStereo(), BOND_STEREO)
    return feats


ATOM_FEATURE_DIM = (
    len(COMMON_ELEMENTS) + len(HYBRIDIZATIONS) + 1 + len(DEGREE_BUCKETS)
    + len(NUM_H_BUCKETS) + 4 + 2
)
BOND_FEATURE_DIM = len(BOND_TYPES) + 2 + len(BOND_STEREO)


def hot_scaffold_flags(mol: Chem.Mol) -> List[int]:
    flags: List[int] = []
    for _, patterns in HOT_SCAFFOLDS:
        try:
            flags.append(int(any(mol.HasSubstructMatch(pat) for pat in patterns)))
        except Exception:
            flags.append(0)
    return flags


def rdkit_global_descriptors(mol: Chem.Mol) -> List[float]:
    vals: List[float] = []
    for _, fn in GLOBAL_DESC_FUNCS:
        try:
            vals.append(safe_float(fn(mol)))
        except Exception:
            vals.append(0.0)
    vals.extend(hot_scaffold_flags(mol))
    return vals


# -----------------------------------------------------------------------------
# Molecule handling
# -----------------------------------------------------------------------------
def largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """The original kept frags[0]; keeping the heaviest fragment is the
    convention that actually drops counter-ions rather than whichever
    fragment happened to be written first."""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) > 1:
        return max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return mol


def parse_smiles(smiles: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return largest_fragment(mol)


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def murcko_scaffold(smiles: str, generic: bool = GENERIC_SCAFFOLDS) -> str:
    """Bemis-Murcko scaffold, falling back to the canonical SMILES for acyclic
    molecules so each becomes its own group.

    The default is the ordinary (non-generic) scaffold, which keeps heteroatoms
    and bond orders. The generic form collapses every ring to a carbocycle,
    which on a drug-like set of this size tends to merge most molecules into a
    single benzene-shaped group and makes a scaffold split impossible. Set
    BIOMIMOX_GENERIC_SCAFFOLD=1 to use the generic form anyway.
    """
    mol = parse_smiles(smiles)
    if mol is None:
        return str(smiles)
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaf = MurckoScaffold.MakeScaffoldGeneric(scaf)
        smi = Chem.MolToSmiles(scaf, canonical=True)
        return smi if smi else Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return Chem.MolToSmiles(mol, canonical=True)


def embed_conformers(smiles: str,
                     n_conformers: int = N_CONFORMERS,
                     seed: int = SEED) -> Optional[Chem.Mol]:
    """Fix 1: a small energy-sorted conformer ensemble rather than one
    arbitrary ETKDG/UFF structure.

    Returns an H-added mol carrying up to n_conformers conformers, ordered by
    force-field energy (lowest first). Conformer ids are 0..n-1.
    """
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.pruneRmsThresh = CONF_PRUNE_RMS
    params.useSmallRingTorsions = True
    params.enforceChirality = True

    # Ask for a few extra: pruning and failed embeddings reduce the yield.
    n_try = max(int(n_conformers) * 4, int(n_conformers) + 4)
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_try, params=params))

    if len(cids) == 0:
        params.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_try, params=params))
    if len(cids) == 0:
        return None

    energies: List[Tuple[float, int]] = []
    optimised = False
    try:
        mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if mp is not None:
            res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=CONF_MAX_ITERS,
                                                    mmffVariant="MMFF94s")
            energies = [(float(e), int(cid)) for (_, e), cid in zip(res, cids)]
            optimised = True
    except Exception:
        optimised = False

    if not optimised:
        try:
            res = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=CONF_MAX_ITERS)
            energies = [(float(e), int(cid)) for (_, e), cid in zip(res, cids)]
        except Exception:
            energies = [(0.0, int(cid)) for cid in cids]

    energies = [(e if np.isfinite(e) else 1e9, cid) for e, cid in energies]
    energies.sort(key=lambda t: t[0])
    keep = [cid for _, cid in energies[: int(n_conformers)]]

    out = Chem.Mol(mol)
    out.RemoveAllConformers()
    for cid in keep:
        conf = Chem.Conformer(mol.GetConformer(cid))
        conf.SetId(out.GetNumConformers())
        out.AddConformer(conf, assignId=False)

    if out.GetNumConformers() == 0:
        return None
    return out


# -----------------------------------------------------------------------------
# Input tables
# -----------------------------------------------------------------------------
def load_percepta_table(smiles_csv: Union[str, Path] = SMILES_CSV,
                        percepta_csv: Union[str, Path] = PERCEPTA_CSV,
                        cols: Sequence[str] = tuple(PERCEPTA_COLS)) -> pd.DataFrame:
    df_smi = pd.read_csv(smiles_csv)
    df_p = pd.read_csv(percepta_csv)

    if "SMILES" in df_p.columns and "SMILES" in df_smi.columns:
        df_merged = df_smi[["SMILES"]].merge(df_p, on="SMILES", how="left")
    else:
        if len(df_p) != len(df_smi):
            raise ValueError(
                f"{percepta_csv} has no SMILES column and row count does not match {smiles_csv}"
            )
        df_merged = pd.concat(
            [df_smi.reset_index(drop=True), df_p.reset_index(drop=True)], axis=1
        )

    for c in cols:
        if c not in df_merged.columns:
            raise ValueError(f"Missing Percepta column: {c}")

    out = pd.DataFrame(index=df_merged.index)
    for c in cols:
        out[c] = pd.to_numeric(df_merged[c], errors="coerce")
    for c in ("refractivity", "logD"):
        if c in out.columns:
            out[c] = out[c].fillna(out[c].mean())
    for c in ("neg", "pos"):
        if c in out.columns:
            out[c] = out[c].fillna(0.0)
    return out.fillna(0.0)


def load_molecule_table(smiles_csv: Union[str, Path] = SMILES_CSV,
                        percepta_csv: Union[str, Path] = PERCEPTA_CSV) -> pd.DataFrame:
    """One row per input molecule: index, SMILES, canonical SMILES, scaffold,
    Percepta columns. Shared by every ablation arm."""
    df_smi = pd.read_csv(smiles_csv)
    if "SMILES" not in df_smi.columns:
        raise ValueError(f"{smiles_csv} must have a 'SMILES' column header")
    df_p = load_percepta_table(smiles_csv, percepta_csv)
    if len(df_smi) != len(df_p):
        raise ValueError("Row count mismatch between SMILES and Percepta tables")

    df = pd.DataFrame({
        "orig_idx": np.arange(len(df_smi), dtype=int),
        "SMILES": df_smi["SMILES"].astype(str).tolist(),
    })
    df["canonical_smiles"] = [canonicalize_smiles(s) or s for s in df["SMILES"]]
    df["scaffold"] = [murcko_scaffold(s) for s in df["SMILES"]]
    for c in PERCEPTA_COLS:
        df[c] = df_p[c].astype(float).values
    return df


def load_targets(targets: Sequence[str] = TARGET_COLUMNS,
                 n_rows: Optional[int] = None,
                 input_dir: Union[str, Path] = INPUT_DIR) -> pd.DataFrame:
    """Wide 0/1 label frame; NaN where a label file is absent (multitask mask)."""
    input_dir = Path(input_dir)
    frames: Dict[str, np.ndarray] = {}
    for t in targets:
        path = input_dir / f"{t}_outcomes.csv"
        if not path.exists():
            warnings.warn(f"Missing outcome file {path}; target '{t}' will be all-NaN")
            frames[t] = np.full(n_rows if n_rows else 0, np.nan)
            continue
        df_y = pd.read_csv(path)
        if t not in df_y.columns:
            raise ValueError(f"Missing target column '{t}' in {path}")
        vals = pd.to_numeric(df_y[t], errors="coerce").values.astype(float)
        if n_rows is not None and len(vals) != n_rows:
            raise ValueError(f"{path}: {len(vals)} rows, expected {n_rows}")
        frames[t] = vals
    return pd.DataFrame(frames)


# -----------------------------------------------------------------------------
# Fingerprints and baseline feature blocks
# -----------------------------------------------------------------------------
def morgan_fp_from_smiles(smiles: str,
                          radius: int = MORGAN_RADIUS,
                          nbits: int = MORGAN_NBITS):
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    gen = GetMorganGenerator(radius=radius, fpSize=nbits)
    return gen.GetFingerprint(mol)


def ecfp_matrix(smiles_list: Sequence[str],
                radius: int = MORGAN_RADIUS,
                nbits: int = MORGAN_NBITS) -> np.ndarray:
    X = np.zeros((len(smiles_list), nbits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        fp = morgan_fp_from_smiles(smi, radius=radius, nbits=nbits)
        if fp is None:
            continue
        arr = np.zeros((nbits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i] = arr.astype(np.float32)
    return X


def hot_flag_matrix(smiles_list: Sequence[str]) -> np.ndarray:
    X = np.zeros((len(smiles_list), len(HOT_SCAFFOLDS)), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = parse_smiles(smi)
        if mol is None:
            continue
        X[i] = np.asarray(hot_scaffold_flags(mol), dtype=np.float32)
    return X


def rdkit_descriptor_matrix(smiles_list: Sequence[str]) -> np.ndarray:
    """The 9 plain RDKit descriptors only (no hot flags)."""
    n_desc = len(GLOBAL_DESC_FUNCS)
    X = np.zeros((len(smiles_list), n_desc), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = parse_smiles(smi)
        if mol is None:
            continue
        vals = []
        for _, fn in GLOBAL_DESC_FUNCS:
            try:
                vals.append(safe_float(fn(mol)))
            except Exception:
                vals.append(0.0)
        X[i] = np.asarray(vals, dtype=np.float32)
    return X


def percepta_matrix(mol_df: pd.DataFrame) -> np.ndarray:
    return mol_df[PERCEPTA_COLS].values.astype(np.float32)


def feature_block_names() -> Dict[str, List[str]]:
    return {
        "ecfp4": [f"ecfp4_{i}" for i in range(MORGAN_NBITS)],
        "hot": list(HOT_SCAFFOLD_NAMES),
        "rdkit": [n for n, _ in GLOBAL_DESC_FUNCS],
        "percepta": list(PERCEPTA_COLS),
    }


# -----------------------------------------------------------------------------
# Splitting (fix 4)
# -----------------------------------------------------------------------------
def scaffold_split(scaffolds: Sequence[str],
                   y: np.ndarray,
                   test_size: float = TEST_SIZE,
                   seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Group molecules by Murcko scaffold and fill the test set with whole
    groups, so no scaffold appears on both sides of the split.

    Groups are considered largest-first, but a group is only accepted if it
    still fits inside the size budget -- otherwise one oversized group would
    swallow the whole test set. Among groups that fit, one carrying positives
    is preferred while the test positive rate is below target, which keeps rare
    labels from ending up entirely in the training pool.

    Returns (train_idx, test_idx). May return an empty test set when the groups
    cannot be packed sensibly; make_split() detects that and falls back.
    """
    y = np.asarray(y).astype(int).reshape(-1)
    n = len(y)
    n_test_target = max(int(round(test_size * n)), 1)
    n_test_cap = max(int(np.ceil(n_test_target * SCAFFOLD_SIZE_TOLERANCE)), 1)
    n_pos_target = max(int(round(test_size * int(y.sum()))), 1)

    groups: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(scaffolds):
        groups[str(s)].append(i)

    # One dominant group makes a scaffold split meaningless.
    largest = max(len(v) for v in groups.values())
    if largest > MAX_SCAFFOLD_GROUP_FRACTION * n:
        return np.arange(n, dtype=int), np.asarray([], dtype=int)

    rng = np.random.RandomState(seed)
    keys = list(groups.keys())
    rng.shuffle(keys)
    keys.sort(key=lambda k: len(groups[k]), reverse=True)

    test_idx: List[int] = []
    test_pos = 0

    def accept(k: str) -> bool:
        nonlocal test_pos
        members = groups[k]
        if len(test_idx) + len(members) > n_test_cap:
            return False
        test_idx.extend(members)
        test_pos += int(y[members].sum())
        return True

    used: set = set()
    # Pass 1: groups that contain positives, while the positive budget allows.
    for k in keys:
        if len(test_idx) >= n_test_target:
            break
        if int(y[groups[k]].sum()) == 0:
            continue
        if test_pos >= n_pos_target:
            break
        if accept(k):
            used.add(k)

    # Pass 2: everything else, to reach the size target.
    for k in keys:
        if k in used:
            continue
        if len(test_idx) >= n_test_target:
            break
        if accept(k):
            used.add(k)

    # Pass 3: if still short, take the smallest unused groups that fit.
    if len(test_idx) < n_test_target:
        for k in sorted((k for k in keys if k not in used), key=lambda k: len(groups[k])):
            if len(test_idx) >= n_test_target:
                break
            if accept(k):
                used.add(k)

    test_idx_arr = np.sort(np.asarray(sorted(set(test_idx)), dtype=int))
    train_idx_arr = np.sort(np.setdiff1d(np.arange(n, dtype=int), test_idx_arr))
    return train_idx_arr, test_idx_arr


def make_split(mol_df: pd.DataFrame,
               y: np.ndarray,
               mode: str = SPLIT_MODE,
               test_size: float = TEST_SIZE,
               seed: int = SEED) -> Tuple[np.ndarray, np.ndarray, str]:
    """Returns (train_idx, test_idx, mode_used). Falls back to a stratified
    random split if a scaffold split leaves a class empty on either side."""
    y = np.asarray(y).astype(int).reshape(-1)
    idx = np.arange(len(y), dtype=int)

    if mode == "scaffold":
        tr, te = scaffold_split(mol_df["scaffold"].tolist(), y, test_size, seed)
        n_test_cap = int(np.ceil(test_size * len(y) * 2.0))
        ok = (
            len(tr) > 0 and len(te) > 0
            and len(te) <= n_test_cap
            and len(te) < len(tr)
            and np.unique(y[tr]).size == 2
            and np.unique(y[te]).size == 2
        )
        if ok:
            return tr, te, "scaffold"
        warnings.warn(
            "Scaffold split degenerate for this target (scaffold groups too few "
            "or too large to pack); using a stratified random split instead"
        )

    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    return np.sort(tr), np.sort(te), "random"


def repeated_cv_indices(y: np.ndarray,
                        groups: Optional[Sequence[str]] = None,
                        n_splits: int = CV_SPLITS,
                        n_repeats: int = CV_REPEATS,
                        seed: int = SEED,
                        mode: str = SPLIT_MODE) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    """Fix 5: repeated CV so every metric carries a spread.

    Yields (repeat, fold, train_local_idx, val_local_idx) over positions of y.
    Scaffold mode uses StratifiedGroupKFold when available so folds respect
    scaffold groups too.
    """
    y = np.asarray(y).astype(int).reshape(-1)
    n_splits = max(MIN_CV_SPLITS, min(int(n_splits), int(np.bincount(y, minlength=2).min())))
    out: List[Tuple[int, int, np.ndarray, np.ndarray]] = []

    use_groups = (mode == "scaffold") and groups is not None
    sgkf_cls = None
    if use_groups:
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            sgkf_cls = StratifiedGroupKFold
        except Exception:
            sgkf_cls = None

    for rep in range(int(n_repeats)):
        rs = int(seed) + 1000 * rep
        if sgkf_cls is not None:
            try:
                splitter = sgkf_cls(n_splits=n_splits, shuffle=True, random_state=rs)
                folds = list(splitter.split(np.zeros(len(y)), y, groups=list(groups)))
            except Exception:
                splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
                folds = list(splitter.split(np.zeros(len(y)), y))
        else:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
            folds = list(splitter.split(np.zeros(len(y)), y))

        for k, (tr, va) in enumerate(folds):
            out.append((rep, k, np.asarray(tr, dtype=int), np.asarray(va, dtype=int)))
    return out


def save_split_manifest(path: Union[str, Path],
                        mol_df: pd.DataFrame,
                        valid_mask: np.ndarray,
                        train_idx: np.ndarray,
                        test_idx: np.ndarray,
                        y: np.ndarray,
                        mode_used: str) -> pd.DataFrame:
    """Single source of truth for the split, so every ablation arm consumes
    exactly the same molecules. `train_idx`/`test_idx` are positions within the
    valid (kept) subset."""
    kept = mol_df.loc[valid_mask].reset_index(drop=True).copy()
    kept["dataset_idx"] = np.arange(len(kept), dtype=int)
    kept["y"] = np.asarray(y).astype(int)
    kept["split"] = "unused"
    kept.loc[train_idx, "split"] = "train"
    kept.loc[test_idx, "split"] = "test"
    kept["split_mode"] = mode_used
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["dataset_idx", "orig_idx", "SMILES", "canonical_smiles",
            "scaffold", "y", "split", "split_mode"]
    kept[cols].to_csv(path, index=False)
    return kept


def load_split_manifest(path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(path)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def ranking_metrics(y_true, probs) -> Dict[str, float]:
    y = np.asarray(y_true).astype(int).reshape(-1)
    p = np.asarray(probs).astype(float).reshape(-1)
    ok = np.isfinite(p)
    if ok.sum() == 0 or np.unique(y[ok]).size < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y[ok], p[ok])),
        "pr_auc": float(average_precision_score(y[ok], p[ok])),
    }


def optimal_threshold(y_true, probs,
                      grid: np.ndarray = np.linspace(0.05, 0.95, 91)) -> float:
    """F1-optimal threshold, chosen only on out-of-fold training predictions."""
    y = np.asarray(y_true).astype(int).reshape(-1)
    p = np.asarray(probs).astype(float).reshape(-1)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return 0.5
    y, p = y[ok], p[ok]
    best, best_th = -np.inf, 0.5
    for z in grid:
        f1 = f1_score(y, (p >= z).astype(int), zero_division=0)
        if f1 > best:
            best, best_th = f1, float(z)
    return float(best_th)


def threshold_metrics(y_true, probs, threshold: float) -> Dict[str, float]:
    y = np.asarray(y_true).astype(int).reshape(-1)
    p = np.asarray(probs).astype(float).reshape(-1)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if y.size == 0:
        return {k: float("nan") for k in
                ["roc_auc", "pr_auc", "threshold", "f1", "accuracy",
                 "balanced_accuracy", "mcc", "Se", "Sp", "PPV", "NPV",
                 "tn", "fp", "fn", "tp", "n", "prevalence"]}

    pred = (p >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    se = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    sp = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")

    out = ranking_metrics(y, p)
    out.update({
        "threshold": float(threshold),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)) if np.unique(pred).size > 1 else float("nan"),
        "Se": float(se), "Sp": float(sp), "PPV": float(ppv), "NPV": float(npv),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(y.size), "prevalence": float(y.mean()),
    })
    return out


def aggregate_repeat_metrics(rows: Iterable[Dict[str, float]],
                             keys: Sequence[str] = ("roc_auc", "pr_auc", "f1",
                                                    "mcc", "balanced_accuracy",
                                                    "Se", "Sp")) -> Dict[str, float]:
    """mean / sd / n over repeats -- the spread is the point (fix 5)."""
    df = pd.DataFrame(list(rows))
    out: Dict[str, float] = {}
    for k in keys:
        if k not in df.columns:
            out[f"{k}_mean"] = float("nan")
            out[f"{k}_sd"] = float("nan")
            continue
        vals = pd.to_numeric(df[k], errors="coerce").values.astype(float)
        vals = vals[np.isfinite(vals)]
        out[f"{k}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
        out[f"{k}_sd"] = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")
    out["n_repeats"] = int(len(df))
    return out


# -----------------------------------------------------------------------------
# Applicability domain (reported, never used for selection)
# -----------------------------------------------------------------------------
def nn_tanimoto_distance(query_fp, ref_fps: Sequence) -> float:
    if query_fp is None or len(ref_fps) == 0:
        return float("nan")
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, list(ref_fps))
    if len(sims) == 0:
        return float("nan")
    return float(1.0 - max(sims))


def leave_one_out_nn_distances(ref_fps: Sequence) -> np.ndarray:
    out = []
    ref_fps = list(ref_fps)
    for i in range(len(ref_fps)):
        others = ref_fps[:i] + ref_fps[i + 1:]
        out.append(nn_tanimoto_distance(ref_fps[i], others) if others else float("nan"))
    return np.asarray(out, dtype=float)


def build_ood_reference(reference_smiles: Sequence[str],
                        quantile: float = OOD_DISTANCE_QUANTILE) -> Dict[str, object]:
    ref_smiles = [str(s) for s in reference_smiles if isinstance(s, str)]
    ref_fps = [morgan_fp_from_smiles(s) for s in ref_smiles]
    ref_fps = [fp for fp in ref_fps if fp is not None]
    loo = leave_one_out_nn_distances(ref_fps)
    loo = loo[np.isfinite(loo)]
    cutoff = float(np.quantile(loo, quantile)) if loo.size else float("nan")
    return {
        "reference_smiles": ref_smiles,
        "loo_nn_distances": loo.tolist(),
        "cutoff": cutoff,
        "quantile": float(quantile),
        "radius": int(MORGAN_RADIUS),
        "nbits": int(MORGAN_NBITS),
    }


def score_ood_smiles(query_smiles: Sequence[str],
                     ref_smiles: Sequence[str],
                     cutoff: float,
                     radius: int = MORGAN_RADIUS,
                     nbits: int = MORGAN_NBITS) -> pd.DataFrame:
    ref_fps = [morgan_fp_from_smiles(s, radius, nbits) for s in ref_smiles]
    ref_fps = [fp for fp in ref_fps if fp is not None]
    rows = []
    for smi in query_smiles:
        qfp = morgan_fp_from_smiles(smi, radius, nbits)
        dist = nn_tanimoto_distance(qfp, ref_fps)
        rows.append({
            "ood_nn_distance": dist,
            "ood_nn_similarity": float("nan") if not np.isfinite(dist) else float(1.0 - dist),
            "ood_flag": int(np.isfinite(dist) and np.isfinite(cutoff) and dist > cutoff),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def write_json(path: Union[str, Path], obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=default)


def ensure_dirs() -> None:
    for d in (SPLIT_DIR, RESULTS_DIR, CACHE_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)


def describe_config() -> Dict[str, object]:
    return {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "split_mode": SPLIT_MODE,
        "cv_splits": CV_SPLITS,
        "cv_repeats": CV_REPEATS,
        "n_conformers": N_CONFORMERS,
        "morgan_radius": MORGAN_RADIUS,
        "morgan_nbits": MORGAN_NBITS,
        "atom_feature_dim": ATOM_FEATURE_DIM,
        "bond_feature_dim": BOND_FEATURE_DIM,
        "global_feature_names": GLOBAL_FEATURE_NAMES,
    }

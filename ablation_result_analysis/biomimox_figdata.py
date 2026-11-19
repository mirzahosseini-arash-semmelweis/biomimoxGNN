# biomimox_figdata.py
# Single source of truth for every number that appears in the manuscript
# figures. Each figure script imports from here rather than re-deriving counts.
#
# ---------------------------------------------------------------------------
# DATA MODEL (inferred from data_20260611.csv and verified against it)
# ---------------------------------------------------------------------------
# The file mixes two kinds of row, distinguished by `conv`:
#
#   conv == 1   CONVERSION rows. Exactly one per (name, pH, biomim) -- verified
#               no duplicates. `percent` = conversion of the parent compound.
#               No reaction flags are set, and `invivo` is NA (as documented).
#
#   conv == 0   METABOLITE rows, on a FULL grid of
#               (name, metabolite) x pH x biomim. `percent` = abundance of that
#               metabolite under that condition; 0 means not detected there.
#               Reaction flags, `metabolite_gen` and `invivo` are properties of
#               the (name, metabolite) pair and are constant across the grid
#               (verified; one single exception in Bn_ox, resolved by max()).
#
# Therefore:
#   * a metabolite is CATALOGUED if the pair exists at all
#   * a metabolite is OBSERVED under a condition iff percent > 0 there
#   * every catalogued metabolite carries at least one reaction flag (verified)
#
# `metabolite_gen` is known for only 180 of 437 metabolites (values 0-4 and M);
# the rest are NA and are reported as "generation not assigned" rather than
# silently dropped or counted as generation 0.

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, BRICS, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

RDLogger.DisableLog("rdApp.*")

# =============================================================================
# These are NOT derivable from data_20260611.csv, but known from chemists.
# Set the counts to the number of DISTINCT METABOLITES taken to each stage.
WORKFLOW_COUNTS = {
    "hrms_metabolites": 345,
    "nmr_metabolites": 11,
}
WORKFLOW_PLACEHOLDER_FRACTIONS = {
    "hrms_metabolites": 0.50,
    "nmr_metabolites": 0.12,
}

# =============================================================================
# Configuration
# =============================================================================
BASE = Path(__file__).resolve().parent
DATA_FILE = BASE.parent / "raw_data" / "data_20260611.csv"
FDA_FILE = BASE / "fda_drugs.csv"
ABLATION_DIR = BASE.parent / "ablation_results"
ABLATION_TABLE = ABLATION_DIR / "ablation_table.csv"
TABLEDIR = BASE / "figure_tables"                   # numbers behind each panel

RANDOM_STATE = 42
FP_RADIUS = 2
FP_NBITS = 2048
MIN_SCAFFOLD_ATOMS = 5

# Conversion categories. Zero conversion is kept as its own slice by default:
# at pH 2.7 non-biomimetic, 94 of 101 compounds show no conversion at all, and
# folding those into "low" makes the pie chart a statement about how many
# compounds were tested rather than about conversion. Set to False for the
# strict three-category version.
CONVERSION_SPLIT_ZERO = True
CONVERSION_BINS = [0.0, 33.0, 67.0, 100.0]          # low <33, medium 33-67, high >67
CONVERSION_LABELS = ["low", "medium", "high"]

# A metabolite counts as observed when its abundance is above this.
DETECTION_THRESHOLD = 0.0

# Reaction types for the Figure 1 bar panel, in the order specified
# (ascending count at pH 2.7). Recomputing from the data gives a rank
# correlation of 0.98 against this order -- the differences are all near-ties
# (N_ox = 1 vs O_dearyl = 3; Bn_ox / Ar_ox / O_dealk all tie at 7);
# explicit order is used unless set RECOMPUTE_REACTION_ORDER = True.
RX_FOCUS: List[str] = [
    "O_dearyl", "N_ox", "S_ox", "O_dealk", "Bn_ox", "N_dearyl", "Ar_ox",
    "ox_clv", "R_ox", "OH_ox", "deH_ox", "N_dealk", "unk_ox",
]
RECOMPUTE_REACTION_ORDER = False
REACTION_ORDER_REFERENCE_PH = 2.7

# Every reaction flag in the file (used for completeness checks only).
RX_ALL: List[str] = [
    "R_ox", "Ar_ox", "Bn_ox", "unk_ox", "H_prox", "X_exc", "N_ox", "S_ox",
    "OH_ox", "ox_clv", "N_dealk", "N_dearyl", "O_dealk", "O_dearyl", "deH_ox",
    "sp3_sp2", "sp2_sp", "R_Ar", "Q", "uncat", "unk",
]

# Reaction types that have a HotSMARTS definition (Figure 2 structure grid).
RX_HOT_GRID: List[str] = [
    "Ar_ox", "Bn_ox", "deH_ox", "N_dealk", "N_dearyl",
    "O_dealk", "O_dearyl", "OH_ox", "R_ox", "S_ox",
]
# Reaction types for the alluvial panel.
RX_ALLUVIAL: List[str] = ["Bn_ox", "deH_ox", "N_dealk", "R_ox"]

PH_ORDER: List[float] = [2.7, 4.5, 6.5, 7.4]

# Pretty names for axis labels.
RX_PRETTY: Dict[str, str] = {
    "Ar_ox": "aromatic C–H ox.",
    "Bn_ox": "benzylic ox.",
    "deH_ox": "dehydrogenation",
    "N_dealk": "N-dealkylation",
    "N_dearyl": "N-dearylation",
    "N_ox": "N-oxidation",
    "O_dealk": "O-dealkylation",
    "O_dearyl": "O-dearylation",
    "OH_ox": "alcohol ox.",
    "ox_clv": "oxidative cleavage",
    "R_ox": "aliphatic C–H ox.",
    "S_ox": "S-oxidation",
    "unk_ox": "unspecified ox.",
    "H_prox": "H abstraction (prox.)",
    "X_exc": "halogen exchange",
    "sp3_sp2": "sp3→sp2",
    "sp2_sp": "sp2→sp",
    "R_Ar": "alkyl→aryl",
    "Q": "quinone formation",
    "uncat": "uncategorised",
    "unk": "unknown",
}


def rx_label(rx: str, pretty: bool = True) -> str:
    if not pretty:
        return rx
    name = RX_PRETTY.get(rx)
    return f"{rx}" if name is None else f"{rx}"


def rx_label_long(rx: str) -> str:
    name = RX_PRETTY.get(rx, rx)
    return f"{rx}  ({name})" if name != rx else rx


# =============================================================================
# HotSMARTS -- identical patterns and ordering to the modelling scripts
# =============================================================================
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

ASSAY_TO_HOT: Dict[str, str] = {
    "Ar_ox": "hot_Ar_ox", "Bn_ox": "hot_Bn_ox", "deH_ox": "hot_deH_ox",
    "N_dealk": "hot_N_dealk", "N_dearyl": "hot_N_dearyl", "N_ox": "hot_N_ox",
    "O_dealk": "hot_O_dealk", "O_dearyl": "hot_O_dearyl", "OH_ox": "hot_OH_ox",
    "R_ox": "hot_R_ox", "S_ox": "hot_S_ox", "unk_ox": "hot_unk_ox",
    "unk": "hot_unk_ox",
}


def compile_hot_scaffolds() -> Dict[str, List[Chem.Mol]]:
    out: Dict[str, List[Chem.Mol]] = {}
    for name, smarts_like in HOT_SCAFFOLD_SMARTS.items():
        pats = []
        for sm in _as_smarts_list(smarts_like):
            try:
                p = Chem.MolFromSmarts(sm)
            except Exception:
                p = None
            if p is not None:
                pats.append(p)
        if pats:
            out[name] = pats
    return out


HOT_PATTERNS: Dict[str, List[Chem.Mol]] = compile_hot_scaffolds()
HOT_NAMES: List[str] = list(HOT_PATTERNS.keys())


def mol_matches_any(mol: Optional[Chem.Mol], patterns: Sequence[Chem.Mol]) -> bool:
    if mol is None:
        return False
    for p in patterns:
        try:
            if mol.HasSubstructMatch(p):
                return True
        except Exception:
            continue
    return False


def first_hot_match(mol: Chem.Mol, patterns: Sequence[Chem.Mol]
                    ) -> Tuple[List[int], List[int]]:
    """Atom and bond indices of the first matching HotSMARTS pattern, for the
    highlighted structure panel."""
    if mol is None:
        return [], []
    for p in patterns:
        try:
            match = mol.GetSubstructMatch(p)
        except Exception:
            continue
        if not match:
            continue
        atoms = list(match)
        bonds = []
        aset = set(atoms)
        for b in mol.GetBonds():
            if b.GetBeginAtomIdx() in aset and b.GetEndAtomIdx() in aset:
                bonds.append(b.GetIdx())
        return atoms, bonds
    return [], []


# =============================================================================
# Chemistry helpers
# =============================================================================
DESCRIPTORS = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHAcceptors", rdMolDescriptors.CalcNumHBA),
    ("NumHDonors", rdMolDescriptors.CalcNumHBD),
    ("NumRotatableBonds", rdMolDescriptors.CalcNumRotatableBonds),
    ("RingCount", rdMolDescriptors.CalcNumRings),
    ("AromaticRings", rdMolDescriptors.CalcNumAromaticRings),
    ("HeavyAtomCount", rdMolDescriptors.CalcNumHeavyAtoms),
    ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
    ("NumHeteroatoms", rdMolDescriptors.CalcNumHeteroatoms),
    ("NumSaturatedRings", rdMolDescriptors.CalcNumSaturatedRings),
    ("LabuteASA", rdMolDescriptors.CalcLabuteASA),
    ("BalabanJ", Descriptors.BalabanJ),
    ("BertzCT", Descriptors.BertzCT),
]
DESCRIPTOR_NAMES = [n for n, _ in DESCRIPTORS]


def parse_mol(smiles: object) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return mol


def canon(smiles: object) -> Optional[str]:
    m = parse_mol(smiles)
    return Chem.MolToSmiles(m, canonical=True) if m is not None else None


def descriptors_for(mol: Optional[Chem.Mol]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, fn in DESCRIPTORS:
        try:
            v = float(fn(mol)) if mol is not None else np.nan
        except Exception:
            v = np.nan
        out[name] = v if np.isfinite(v) else np.nan
    return out


def ecfp_array(mol: Optional[Chem.Mol], radius: int = FP_RADIUS,
               nbits: int = FP_NBITS) -> np.ndarray:
    from rdkit import DataStructs
    arr = np.zeros((nbits,), dtype=np.int8)
    if mol is None:
        return arr.astype(np.float32)
    gen = GetMorganGenerator(radius=radius, fpSize=nbits)
    DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
    return arr.astype(np.float32)


def murcko_smiles(mol: Optional[Chem.Mol]) -> Optional[str]:
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        smi = Chem.MolToSmiles(scaf, canonical=True)
        return smi or None
    except Exception:
        return None


def ring_system_smiles(mol: Optional[Chem.Mol],
                       min_atoms: int = MIN_SCAFFOLD_ATOMS) -> List[str]:
    """Fused ring systems as separate scaffolds."""
    if mol is None:
        return []
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    if not rings:
        return []
    merged: List[set] = []
    for r in rings:
        hit = [m for m in merged if m & r]
        if not hit:
            merged.append(set(r))
        else:
            new = set(r)
            for m in hit:
                new |= m
                merged.remove(m)
            merged.append(new)
    out: List[str] = []
    for atoms in merged:
        if len(atoms) < min_atoms:
            continue
        try:
            smi = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms),
                                           canonical=True)
            if smi:
                out.append(smi)
        except Exception:
            continue
    return out


def brics_fragments(mol: Optional[Chem.Mol],
                    min_atoms: int = MIN_SCAFFOLD_ATOMS) -> List[str]:
    if mol is None:
        return []
    try:
        frags = BRICS.BRICSDecompose(mol, returnMols=False, minFragmentSize=min_atoms)
    except Exception:
        return []
    out = []
    for f in frags or []:
        clean = Chem.MolFromSmiles(
            "".join(ch for ch in str(f)).replace("[*]", "").replace("*", ""))
        if clean is None:
            continue
        if clean.GetNumHeavyAtoms() < min_atoms:
            continue
        smi = Chem.MolToSmiles(clean, canonical=True)
        if smi:
            out.append(smi)
    return out


# =============================================================================
# The bundle
# =============================================================================
@dataclass
class FigData:
    """Every table the figures draw from."""
    raw: pd.DataFrame
    parents: pd.DataFrame            # one row per distinct parent compound
    metabolites: pd.DataFrame        # one row per distinct (name, metabolite)
    metab_grid: pd.DataFrame         # (name, metabolite) x pH x biomim
    conversions: pd.DataFrame        # conv == 1 rows, with category
    rx_focus: List[str] = field(default_factory=lambda: list(RX_FOCUS))
    notes: List[str] = field(default_factory=list)

    # -- headline counts -----------------------------------------------------
    @property
    def n_parents(self) -> int:
        return int(self.parents["parent_id"].nunique())

    @property
    def n_parent_names(self) -> int:
        return int(self.raw["name"].nunique())

    @property
    def n_metabolites(self) -> int:
        return int(len(self.metabolites))

    @property
    def n_metabolites_observed(self) -> int:
        obs = self.metab_grid[self.metab_grid["observed"]]
        return int(obs.groupby(["name", "metabolite"]).ngroups)

    def workflow_counts(self) -> Tuple[Dict[str, int], bool]:
        """(counts, is_placeholder) for the parents -> metabolites -> HRMS -> NMR
        funnel. Falls back to fractions of the metabolite count when the real
        numbers have not been supplied."""
        n_met = self.n_metabolites
        placeholder = False
        out = {"parents": self.n_parents, "metabolites": n_met}
        for key in ("hrms_metabolites", "nmr_metabolites"):
            val = WORKFLOW_COUNTS.get(key)
            if val is None:
                placeholder = True
                val = int(round(WORKFLOW_PLACEHOLDER_FRACTIONS[key] * n_met))
            out[key] = int(val)
        return out, placeholder

    # -- derived views -------------------------------------------------------
    def metabolites_per_parent(self, biomim: Optional[int] = None) -> pd.DataFrame:
        """Distinct observed metabolites per parent compound, per pH (and per
        biomim unless one is selected). Parents with zero observed metabolites
        are included as zeros -- dropping them would inflate the medians."""
        g = self.metab_grid[self.metab_grid["observed"]]
        if biomim is not None:
            g = g[g["biomim"] == biomim]
        counted = (g.groupby(["name", "pH", "biomim"])["metabolite"]
                   .nunique().rename("n_metabolites").reset_index())

        # full scaffold of every (parent, pH, biomim) actually tested
        tested = (self.conversions[["name", "pH", "biomim"]].drop_duplicates())
        if biomim is not None:
            tested = tested[tested["biomim"] == biomim]
        out = tested.merge(counted, on=["name", "pH", "biomim"], how="left")
        out["n_metabolites"] = out["n_metabolites"].fillna(0).astype(int)
        return out

    def conversion_categories(self) -> pd.DataFrame:
        """Counts per (pH, biomim, category)."""
        c = self.conversions
        rows = []
        cats = (["none"] if CONVERSION_SPLIT_ZERO else []) + CONVERSION_LABELS
        for (ph, bm), sub in c.groupby(["pH", "biomim"]):
            vc = sub["conv_category"].value_counts()
            for cat in cats:
                rows.append({"pH": ph, "biomim": int(bm), "category": cat,
                             "n": int(vc.get(cat, 0)), "n_total": int(len(sub))})
        return pd.DataFrame(rows)

    def reaction_counts_by_ph(self, biomim: Optional[int] = None,
                              observed_only: bool = True,
                              unit: str = "metabolite") -> pd.DataFrame:
        """Reaction-type counts per pH.

        unit="metabolite": distinct (name, metabolite) pairs carrying the flag
        unit="parent":     distinct parent compounds showing the reaction
        """
        g = self.metab_grid
        if observed_only:
            g = g[g["observed"]]
        if biomim is not None:
            g = g[g["biomim"] == biomim]
        key = ["name", "metabolite"] if unit == "metabolite" else ["name"]
        rows = []
        for ph in PH_ORDER:
            sub = g[g["pH"] == ph]
            if sub.empty:
                for rx in self.rx_focus:
                    rows.append({"pH": ph, "reaction": rx, "n": 0})
                continue
            per = sub.groupby(key)[self.rx_focus].max()
            for rx in self.rx_focus:
                rows.append({"pH": ph, "reaction": rx, "n": int(per[rx].sum())})
        return pd.DataFrame(rows)

    def expected_vs_observed(self) -> pd.DataFrame:
        """Per reaction type, at parent-compound level:

        expected  -- the parent matches the corresponding HotSMARTS pattern
        observed  -- the parent actually yielded a metabolite with that flag
        both      -- expected and observed (true positive of the SMARTS rule)
        unexpected-- observed without the pattern matching
        missed    -- pattern matched but never observed
        """
        obs = self.metab_grid[self.metab_grid["observed"]]
        observed_by_parent = obs.groupby("name")[self.rx_focus].max()
        rows = []
        for rx in self.rx_focus:
            hot = ASSAY_TO_HOT.get(rx)
            if hot is None or hot not in self.parents.columns:
                exp_names = set()
            else:
                exp_names = set(self.parents.loc[self.parents[hot] == 1, "name"])
            if rx in observed_by_parent.columns:
                obs_names = set(observed_by_parent.index[observed_by_parent[rx] == 1])
            else:
                obs_names = set()
            all_names = set(self.parents["name"])
            rows.append({
                "reaction": rx,
                "has_hot_pattern": hot is not None and hot in self.parents.columns,
                "n_expected": len(exp_names),
                "n_observed": len(obs_names),
                "n_both": len(exp_names & obs_names),
                "n_missed": len(exp_names - obs_names),
                "n_unexpected": len(obs_names - exp_names),
                "n_parents": len(all_names),
            })
        out = pd.DataFrame(rows)
        out["recall_of_observed"] = np.where(
            out["n_observed"] > 0, out["n_both"] / out["n_observed"], np.nan)
        out["precision_of_pattern"] = np.where(
            out["n_expected"] > 0, out["n_both"] / out["n_expected"], np.nan)
        return out

    def invivo_summary(self) -> pd.DataFrame:
        """In vivo status of distinct observed metabolites, per reaction type."""
        obs_pairs = (self.metab_grid[self.metab_grid["observed"]]
                     [["name", "metabolite"]].drop_duplicates())
        m = self.metabolites.merge(obs_pairs, on=["name", "metabolite"], how="inner")
        rows = []
        for rx in self.rx_focus:
            sub = m[m[rx] == 1]
            total = len(sub)
            for code in (1, 2, 0):
                rows.append({"reaction": rx, "invivo": code,
                             "n": int((sub["invivo"] == code).sum()),
                             "n_total": total})
            rows.append({"reaction": rx, "invivo": -1,
                         "n": int(sub["invivo"].isna().sum()), "n_total": total})
        return pd.DataFrame(rows)

    def generation_summary(self) -> pd.DataFrame:
        """Observed metabolites by generation x in vivo status."""
        obs_pairs = (self.metab_grid[self.metab_grid["observed"]]
                     [["name", "metabolite"]].drop_duplicates())
        m = self.metabolites.merge(obs_pairs, on=["name", "metabolite"], how="inner")
        m = m.copy()
        m["gen_label"] = m["metabolite_gen"].apply(_gen_label)
        rows = []
        for gen in GEN_ORDER:
            sub = m[m["gen_label"] == gen]
            for code in (1, 2, 0):
                rows.append({"generation": gen, "invivo": code,
                             "n": int((sub["invivo"] == code).sum())})
            rows.append({"generation": gen, "invivo": -1,
                         "n": int(sub["invivo"].isna().sum())})
        return pd.DataFrame(rows)


GEN_ORDER = ["1", "2", "3", "4", "M", "not assigned"]


def _gen_label(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "not assigned"
    s = str(v).strip()
    if s in ("", "nan", "None", "<NA>"):
        return "not assigned"
    if s in ("0",):
        return "not assigned"
    if s in GEN_ORDER:
        return s
    try:
        f = float(s)
        if f.is_integer():
            si = str(int(f))
            if si == "0":
                return "not assigned"
            if si in GEN_ORDER:
                return si
    except Exception:
        pass
    return "not assigned"


# =============================================================================
# Loading
# =============================================================================
def load(data_file: Union[str, Path] = DATA_FILE,
         verbose: bool = True) -> FigData:
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(
            f"{data_file} not found. Put data_20260611.csv next to these scripts, "
            f"or pass --data /path/to/data.csv")

    # utf-8-sig: the file carries a BOM, which otherwise corrupts the 'name' column
    df = pd.read_csv(data_file, encoding="utf-8-sig")
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    df.columns = [c.strip() for c in df.columns]

    required = {"name", "smiles", "pH", "biomim", "metabolite", "metabolite_gen",
                "invivo", "conv", "percent"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for c in ["pH", "biomim", "invivo", "conv", "percent"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    rx_present = [c for c in RX_ALL if c in df.columns]
    for c in rx_present:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    notes: List[str] = []

    # ---- parents --------------------------------------------------------
    par = (df[["name", "smiles"]].drop_duplicates()
           .reset_index(drop=True).copy())
    par["mol"] = par["smiles"].apply(parse_mol)
    bad = int(par["mol"].isna().sum())
    if bad:
        notes.append(f"{bad} parent SMILES could not be parsed and were dropped")
        par = par[par["mol"].notna()].copy()
    par["parent_id"] = par["mol"].apply(lambda m: Chem.MolToSmiles(m, canonical=True))

    # One SMILES is shared by two names in this file; keep both names (they are
    # distinct study entries) but flag it, because "number of distinct
    # compounds" then depends on whether you count names or structures.
    shared = par.groupby("parent_id")["name"].nunique()
    shared = shared[shared > 1]
    if len(shared):
        dup_names = (par[par["parent_id"].isin(shared.index)]
                     .groupby("parent_id")["name"].apply(list).to_dict())
        notes.append(
            f"{len(shared)} structure(s) appear under more than one name "
            f"({dup_names}); n_parents counts distinct structures "
            f"({par['parent_id'].nunique()}), n_parent_names counts names "
            f"({df['name'].nunique()})")

    desc = par["mol"].apply(descriptors_for).apply(pd.Series)
    par = pd.concat([par.reset_index(drop=True), desc.reset_index(drop=True)], axis=1)
    for hot, pats in HOT_PATTERNS.items():
        par[hot] = par["mol"].apply(lambda m, p=pats: int(mol_matches_any(m, p)))
    par["murcko"] = par["mol"].apply(murcko_smiles)
    par["n_hot_patterns"] = par[[h for h in HOT_PATTERNS]].sum(axis=1)

    # ---- conversion rows -------------------------------------------------
    conv = df[df["conv"] == 1].copy()
    if conv.empty:
        raise ValueError("No conv == 1 rows found; cannot build conversion panels")
    dup = conv.duplicated(["name", "pH", "biomim"]).sum()
    if dup:
        notes.append(f"{dup} duplicated (name, pH, biomim) conversion rows; "
                     f"kept the first of each")
        conv = conv.drop_duplicates(["name", "pH", "biomim"], keep="first")
    conv["conv_category"] = conv["percent"].apply(_conv_category)

    # ---- metabolite rows -------------------------------------------------
    met = df[df["conv"] == 0].copy()
    met["observed"] = met["percent"] > DETECTION_THRESHOLD

    inconsistent = []
    for col in rx_present + ["invivo"]:
        n = met.groupby(["name", "metabolite"])[col].nunique(dropna=False)
        if int((n > 1).sum()):
            inconsistent.append((col, int((n > 1).sum())))
    if inconsistent:
        notes.append(
            "per-metabolite annotation is not constant across the pH/biomim grid "
            f"for {inconsistent}; resolved with max() for flags and first "
            "non-null for in vivo status")

    agg = {c: "max" for c in rx_present}
    met_unique = (met.groupby(["name", "metabolite"], as_index=False)
                  .agg(**{c: (c, "max") for c in rx_present},
                       metabolite_gen=("metabolite_gen",
                                       lambda s: s.dropna().iloc[0]
                                       if s.notna().any() else np.nan),
                       invivo=("invivo", lambda s: s.dropna().iloc[0]
                               if s.notna().any() else np.nan),
                       smiles=("smiles", "first"),
                       n_grid_rows=("percent", "size"),
                       max_percent=("percent", "max")))

    zero_flag = met_unique[rx_present].sum(axis=1) == 0
    if int(zero_flag.sum()):
        notes.append(f"{int(zero_flag.sum())} metabolites carry no reaction flag")

    grid_cols = ["name", "metabolite", "pH", "biomim", "percent", "observed"]
    metab_grid = met[grid_cols + rx_present].copy()

    fd = FigData(raw=df, parents=par, metabolites=met_unique,
                 metab_grid=metab_grid, conversions=conv, notes=notes)

    if RECOMPUTE_REACTION_ORDER:
        counts = fd.reaction_counts_by_ph(biomim=None, observed_only=True)
        ref = (counts[counts["pH"] == REACTION_ORDER_REFERENCE_PH]
               .sort_values("n", kind="stable"))
        fd.rx_focus = ref["reaction"].tolist()
        notes.append(f"reaction order recomputed at pH "
                     f"{REACTION_ORDER_REFERENCE_PH}: {fd.rx_focus}")

    missing_rx = [r for r in fd.rx_focus if r not in df.columns]
    if missing_rx:
        notes.append(f"requested reaction types absent from the file: {missing_rx}")
        fd.rx_focus = [r for r in fd.rx_focus if r in df.columns]

    if verbose:
        print(f"Loaded {data_file.name}: {len(df):,} rows")
        print(f"  parent structures      {fd.n_parents}  "
              f"(names: {fd.n_parent_names})")
        print(f"  distinct metabolites   {fd.n_metabolites}  "
              f"(ever observed: {fd.n_metabolites_observed})")
        print(f"  conversion rows        {len(conv):,}")
        print(f"  pH values              {sorted(df['pH'].dropna().unique())}")
        for n in notes:
            print(f"  note: {n}")

    return fd


def _conv_category(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "none"
    if CONVERSION_SPLIT_ZERO and p <= 0:
        return "none"
    if p < CONVERSION_BINS[1]:
        return "low"
    if p <= CONVERSION_BINS[2]:
        return "medium"
    return "high"


# =============================================================================
# Ablation results
# =============================================================================
ARM_PRETTY: Dict[str, str] = {
    "0_prior": "class prior",
    "1_hot_logreg": "HotSMARTS + logistic",
    "2a_ecfp4_logreg": "ECFP4 + logistic",
    "2b_ecfp4_rf": "ECFP4 + RF",
    "3_ecfp4_desc_rf": "ECFP4 + desc. + RF",
    "4_all_rf": "ECFP4 + desc. + HotSMARTS + RF",
    "5_gnn_dimenetpp": "DimeNet++ (single-task)",
    "6_multitask_dimenetpp": "DimeNet++ (multitask)",
    "5_gnn_schnet": "SchNet (single-task)",
    "6_multitask_schnet": "SchNet (multitask)",
    "5_gnn_gine2d": "GINE 2-D (single-task)",
    "6_multitask_gine2d": "GINE 2-D (multitask)",
}
ARM_FAMILY: Dict[str, str] = {
    "0_prior": "prior",
    "1_hot_logreg": "smarts",
    "2a_ecfp4_logreg": "fingerprint",
    "2b_ecfp4_rf": "fingerprint",
    "3_ecfp4_desc_rf": "fingerprint",
    "4_all_rf": "combined",
}
# Repeats behind the SD: the baselines report all CV repeats, the neural arms
# report the repeats not used for configuration selection (CV_REPEATS - 1).
DEFAULT_REPEATS = {"baseline": 5, "gnn_single": 4, "gnn_multitask": 4}


def load_ablation(table: Union[str, Path] = ABLATION_TABLE,
                  ablation_dir: Union[str, Path] = ABLATION_DIR,
                  verbose: bool = True) -> pd.DataFrame:
    """Tidy ablation table with confidence intervals and prevalence-adjusted lift.

    PR-AUC is not comparable across assays here -- prevalence spans 0.024 to
    0.378 -- so `lift` (PR-AUC minus prevalence, the gain over the class prior)
    is computed alongside it. Within one assay the ranking is identical; across
    assays only the lift is meaningful.
    """
    table = Path(table)
    if not table.exists():
        raise FileNotFoundError(f"{table} not found; run biomimox_compare.py first")
    t = pd.read_csv(table)

    # Repeat counts live in the per-arm summary files, not the tidy table.
    reps: Dict[Tuple[str, str], float] = {}
    for pattern in ("baselines_summary.csv", "gnn_summary*.csv",
                    "multitask_summary.csv"):
        for path in sorted(Path(ablation_dir).glob(pattern)):
            try:
                s = pd.read_csv(path)
            except Exception:
                continue
            if "cv_n_repeats" in s.columns:
                for _, r in s.iterrows():
                    reps[(str(r.get("target")), str(r.get("arm")))] = \
                        float(r["cv_n_repeats"])

    def n_rep(row) -> float:
        got = reps.get((str(row["target"]), str(row["arm"])))
        if got and np.isfinite(got) and got > 0:
            return got
        fam = str(row.get("family", "baseline"))
        return float(DEFAULT_REPEATS.get(fam, 5))

    t["n_repeats"] = t.apply(n_rep, axis=1)
    t["arm_pretty"] = t["arm"].map(lambda a: ARM_PRETTY.get(a, a))
    t["family_group"] = t.apply(
        lambda r: ARM_FAMILY.get(str(r["arm"]), str(r.get("family", "other"))), axis=1)

    # Normal-approximation CI on the repeat mean. With 4-5 repeats a t
    # multiplier is the honest choice, so use it.
    from scipy import stats as _st
    sd = pd.to_numeric(t.get("cv_pr_auc_sd"), errors="coerce")
    n = t["n_repeats"].clip(lower=1)
    sem = sd / np.sqrt(n)
    tcrit = np.where(n > 1, _st.t.ppf(0.975, np.maximum(n - 1, 1)), np.nan)
    t["cv_pr_auc_sem"] = sem
    t["cv_pr_auc_ci"] = tcrit * sem
    t["cv_pr_auc_lo"] = t["cv_pr_auc_mean"] - t["cv_pr_auc_ci"]
    t["cv_pr_auc_hi"] = t["cv_pr_auc_mean"] + t["cv_pr_auc_ci"]
    t["lift"] = t["cv_pr_auc_mean"] - t["prevalence"]
    t["lift_lo"] = t["cv_pr_auc_lo"] - t["prevalence"]
    t["lift_hi"] = t["cv_pr_auc_hi"] - t["prevalence"]

    if verbose:
        print(f"Loaded {table.name}: {len(t)} assay x arm rows, "
              f"{t['target'].nunique()} assays, {t['arm'].nunique()} arms")
        if not reps:
            print(f"  note: no *_summary.csv with cv_n_repeats in {ablation_dir}; "
                  f"CIs use assumed repeat counts {DEFAULT_REPEATS}")
    return t


# =============================================================================
# FDA reference set
# =============================================================================
def load_fda(fda_file: Union[str, Path] = FDA_FILE,
             verbose: bool = True) -> Optional[pd.DataFrame]:
    fda_file = Path(fda_file)
    if not fda_file.exists():
        if verbose:
            print(f"  note: {fda_file.name} not found -- the chemical-space panel "
                  f"will show only the study compounds. Add a CSV with 'name' and "
                  f"'smiles' columns to overlay the FDA reference set.")
        return None
    fda = pd.read_csv(fda_file)
    fda = fda.loc[:, ~fda.columns.str.match(r"^Unnamed")]
    fda.columns = [c.strip().lower() for c in fda.columns]
    if not {"name", "smiles"}.issubset(fda.columns):
        warnings.warn(f"{fda_file.name} needs 'name' and 'smiles' columns; ignoring")
        return None
    fda["mol"] = fda["smiles"].apply(parse_mol)
    fda = fda[fda["mol"].notna()].copy()
    fda["parent_id"] = fda["mol"].apply(lambda m: Chem.MolToSmiles(m, canonical=True))
    fda = fda.drop_duplicates("parent_id").reset_index(drop=True)
    if verbose:
        print(f"  FDA reference set: {len(fda)} parsed structures")
    return fda


# =============================================================================
# Table export
# =============================================================================
def export_table(df: pd.DataFrame, stem: str, outdir: Union[str, Path] = TABLEDIR,
                 verbose: bool = True) -> Path:
    """Write the numbers behind a panel, so every figure value is traceable."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{stem}.csv"
    df.to_csv(path, index=False)
    if verbose:
        print(f"    table {path}")
    return path

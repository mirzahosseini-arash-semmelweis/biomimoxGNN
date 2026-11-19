# figstyle.py
# Shared figure styles.
#
# One place to change font sizes, palette and output format so all five
# figures stay visually consistent. Nothing here touches the data.

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
FIGDIR = Path("figures")
DPI_RASTER = 600                # for PNG/TIFF
SAVE_FORMATS = ("pdf", "png")

# Single/double column widths in inches
COL1 = 3.54
COL2 = 7.28
COL2_WIDE = 9.00            # for the few panels that genuinely need it

# -----------------------------------------------------------------------------
# Palette
# -----------------------------------------------------------------------------
# Colour-blind-safe (Okabe-Ito derived), consistent across every figure.
C_BLACK = "#000000"
C_ORANGE = "#E69F00"
C_SKY = "#56B4E9"
C_GREEN = "#009E73"
C_YELLOW = "#F0E442"
C_BLUE = "#0072B2"
C_VERM = "#D55E00"
C_PURPLE = "#CC79A7"
C_GREY = "#555555"
C_LGREY = "#999999"
C_LGREY2 = "#D9D9D9"

# Fixed semantic colours.
COLOR_BIOMIM = {0: C_GREY, 1: C_BLUE}
LABEL_BIOMIM = {0: "non-biomimetic", 1: "biomimetic"}

# pH is ordinal -> sequential ramp, not categorical hues
COLOR_PH = {2.7: "#8A6D00", 4.5: "#FDAE61", 6.5: "#3288BD", 7.4: "#1A4F7A"}

COLOR_CONV = {
    "none": C_LGREY2,
    "low": C_YELLOW,
    "medium": C_ORANGE,
    "high": C_VERM,
}

COLOR_INVIVO = {
    0: C_LGREY,       # no data
    2: C_SKY,         # probably in vivo, structure unknown
    1: C_GREEN,       # occurs in vivo
}
LABEL_INVIVO = {
    0: "no in vivo data",
    2: "probable in vivo",
    1: "confirmed in vivo",
}

COLOR_EXPECTED = C_GREY
COLOR_OBSERVED = C_BLUE

# Arm families for the ablation figure
COLOR_FAMILY = {
    "prior": C_LGREY,
    "smarts": C_PURPLE,
    "fingerprint": C_ORANGE,
    "combined": C_VERM,
    "gnn_single": C_BLUE,
    "gnn_multitask": C_SKY,
}

# Per-arm colours. The three fingerprint arms share a family but need to be
# told apart inside a crowded panel, so they get a light-to-dark ramp of the
# same hue rather than one flat colour.
COLOR_ARM = {
    "0_prior": C_LGREY,
    "1_hot_logreg": C_PURPLE,
    "2a_ecfp4_logreg": "#F6C445",
    "2b_ecfp4_rf": "#E69F00",
    "3_ecfp4_desc_rf": "#B5750B",
    "4_all_rf": C_VERM,
    "5_gnn_dimenetpp": C_BLUE,
    "5_gnn_schnet": "#1F6FA8",
    "5_gnn_gine2d": "#2E86C1",
    "6_multitask_dimenetpp": C_SKY,
    "6_multitask_schnet": "#8DCBF0",
    "6_multitask_gine2d": "#A6D8F5",
}


def arm_colour(arm: str, family_group: str = "other") -> str:
    if arm in COLOR_ARM:
        return COLOR_ARM[arm]
    if str(arm).startswith("5_"):
        return C_BLUE
    if str(arm).startswith("6_"):
        return C_SKY
    return COLOR_FAMILY.get(family_group, C_GREY)


def diverging_cmap():
    return plt.get_cmap("RdBu_r")


def sequential_cmap():
    return plt.get_cmap("viridis")


# -----------------------------------------------------------------------------
# rcParams
# -----------------------------------------------------------------------------
def apply_style(base_fontsize: float = 7.0) -> None:
    """Defaults: small sans-serif type, thin spines, no chartjunk.

    Fonts are embedded as TrueType (Type 42).
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"):
        if candidate in available:
            family = candidate
            break
    else:
        family = "sans-serif"

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": DPI_RASTER,
        "savefig.bbox": "standard",     # 'tight' fights fixed panel geometry
        "savefig.transparent": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        "font.family": "sans-serif",
        "font.sans-serif": [family, "DejaVu Sans"],
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 0.5,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize - 0.5,
        "ytick.labelsize": base_fontsize - 0.5,
        "legend.fontsize": base_fontsize - 0.5,
        "figure.titlesize": base_fontsize + 1.5,

        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "axes.titlepad": 3.0,
        "axes.labelpad": 2.0,

        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "patch.linewidth": 0.5,

        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.2,

        "grid.linewidth": 0.4,
        "grid.color": C_LGREY,
        "grid.alpha": 0.8,

        "errorbar.capsize": 1.5,
    })


# -----------------------------------------------------------------------------
# Panel helpers
# -----------------------------------------------------------------------------
def panel_label(ax, letter: str, dx: float = -0.085, dy: float = 1.045,
                fontsize: Optional[float] = None) -> None:
    """Bold lower-case panel letter, positioned in axes coordinates."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=(fontsize or plt.rcParams["font.size"] + 2.5),
            fontweight="bold", va="bottom", ha="left")


def panel_label_fig(fig, letter: str, x: float, y: float,
                    fontsize: Optional[float] = None) -> None:
    """Panel letter in figure coordinates, for panels built from sub-gridspecs
    where no single axes is the natural anchor."""
    fig.text(x, y, letter,
             fontsize=(fontsize or plt.rcParams["font.size"] + 2.5),
             fontweight="bold", va="bottom", ha="left")


def despine(ax, left: bool = False, bottom: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False, labelleft=False)
    if bottom:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False)


def hide(ax) -> None:
    ax.set_axis_off()


def thousands(ax, axis: str = "y") -> None:
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f"{int(v):,}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def annotate_bars(ax, bars, values: Optional[Sequence] = None,
                  fmt: str = "{:.0f}", pad: float = 1.0,
                  horizontal: bool = False, fontsize: Optional[float] = None,
                  skip_zero: bool = True) -> None:
    vals = list(values) if values is not None else [
        (b.get_width() if horizontal else b.get_height()) for b in bars]
    fs = fontsize or plt.rcParams["font.size"] - 1.0
    for b, v in zip(bars, vals):
        if skip_zero and (v is None or (isinstance(v, (int, float)) and v == 0)):
            continue
        if horizontal:
            ax.text(b.get_width() + pad, b.get_y() + b.get_height() / 2,
                    fmt.format(v), va="center", ha="left", fontsize=fs)
        else:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + pad,
                    fmt.format(v), ha="center", va="bottom", fontsize=fs)


def legend_patches(mapping: Dict, labels: Optional[Dict] = None) -> List[Patch]:
    out = []
    for key, colour in mapping.items():
        lab = labels.get(key, str(key)) if labels else str(key)
        out.append(Patch(facecolor=colour, edgecolor="none", label=lab))
    return out


def legend_markers(mapping: Dict, labels: Optional[Dict] = None,
                   marker: str = "o", markersize: float = 3.5) -> List[Line2D]:
    out = []
    for key, colour in mapping.items():
        lab = labels.get(key, str(key)) if labels else str(key)
        out.append(Line2D([0], [0], marker=marker, color="none",
                          markerfacecolor=colour, markeredgecolor="none",
                          markersize=markersize, label=lab))
    return out


def sig_stars(p: float) -> str:
    if p is None or not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return ""


def wrap_label(s: str, width: int = 18) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(str(s), width=width)) or str(s)


# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------
def save(fig, stem: str, outdir: Path = FIGDIR,
         formats: Iterable[str] = SAVE_FORMATS, verbose: bool = True) -> List[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for ext in formats:
        path = outdir / f"{stem}.{ext}"
        fig.savefig(path, format=ext)
        written.append(path)
        if verbose:
            size = path.stat().st_size / 1024.0
            print(f"    wrote {path}  ({size:,.0f} KB)")
    plt.close(fig)
    return written


# -----------------------------------------------------------------------------
# RDKit structure rendering into matplotlib axes
# -----------------------------------------------------------------------------
def mol_to_image(mol, width: int = 500, height: int = 380,
                 highlight_atoms: Optional[Sequence[int]] = None,
                 highlight_bonds: Optional[Sequence[int]] = None,
                 highlight_colour: Tuple[float, float, float] = (0.95, 0.55, 0.15),
                 scale_to_fit: bool = True):
    """Render an RDKit mol to an RGBA array suitable for ax.imshow().

    Highlighting is used for the HotSMARTS panel: the matched atoms and bonds
    are drawn in `highlight_colour` over an otherwise black-and-white skeleton.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    from PIL import Image

    if mol is None:
        return None
    mol = Chem.Mol(mol)
    try:
        Chem.rdmolops.Kekulize(mol, clearAromaticFlags=False)
    except Exception:
        pass
    if mol.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(mol)
    rdDepictor.StraightenDepiction(mol)

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 1
    opts.multipleBondOffset = 0.14
    opts.fixedBondLength = -1
    opts.additionalAtomLabelPadding = 0.06
    opts.highlightRadius = 0.32
    if scale_to_fit:
        opts.padding = 0.06

    hl_atoms = list(highlight_atoms or [])
    hl_bonds = list(highlight_bonds or [])
    atom_cols = {i: highlight_colour for i in hl_atoms}
    bond_cols = {i: highlight_colour for i in hl_bonds}

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=hl_atoms or None,
        highlightBonds=hl_bonds or None,
        highlightAtomColors=atom_cols or None,
        highlightBondColors=bond_cols or None,
    )
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(png)).convert("RGBA")
    return np.asarray(img)


def draw_mol_on_ax(ax, mol, title: Optional[str] = None,
                   highlight_atoms: Optional[Sequence[int]] = None,
                   highlight_bonds: Optional[Sequence[int]] = None,
                   highlight_colour: Tuple[float, float, float] = (0.95, 0.55, 0.15),
                   subtitle: Optional[str] = None,
                   width: int = 500, height: int = 380,
                   title_fontsize: Optional[float] = None) -> None:
    arr = mol_to_image(mol, width=width, height=height,
                       highlight_atoms=highlight_atoms,
                       highlight_bonds=highlight_bonds,
                       highlight_colour=highlight_colour)
    ax.set_axis_off()
    if arr is None:
        ax.text(0.5, 0.5, "structure\nunavailable", ha="center", va="center",
                fontsize=plt.rcParams["font.size"] - 1, color=C_GREY,
                transform=ax.transAxes)
        return
    ax.imshow(arr, interpolation="lanczos")
    if title:
        ax.set_title(title, fontsize=(title_fontsize or plt.rcParams["font.size"]),
                     loc="center", pad=1.5)
    if subtitle:
        # Placed inside the canvas: RDKit pads its drawing, so putting the
        # caption below the axes leaves a visible gap under the structure.
        ax.text(0.5, 0.015, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=plt.rcParams["font.size"] - 1.5,
                color=C_GREY, clip_on=False)

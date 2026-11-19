# fig2_chemspace.py
# Figure 2 -- chemical space, HotSMARTS patterns, scaffold diversity and the
# relationship between an expected pattern and an observed reaction.
#
#   (a) UMAP of ECFP4, study compounds over an FDA reference set
#   (b) reaction types per compound: expected (HotSMARTS) vs observed
#   (c) one representative molecule per reaction type, matched atoms highlighted
#   (d) the five most frequent scaffolds, with scaffold-diversity statistics
#   (e) alluvial: HotSMARTS present/absent -> reaction observed/not observed
#

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Patch

from rdkit import Chem

import biomimox_figdata as D
import figstyle as S


# -----------------------------------------------------------------------------
# (a) chemical space
# -----------------------------------------------------------------------------
def panel_umap(ax, fd: D.FigData, fda: Optional[pd.DataFrame],
               random_state: int = D.RANDOM_STATE) -> pd.DataFrame:
    from sklearn.decomposition import PCA

    study_fp = np.vstack([D.ecfp_array(m) for m in fd.parents["mol"]])
    labels = ["study"] * len(study_fp)
    X = study_fp

    if fda is not None and len(fda):
        fda_fp = np.vstack([D.ecfp_array(m) for m in fda["mol"]])
        X = np.vstack([X, fda_fp])
        labels += ["fda"] * len(fda_fp)
    labels = np.asarray(labels)

    # PCA first (standard practice: denoises the sparse bit vectors and makes
    # UMAP both faster and more stable), then 2-D embedding.
    n_comp = int(min(50, X.shape[0] - 1, X.shape[1]))
    Xp = PCA(n_components=n_comp, random_state=random_state).fit_transform(
        X.astype(float))

    method = "PCA"
    try:
        import umap
        if X.shape[0] >= 15:
            reducer = umap.UMAP(n_neighbors=min(25, max(5, X.shape[0] // 6)),
                                min_dist=0.15, metric="euclidean",
                                random_state=random_state)
            xy = reducer.fit_transform(Xp)
            method = "UMAP"
        else:
            xy = Xp[:, :2]
    except Exception as exc:
        print(f"    note: UMAP unavailable ({type(exc).__name__}); using PCA")
        xy = Xp[:, :2]

    if fda is not None and len(fda):
        m = labels == "fda"
        ax.scatter(xy[m, 0], xy[m, 1], s=4, c=S.C_LGREY2, linewidths=0,
                   alpha=0.75, zorder=1, rasterized=True, label="FDA drugs")
    m = labels == "study"
    ax.scatter(xy[m, 0], xy[m, 1], s=11, c=S.C_BLUE, linewidths=0.3,
               edgecolors="white", alpha=0.95, zorder=3,
               label="study compounds")

    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_xticks([])
    ax.set_yticks([])

    qx1, qx2 = np.percentile(xy[:, 0], [1, 99])
    qy1, qy2 = np.percentile(xy[:, 1], [1, 99])
    dx, dy = (qx2 - qx1) * 0.05, (qy2 - qy1) * 0.05
    ax.set_xlim(qx1 - dx, qx2 + dx)
    ax.set_ylim(qy1 - dy, qy2 + dy)

    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(True)
    ax.legend(loc="upper right", markerscale=1.4,
              bbox_to_anchor=(-0.02, 1.02))
    title = "Chemical space (ECFP4)"
    if fda is None:
        ax.set_title(title, loc="left")
        ax.text(0.5, -0.09, "fda_drugs.csv not found — no reference overlay",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=plt.rcParams["font.size"] - 1.5, color=S.C_VERM)
    else:
        ax.set_title(title, loc="left")

    out = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "set": labels})
    out.loc[out["set"] == "study", "name"] = fd.parents["name"].values
    return out


# -----------------------------------------------------------------------------
# (b) reaction-type diversity per compound
# -----------------------------------------------------------------------------
def panel_reaction_diversity(ax, fd: D.FigData) -> pd.DataFrame:
    """How many distinct reaction types does each compound carry a pattern for,
    and how many does it actually show?"""
    hot_cols = [D.ASSAY_TO_HOT[r] for r in fd.rx_focus
                if r in D.ASSAY_TO_HOT and D.ASSAY_TO_HOT[r] in fd.parents.columns]
    expected = fd.parents[hot_cols].sum(axis=1).values

    obs = fd.metab_grid[fd.metab_grid["observed"]]
    per_parent = obs.groupby("name")[fd.rx_focus].max()
    per_parent = per_parent.reindex(fd.parents["name"]).fillna(0)
    observed = per_parent.sum(axis=1).values

    bins = np.arange(-0.5, max(expected.max(), observed.max()) + 1.5, 1)
    centres = (bins[:-1] + bins[1:]) / 2
    he, _ = np.histogram(expected, bins=bins)
    ho, _ = np.histogram(observed, bins=bins)

    w = 0.42
    b1 = ax.bar(centres - w / 2, he, width=w, color=S.COLOR_EXPECTED,
                label="expected (HotSMARTS match)", edgecolor="none")
    b2 = ax.bar(centres + w / 2, ho, width=w, color=S.COLOR_OBSERVED,
                label="observed", edgecolor="none")

    ax.set_xlabel("distinct reaction types per compound")
    ax.set_ylabel("compounds")
    ax.set_xticks(np.arange(0, centres.max() + 1, 1))
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Reaction-type diversity", loc="left")

    ax.text(0.98, 0.62,
            f"expected  median {np.median(expected):.0f}\n"
            f"observed  median {np.median(observed):.0f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=plt.rcParams["font.size"] - 1, color=S.C_GREY)

    return pd.DataFrame({"n_reaction_types": centres.astype(int),
                         "n_compounds_expected": he,
                         "n_compounds_observed": ho})


# -----------------------------------------------------------------------------
# (c) representative molecules with highlighted HotSMARTS
# -----------------------------------------------------------------------------
def choose_representative(fd: D.FigData, rx: str) -> Optional[pd.Series]:
    """Smallest compound that both matches the pattern and shows the reaction;
    small molecules are legible at thumbnail size. Falls back to a pattern
    match alone when no compound shows the reaction."""
    hot = D.ASSAY_TO_HOT.get(rx)
    if hot is None or hot not in fd.parents.columns:
        return None

    obs = fd.metab_grid[fd.metab_grid["observed"]]
    if rx in obs.columns:
        names_observed = set(obs.loc[obs[rx] == 1, "name"])
    else:
        names_observed = set()

    cand = fd.parents[fd.parents[hot] == 1].copy()
    if cand.empty:
        return None
    cand["shows_reaction"] = cand["name"].isin(names_observed).astype(int)
    cand = cand.sort_values(["shows_reaction", "HeavyAtomCount"],
                            ascending=[False, True])
    return cand.iloc[0]


def panel_smarts_grid(fig, spec, fd: D.FigData,
                      reactions: Sequence[str] = tuple(D.RX_HOT_GRID),
                      ncols: int = 5) -> pd.DataFrame:
    nrows = int(np.ceil(len(reactions) / ncols))
    inner = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=spec,
                                    wspace=0.05, hspace=0.42)
    rows = []
    for i, rx in enumerate(reactions):
        ax = fig.add_subplot(inner[i // ncols, i % ncols])
        rep = choose_representative(fd, rx)
        if rep is None:
            S.hide(ax)
            ax.set_title(rx, fontsize=plt.rcParams["font.size"] * 1.15, pad=1)
            continue
        hot = D.ASSAY_TO_HOT[rx]
        atoms, bonds = D.first_hot_match(rep["mol"], D.HOT_PATTERNS[hot])
        S.draw_mol_on_ax(
            ax, rep["mol"],
            title=f"{rx}",
            highlight_atoms=atoms, highlight_bonds=bonds,
            highlight_colour=(0.90, 0.47, 0.13),
            subtitle=str(rep["name"])[:24],
            width=483, height=380,
            title_fontsize=plt.rcParams["font.size"] * 1.15)
        rows.append({"reaction": rx, "hot_pattern": hot,
                     "representative": rep["name"],
                     "smiles": rep["smiles"],
                     "n_atoms_matched": len(atoms),
                     "shows_reaction": int(rep["shows_reaction"])})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# (d) top scaffolds
# -----------------------------------------------------------------------------
def scaffold_table(fd: D.FigData) -> Tuple[pd.DataFrame, Dict[str, float]]:
    recs = []
    for _, r in fd.parents.iterrows():
        scaf = r["murcko"]
        if scaf:
            recs.append({"name": r["name"], "scaffold": scaf, "source": "murcko"})
        for rs in D.ring_system_smiles(r["mol"]):
            recs.append({"name": r["name"], "scaffold": rs, "source": "ring_system"})
    smap = pd.DataFrame(recs).drop_duplicates()

    murcko = smap[smap["source"] == "murcko"]
    counts = (murcko.groupby("scaffold")["name"].nunique()
              .sort_values(ascending=False).rename("n_compounds").reset_index())
    n_scaf = len(counts)
    singletons = int((counts["n_compounds"] == 1).sum())
    stats = {
        "n_compounds": fd.n_parents,
        "n_murcko_scaffolds": n_scaf,
        "singleton_scaffolds": singletons,
        "singleton_fraction": singletons / n_scaf if n_scaf else np.nan,
        "scaffolds_per_compound": n_scaf / fd.n_parents if fd.n_parents else np.nan,
        "n_ring_systems": int(smap[smap["source"] == "ring_system"]["scaffold"].nunique()),
    }
    return counts, stats


def panel_top_scaffolds(fig, spec, fd: D.FigData, top_n: int = 5
                        ) -> Tuple[pd.DataFrame, Dict[str, float]]:
    counts, stats = scaffold_table(fd)
    top = counts.head(top_n)
    inner = GridSpecFromSubplotSpec(1, top_n + 1, subplot_spec=spec,
                                    width_ratios=[1] * top_n + [1.15],
                                    wspace=0.06)
    for i, (_, r) in enumerate(top.iterrows()):
        ax = fig.add_subplot(inner[0, i])
        mol = Chem.MolFromSmiles(r["scaffold"]) if r["scaffold"] else None
        S.draw_mol_on_ax(ax, mol,
                         title=f"n = {int(r['n_compounds'])}",
                         width=380, height=300,
                         title_fontsize=plt.rcParams["font.size"])
    ax = fig.add_subplot(inner[0, top_n])
    S.hide(ax)
    txt = (f"{stats['n_compounds']} compounds\n"
           f"{stats['n_murcko_scaffolds']} Murcko scaffolds\n"
           f"{stats['singleton_scaffolds']} seen once "
           f"({100 * stats['singleton_fraction']:.0f}%)\n"
           f"{stats['n_ring_systems']} distinct ring systems")
    ax.text(0.0, 0.5, txt, transform=ax.transAxes, ha="left", va="center",
            fontsize=plt.rcParams["font.size"] - 0.5, linespacing=1.6)
    return top, stats


# -----------------------------------------------------------------------------
# (e) alluvial
# -----------------------------------------------------------------------------
def _ribbon(ax, x0: float, x1: float, y0a: float, y0b: float,
            y1a: float, y1b: float, colour: str, alpha: float = 0.55) -> None:
    """One flow band drawn as a pair of cubic Beziers."""
    cx = (x0 + x1) / 2.0
    verts = [
        (x0, y0a), (cx, y0a), (cx, y1a), (x1, y1a),
        (x1, y1b), (cx, y1b), (cx, y0b), (x0, y0b), (x0, y0a),
    ]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=colour,
                           edgecolor="none", alpha=alpha, zorder=1))


def alluvial_2x2(ax, n_both: int, n_missed: int, n_unexpected: int,
                 n_neither: int, title: str = "",
                 show_axis_labels: bool = False) -> None:
    """Left stack: HotSMARTS present / absent. Right stack: reaction observed /
    not observed. Band widths are compound counts."""
    total = max(n_both + n_missed + n_unexpected + n_neither, 1)
    gap = 0.045 * total

    left_present = n_both + n_missed
    left_absent = n_unexpected + n_neither
    right_obs = n_both + n_unexpected
    right_not = n_missed + n_neither

    xl, xr, bw = 0.06, 0.94, 0.055

    # node tops, measured downwards from the top of the plot
    lp_top, lp_bot = 0.0, left_present
    la_top, la_bot = left_present + gap, left_present + gap + left_absent
    ro_top, ro_bot = 0.0, right_obs
    rn_top, rn_bot = right_obs + gap, right_obs + gap + right_not

    for (x, top, bot, colour) in (
            (xl, lp_top, lp_bot, S.C_PURPLE), (xl, la_top, la_bot, S.C_LGREY),
            (xr, ro_top, ro_bot, S.C_BLUE), (xr, rn_top, rn_bot, S.C_LGREY)):
        ax.add_patch(plt.Rectangle((x - bw / 2, top), bw, max(bot - top, 0),
                                   facecolor=colour, edgecolor="none", zorder=3))

    # flows, stacked in the same order on both sides
    lcur, rcur = lp_top, ro_top
    _ribbon(ax, xl + bw / 2, xr - bw / 2, lcur, lcur + n_both,
            rcur, rcur + n_both, S.C_GREEN, 0.60)
    lcur += n_both
    rcur_n = rn_top
    _ribbon(ax, xl + bw / 2, xr - bw / 2, lcur, lcur + n_missed,
            rcur_n, rcur_n + n_missed, S.C_VERM, 0.45)
    rcur_n += n_missed

    lcur2 = la_top
    _ribbon(ax, xl + bw / 2, xr - bw / 2, lcur2, lcur2 + n_unexpected,
            rcur + n_both, rcur + n_both + n_unexpected, S.C_ORANGE, 0.60)
    lcur2 += n_unexpected
    _ribbon(ax, xl + bw / 2, xr - bw / 2, lcur2, lcur2 + n_neither,
            rcur_n, rcur_n + n_neither, S.C_LGREY, 0.45)

    ax.set_xlim(0, 1)
    ax.set_ylim(la_bot if la_bot > rn_bot else rn_bot, -0.06 * total)
    S.hide(ax)
    if title:
        ax.set_title(title, loc="center", fontsize=plt.rcParams["font.size"] + 1.5,
                     pad=2)

    fs = plt.rcParams["font.size"]
    ax.text(xl, lp_top - 0.015 * total, f"{left_present}", ha="center",
            va="bottom", fontsize=fs, color=S.C_PURPLE, fontweight="bold")
    ax.text(xr, ro_top - 0.015 * total, f"{right_obs}", ha="center",
            va="bottom", fontsize=fs, color=S.C_BLUE, fontweight="bold")
    if show_axis_labels:
        ax.text(xl, la_bot + 0.02 * total, "pattern", ha="center", va="top",
                fontsize=fs, color=S.C_GREY)
        ax.text(xr, rn_bot + 0.02 * total, "reaction", ha="center", va="top",
                fontsize=fs, color=S.C_GREY)


def panel_alluvial(fig, spec, fd: D.FigData,
                   reactions: Sequence[str] = tuple(D.RX_ALLUVIAL)
                   ) -> pd.DataFrame:
    evo = fd.expected_vs_observed().set_index("reaction")
    inner = GridSpecFromSubplotSpec(1, len(reactions), subplot_spec=spec,
                                    wspace=0.12)
    rows = []
    for i, rx in enumerate(reactions):
        ax = fig.add_subplot(inner[0, i])
        if rx not in evo.index:
            S.hide(ax)
            continue
        r = evo.loc[rx]
        n_neither = int(r["n_parents"] - r["n_both"] - r["n_missed"]
                        - r["n_unexpected"])
        alluvial_2x2(ax, int(r["n_both"]), int(r["n_missed"]),
                     int(r["n_unexpected"]), n_neither, title=rx,
                     show_axis_labels=(i == 0))
        rows.append({"reaction": rx, "n_both": int(r["n_both"]),
                     "n_pattern_only": int(r["n_missed"]),
                     "n_reaction_only": int(r["n_unexpected"]),
                     "n_neither": n_neither})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
def build(fd: D.FigData, fda: Optional[pd.DataFrame] = None,
          outdir: Path = S.FIGDIR, formats=S.SAVE_FORMATS) -> None:
    S.apply_style()
    fig = plt.figure(figsize=(S.COL2_WIDE, 9.4))
    gs = GridSpec(4, 2, figure=fig,
                  height_ratios=[2.25, 2.35, 1.55, 1.85],
                  width_ratios=[1.0, 1.0],
                  left=0.075, right=0.980, top=0.950, bottom=0.085,
                  wspace=0.22, hspace=0.46)

    ax_a = fig.add_subplot(gs[0, 0])
    t_a = panel_umap(ax_a, fd, fda)
    S.panel_label(ax_a, "a", dx=-0.10, dy=1.02)

    ax_b = fig.add_subplot(gs[0, 1])
    t_b = panel_reaction_diversity(ax_b, fd)
    S.panel_label(ax_b, "b", dx=-0.12, dy=1.02)

    pos_c = gs[1, :].get_position(fig)
    t_c = panel_smarts_grid(fig, gs[1, :], fd)
    S.panel_label_fig(fig, "c", 0.012, pos_c.y1 + 0.016)
    fig.text(0.075, pos_c.y1 + 0.018,
             "Representative HotSMARTS match per reaction type",
             fontsize=plt.rcParams["font.size"] + 0.5, va="bottom")

    pos_d = gs[2, :].get_position(fig)
    t_d, stats_d = panel_top_scaffolds(fig, gs[2, :], fd)
    S.panel_label_fig(fig, "d", 0.012, pos_d.y1 + 0.014)
    fig.text(0.075, pos_d.y1 + 0.016,
             "Most frequent Murcko scaffolds and scaffold diversity",
             fontsize=plt.rcParams["font.size"] + 0.5, va="bottom")

    pos_e = gs[3, :].get_position(fig)
    t_e = panel_alluvial(fig, gs[3, :], fd)
    S.panel_label_fig(fig, "e", 0.012, pos_e.y1 + 0.020)
    fig.text(0.075, pos_e.y1 + 0.022,
             "HotSMARTS pattern vs observed reaction (compounds)",
             fontsize=plt.rcParams["font.size"] + 0.5, va="bottom")
    fig.legend(handles=[
        Patch(facecolor=S.C_GREEN, alpha=0.6, label="pattern present, reaction observed"),
        Patch(facecolor=S.C_VERM, alpha=0.45, label="pattern present, not observed"),
        Patch(facecolor=S.C_ORANGE, alpha=0.6, label="no pattern, reaction observed"),
        Patch(facecolor=S.C_LGREY, alpha=0.45, label="neither"),
    ], loc="upper center", ncol=4, bbox_to_anchor=(0.54, pos_e.y0 - 0.012),
        frameon=False, fontsize=plt.rcParams["font.size"] + 1.5)

    S.save(fig, "figure2_chemical_space", outdir=outdir, formats=formats)

    D.export_table(t_a, "fig2a_chemical_space_coords")
    D.export_table(t_b, "fig2b_reaction_type_diversity")
    D.export_table(t_c, "fig2c_smarts_representatives")
    D.export_table(t_d, "fig2d_top_scaffolds")
    D.export_table(pd.DataFrame([stats_d]), "fig2d_scaffold_diversity_stats")
    D.export_table(t_e, "fig2e_pattern_vs_reaction")


if __name__ == "__main__":
    fd = D.load()
    build(fd, D.load_fda())

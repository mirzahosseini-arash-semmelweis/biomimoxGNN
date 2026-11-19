# fig3_expected_observed.py
# Figure 3 -- what the SMARTS rules expect, what the assay actually produced,
# and how that relates to what is known in vivo.
#
#   (a) expected vs observed per reaction type (compound level), as a dumbbell
#   (b) agreement breakdown per reaction type: pattern-and-reaction,
#       pattern-only, reaction-only
#   (c) observed metabolites by generation, split by in vivo status
#   (d) in vivo status of observed metabolites per reaction type
#

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import biomimox_figdata as D
import figstyle as S


# -----------------------------------------------------------------------------
# (a) expected vs observed
# -----------------------------------------------------------------------------
def panel_dumbbell(ax, fd: D.FigData) -> pd.DataFrame:
    evo = fd.expected_vs_observed().copy()
    evo = evo.sort_values("n_expected", ascending=True).reset_index(drop=True)
    y = np.arange(len(evo))

    for i, r in evo.iterrows():
        ax.plot([r["n_observed"], r["n_expected"]], [i, i],
                color=S.C_LGREY, lw=1.6, zorder=1, solid_capstyle="round")
    ax.scatter(evo["n_expected"], y, s=22, color=S.COLOR_EXPECTED, zorder=3,
               label="expected (HotSMARTS match)", linewidths=0)
    ax.scatter(evo["n_observed"], y, s=22, color=S.COLOR_OBSERVED, zorder=3,
               label="observed", linewidths=0)

    # reactions with no pattern defined cannot have an "expected" count
    nopat = evo[~evo["has_hot_pattern"]]
    for i, r in nopat.iterrows():
        ax.scatter([r["n_expected"]], [i], s=22, facecolors="none",
                   edgecolors=S.C_VERM, linewidths=0.8, zorder=4)
        ax.text(r["n_observed"] + 3, i, "no pattern defined", va="center",
                ha="left", fontsize=plt.rcParams["font.size"] - 2,
                color=S.C_VERM)

    ax.set_yticks(y)
    ax.set_yticklabels(evo["reaction"])
    ax.set_xlabel("parent compounds")
    ax.set_xlim(-2, max(evo["n_expected"].max(), evo["n_observed"].max()) * 1.22)
    ax.axvline(fd.n_parents, color=S.C_GREY, lw=0.6, ls=":", zorder=0)
    ax.text(fd.n_parents, len(evo) - 0.3, f"all {fd.n_parents}", rotation=90,
            ha="right", va="top", fontsize=plt.rcParams["font.size"] - 2,
            color=S.C_GREY)
    ax.legend(loc="lower right")
    ax.xaxis.grid(True, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Expected vs observed", loc="left")
    return evo


# -----------------------------------------------------------------------------
# (b) agreement breakdown
# -----------------------------------------------------------------------------
def panel_agreement(ax, fd: D.FigData) -> pd.DataFrame:
    evo = fd.expected_vs_observed().copy()
    evo = evo.sort_values("n_expected", ascending=True).reset_index(drop=True)
    y = np.arange(len(evo))

    both = evo["n_both"].values.astype(float)
    pat_only = evo["n_missed"].values.astype(float)
    rx_only = evo["n_unexpected"].values.astype(float)

    ax.barh(y, both, color=S.C_GREEN, height=0.72, label="pattern + reaction")
    ax.barh(y, pat_only, left=both, color=S.C_VERM, alpha=0.75, height=0.72,
            label="pattern only (not observed)")
    ax.barh(y, rx_only, left=both + pat_only, color=S.C_ORANGE, height=0.72,
            label="reaction only (no pattern)")

    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("parent compounds")
    ax.legend(loc="lower right")
    ax.xaxis.grid(True, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Agreement", loc="left")

    # precision of each rule, where it is defined
    for i, r in evo.iterrows():
        if r["has_hot_pattern"] and r["n_expected"] > 0:
            ax.text(both[i] + pat_only[i] + rx_only[i] + 1.5, i,
                    f"{100 * r['precision_of_pattern']:.0f}%", va="center",
                    ha="left", fontsize=plt.rcParams["font.size"] - 2,
                    color=S.C_GREY)
    ax.set_xlim(0, (both + pat_only + rx_only).max() * 1.16)
    return evo[["reaction", "n_both", "n_missed", "n_unexpected",
                "precision_of_pattern", "recall_of_observed"]]


# -----------------------------------------------------------------------------
# (c) generation cascade
# -----------------------------------------------------------------------------
def panel_generations(ax, fd: D.FigData) -> pd.DataFrame:
    gs_df = fd.generation_summary()
    gens = [g for g in D.GEN_ORDER
            if gs_df.loc[gs_df["generation"] == g, "n"].sum() > 0]
    x = np.arange(len(gens))
    bottom = np.zeros(len(gens))

    for code in (1, 2, 0, -1):
        vals = np.array([
            gs_df.loc[(gs_df["generation"] == g) & (gs_df["invivo"] == code),
                      "n"].sum() for g in gens], dtype=float)
        if vals.sum() == 0:
            continue
        colour = S.COLOR_INVIVO.get(code, S.C_GREY) if code >= 0 else "#F2F2F2"
        label = (S.LABEL_INVIVO.get(code) if code >= 0
                 else "in vivo field empty")
        ax.bar(x, vals, bottom=bottom, color=colour, width=0.70, label=label,
               edgecolor="white", linewidth=0.4)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(gens)
    ax.set_xlabel("metabolite generation")
    ax.set_ylabel("observed metabolites")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.46, 1.00),
              fontsize=plt.rcParams["font.size"] - 1.5)
    ax.yaxis.grid(True, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Metabolite generation and in vivo status", loc="left",
                 pad=4)

    for xi, tot in zip(x, bottom):
        if tot > 0:
            ax.text(xi, tot + bottom.max() * 0.015, f"{int(tot)}", ha="center",
                    va="bottom", fontsize=plt.rcParams["font.size"] - 1.5)
    ax.set_ylim(0, bottom.max() * 1.30)
    return gs_df


# -----------------------------------------------------------------------------
# (d) in vivo concordance per reaction type
# -----------------------------------------------------------------------------
def panel_invivo(ax, fd: D.FigData) -> pd.DataFrame:
    iv = fd.invivo_summary()
    order = (iv[iv["invivo"] == 1]
             .assign(frac=lambda d: np.where(d["n_total"] > 0,
                                             d["n"] / d["n_total"], 0))
             .sort_values("frac", ascending=True)["reaction"].tolist())
    y = np.arange(len(order))
    left = np.zeros(len(order))
    totals = np.array([iv.loc[iv["reaction"] == r, "n_total"].iloc[0]
                       for r in order], dtype=float)

    for code in (1, 2, 0, -1):
        vals = np.array([
            iv.loc[(iv["reaction"] == r) & (iv["invivo"] == code), "n"].sum()
            for r in order], dtype=float)
        frac = np.divide(vals, totals, out=np.zeros_like(vals),
                         where=totals > 0) * 100.0
        colour = S.COLOR_INVIVO.get(code, S.C_GREY) if code >= 0 else "#F2F2F2"
        label = (S.LABEL_INVIVO.get(code) if code >= 0
                 else "in vivo field empty")
        ax.barh(y, frac, left=left, color=colour, height=0.72, label=label,
                edgecolor="white", linewidth=0.4)
        left += frac

    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("observed metabolites (%)")
    ax.set_xlim(0, 100)
    ax.set_title("In vivo status by reaction type", loc="left")
    for i, t in enumerate(totals):
        ax.text(101.5, i, f"n={int(t)}", va="center", ha="left",
                fontsize=plt.rcParams["font.size"] - 2, color=S.C_GREY)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.16))
    return iv


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
def build(fd: D.FigData, outdir: Path = S.FIGDIR,
          formats=S.SAVE_FORMATS) -> None:
    S.apply_style()
    fig = plt.figure(figsize=(S.COL2, 6.9))
    gs = GridSpec(2, 2, figure=fig,
                  height_ratios=[1.0, 1.12], width_ratios=[1.0, 0.85],
                  left=0.105, right=0.945, top=0.945, bottom=0.115,
                  wspace=0.14, hspace=0.52)

    ax_a = fig.add_subplot(gs[0, 0])
    t_a = panel_dumbbell(ax_a, fd)
    S.panel_label(ax_a, "a", dx=-0.21, dy=1.02)

    # Deliberately NOT sharey: a shared y-axis propagates (b)'s blank tick
    # labels back onto (a). Both panels use the same sort order, so the rows
    # line up anyway.
    ax_b = fig.add_subplot(gs[0, 1])
    t_b = panel_agreement(ax_b, fd)
    ax_b.set_ylim(ax_a.get_ylim())
    S.panel_label(ax_b, "b", dx=-0.06, dy=1.02)

    ax_c = fig.add_subplot(gs[1, 0])
    t_c = panel_generations(ax_c, fd)
    S.panel_label(ax_c, "c", dx=-0.21, dy=1.02)

    ax_d = fig.add_subplot(gs[1, 1])
    t_d = panel_invivo(ax_d, fd)
    S.panel_label(ax_d, "d", dx=-0.30, dy=1.02)

    S.save(fig, "figure3_expected_observed", outdir=outdir, formats=formats)
    D.export_table(t_a, "fig3a_expected_vs_observed")
    D.export_table(t_b, "fig3b_agreement")
    D.export_table(t_c, "fig3c_generation_invivo")
    D.export_table(t_d, "fig3d_invivo_by_reaction")


if __name__ == "__main__":
    build(D.load())

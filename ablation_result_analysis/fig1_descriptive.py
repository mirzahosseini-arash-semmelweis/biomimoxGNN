# fig1_descriptive.py
# Figure 1 -- descriptive statistics of compounds, metabolites and conversions.
#
#   (a) study flow: parent compounds -> catalogued metabolites -> HRMS -> NMR
#   (b) distinct observed metabolites per parent compound, by pH and condition
#   (c) conversion-category composition, 2 rows (condition) x 4 columns (pH)
#   (d) oxidation reaction-type counts, one column per pH
#

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Polygon, Rectangle

import biomimox_figdata as D
import figstyle as S


# -----------------------------------------------------------------------------
# (a) study flow
# -----------------------------------------------------------------------------
def panel_flow(ax, fd: D.FigData) -> pd.DataFrame:
    counts, placeholder = fd.workflow_counts()
    n_met = counts["metabolites"]

    stages = [
        ("Parent compounds\nstudied", counts["parents"], "compounds", S.C_BLUE),
        ("Metabolites\ncatalogued", counts["metabolites"], "metabolites", S.C_SKY),
        ("Characterised\nby HRMS", counts["hrms_metabolites"], "metabolites", S.C_GREEN),
        ("Confirmed\nby NMR", counts["nmr_metabolites"], "metabolites", S.C_ORANGE),
    ]

    ax.set_xlim(-0.95, 1.45)
    ax.set_ylim(-0.15, len(stages) + (0.95 if placeholder else 0.1))
    ax.invert_yaxis()
    S.hide(ax)

    widest = max(s[1] for s in stages[1:])          # scale on the metabolite axis
    rows = []
    for i, (label, n, unit, colour) in enumerate(stages):
        # The first bar counts compounds, not metabolites: give it a fixed
        # width and mark the unit change rather than implying a shared scale.
        if unit == "compounds":
            frac = 0.60
        else:
            frac = 0.08 + 0.92 * (n / widest if widest else 0)
        x0 = (1.0 - frac) / 2.0
        y0 = i + 0.18
        h = 0.52
        ax.add_patch(Rectangle((x0, y0), frac, h, facecolor=colour,
                               edgecolor="white", linewidth=0.8, zorder=2))
        ax.text(0.5, y0 + h / 2, f"{n:,}", ha="center", va="center",
                fontsize=plt.rcParams["font.size"] + 1.5, fontweight="bold",
                color="white", zorder=3)
        ax.text(-0.10, y0 + h / 2, label, ha="right", va="center",
                fontsize=plt.rcParams["font.size"] - 0.5, zorder=3)

        note = ""
        if unit == "metabolites" and i >= 2 and n_met:
            note = f"{100.0 * n / n_met:.0f}% of metabolites"
        elif i == 1 and counts["parents"]:
            note = f"{n / counts['parents']:.1f} per compound"
        if note:
            ax.text(x0 + frac + 0.02, y0 + h / 2, note, ha="left", va="center",
                    fontsize=plt.rcParams["font.size"] - 1.5, color=S.C_GREY,
                    zorder=3)

        if i < len(stages) - 1:
            ax.annotate("", xy=(0.5, i + 1.12), xytext=(0.5, i + 0.76),
                        arrowprops=dict(arrowstyle="-|>", linewidth=0.7,
                                        color=S.C_GREY, shrinkA=0, shrinkB=0))
        rows.append({"stage": label.replace("\n", " "), "n": n, "unit": unit})

    # The compound/metabolite unit change, stated rather than hidden.
    xb = -0.90
    ax.plot([xb, xb], [0.20, 0.70], color=S.C_GREY, lw=0.7)
    ax.plot([xb, xb], [1.20, 3.70], color=S.C_GREY, lw=0.7)
    ax.text(xb - 0.03, 0.45, "compounds", rotation=90, ha="right", va="center",
            fontsize=plt.rcParams["font.size"] - 2, color=S.C_GREY)
    ax.text(xb - 0.03, 2.45, "metabolites", rotation=90, ha="right", va="center",
            fontsize=plt.rcParams["font.size"] - 2, color=S.C_GREY)

    if placeholder:
        ax.text(0.25, len(stages) + 0.55,
                "HRMS / NMR counts are PLACEHOLDERS —\n"
                "set WORKFLOW_COUNTS in biomimox_figdata.py",
                ha="center", va="center",
                fontsize=plt.rcParams["font.size"] - 1.5, color=S.C_VERM,
                fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3F0",
                          edgecolor=S.C_VERM, linewidth=0.6))
    ax.set_title("Study flow", loc="left", x=-0.33)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# (b) metabolites per parent compound
# -----------------------------------------------------------------------------
def panel_metabolites_per_parent(ax, fd: D.FigData) -> pd.DataFrame:
    mpp = fd.metabolites_per_parent(biomim=None)
    width = 0.34
    positions, ticks, labels = [], [], []

    for i, ph in enumerate(D.PH_ORDER):
        for j, bm in enumerate((0, 1)):
            vals = mpp.loc[(mpp["pH"] == ph) & (mpp["biomim"] == bm),
                           "n_metabolites"].values.astype(float)
            if vals.size == 0:
                continue
            pos = i + (j - 0.5) * width * 1.15
            positions.append(pos)

            bp = ax.boxplot([vals], positions=[pos], widths=width,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", linewidth=1.0),
                            whiskerprops=dict(linewidth=0.6),
                            capprops=dict(linewidth=0.6),
                            boxprops=dict(linewidth=0.5))
            for patch in bp["boxes"]:
                patch.set_facecolor(S.COLOR_BIOMIM[bm])
                patch.set_alpha(0.85 if bm == 1 else 0.6)

            # jittered points: the distribution is zero-heavy and a box alone
            # hides that
            rng = np.random.RandomState(D.RANDOM_STATE + i * 10 + j)
            jit = rng.uniform(-width * 0.33, width * 0.33, size=vals.size)
            ax.scatter(pos + jit, vals, s=1.6, alpha=0.35, linewidths=0,
                       color="black", zorder=3)
            ax.text(pos, -0.9, f"{np.median(vals):.0f}", ha="center", va="top",
                    fontsize=plt.rcParams["font.size"] - 2, color=S.C_GREY)
        ticks.append(i)
        labels.append(f"{ph}")

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("pH")
    ax.set_ylabel("distinct metabolites\nper parent compound")
    top = float(mpp["n_metabolites"].max())
    ax.set_ylim(-1.7, top * 1.18)
    ax.axhline(0, color=S.C_LGREY, lw=0.5, zorder=0)
    ax.text(-0.62, -0.9, "median", ha="right", va="top",
            fontsize=plt.rcParams["font.size"] - 2, color=S.C_GREY)
    ax.legend(handles=S.legend_patches(S.COLOR_BIOMIM, S.LABEL_BIOMIM),
              loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.00))
    ax.set_title("Metabolites per compound", loc="left", pad=14)
    ax.yaxis.grid(True, alpha=0.5)
    ax.set_axisbelow(True)

    out = (mpp.groupby(["pH", "biomim"])["n_metabolites"]
           .agg(n_compounds="size", mean="mean", median="median",
                q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
                max="max", n_with_none=lambda s: int((s == 0).sum()))
           .reset_index())
    return out


# -----------------------------------------------------------------------------
# (c) conversion categories
# -----------------------------------------------------------------------------
def panel_conversion_pies(fig, spec, fd: D.FigData) -> pd.DataFrame:
    cc = fd.conversion_categories()
    cats = (["none"] if D.CONVERSION_SPLIT_ZERO else []) + D.CONVERSION_LABELS
    inner = GridSpecFromSubplotSpec(2, len(D.PH_ORDER), subplot_spec=spec,
                                    wspace=0.02, hspace=0.30)

    for r, bm in enumerate((0, 1)):
        for c, ph in enumerate(D.PH_ORDER):
            ax = fig.add_subplot(inner[r, c])
            sub = cc[(cc["pH"] == ph) & (cc["biomim"] == bm)]
            vals = [int(sub.loc[sub["category"] == k, "n"].sum()) for k in cats]
            total = sum(vals)
            if total == 0:
                S.hide(ax)
                continue
            colours = [S.COLOR_CONV[k] for k in cats]
            # donut: the centre is where the reacting fraction is written
            ax.pie(vals, colors=colours, startangle=90, counterclock=False,
                   wedgeprops=dict(linewidth=0.5, edgecolor="white",
                                   width=0.42),
                   radius=1.25)
            ax.set_aspect("equal")

            reacted = total - (vals[0] if D.CONVERSION_SPLIT_ZERO else 0)
            pct = (reacted / total * 100) if total > 0 else 0
            ax.text(0, 0.12, f"{pct:.0f}%", ha="center", va="center",
                    fontsize=plt.rcParams["font.size"], fontweight="bold")
            ax.text(0, -0.32, "converted", ha="center", va="center",
                    fontsize=plt.rcParams["font.size"] - 2, color=S.C_GREY)

            if r == 0:
                ax.set_title(f"pH {ph}", fontsize=plt.rcParams["font.size"],
                             loc="center", pad=1,
                             color=("#8a6d00" if ph == 2.7 else S.COLOR_PH[ph]))
            if c == 0:
                ax.text(-1.78, 0, S.LABEL_BIOMIM[bm], rotation=90,
                        ha="center", va="center",
                        fontsize=plt.rcParams["font.size"] - 0.5)

    labels = {"none": "no conversion (0%)", "low": "low (<33%)",
              "medium": "medium (33–67%)", "high": "high (>67%)"}
    handles = S.legend_patches({k: S.COLOR_CONV[k] for k in cats}, labels)
    return cc, handles


# -----------------------------------------------------------------------------
# (d) reaction-type counts
# -----------------------------------------------------------------------------
def panel_reaction_counts(fig, spec, fd: D.FigData,
                          biomim: Optional[int] = None) -> pd.DataFrame:
    rc = fd.reaction_counts_by_ph(biomim=biomim, observed_only=True,
                                  unit="metabolite")
    order = list(fd.rx_focus)
    xmax = max(1, int(rc["n"].max()))
    inner = GridSpecFromSubplotSpec(1, len(D.PH_ORDER), subplot_spec=spec,
                                    wspace=0.10)

    axes = []
    for c, ph in enumerate(D.PH_ORDER):
        ax = fig.add_subplot(inner[0, c])
        axes.append(ax)
        sub = rc[rc["pH"] == ph].set_index("reaction").reindex(order)
        y = np.arange(len(order))
        bars = ax.barh(y, sub["n"].values, height=0.72,
                       color=S.COLOR_PH[ph], edgecolor="none")
        S.annotate_bars(ax, bars, sub["n"].values, fmt="{:.0f}",
                        pad=xmax * 0.025, horizontal=True,
                        fontsize=plt.rcParams["font.size"] - 2)
        ax.set_yticks(y)
        if c == 0:
            ax.set_yticklabels(order)
        else:
            ax.set_yticklabels([])
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xlim(0, xmax * 1.22)
        ax.set_title(f"pH {ph}", loc="center", fontsize=plt.rcParams["font.size"],
                     color=S.COLOR_PH[ph] if ph != 2.7 else "#8a6d00", pad=2)
        ax.xaxis.grid(True, alpha=0.5)
        ax.set_axisbelow(True)
        if c == len(D.PH_ORDER) // 2:
            ax.set_xlabel("distinct metabolites observed")
        S.despine(ax)
        ax.spines["left"].set_visible(c == 0)
        ax.tick_params(left=(c == 0))

    return rc


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
def build(fd: D.FigData, outdir: Path = S.FIGDIR,
          biomim_for_reactions: Optional[int] = None,
          formats=S.SAVE_FORMATS) -> None:
    S.apply_style()
    fig = plt.figure(figsize=(S.COL2, 7.3))
    gs = GridSpec(3, 2, figure=fig,
                  height_ratios=[1.85, 1.75, 3.00],
                  width_ratios=[1.0, 1.18],
                  left=0.145, right=0.975, top=0.940, bottom=0.055,
                  wspace=0.40, hspace=0.42)

    ax_a = fig.add_subplot(gs[0, 0])
    t_a = panel_flow(ax_a, fd)
    S.panel_label(ax_a, "a", dx=-0.42, dy=1.00)

    ax_b = fig.add_subplot(gs[0, 1])
    t_b = panel_metabolites_per_parent(ax_b, fd)
    S.panel_label(ax_b, "b", dx=-0.17, dy=1.09)

    # Section titles and panel letters are anchored to the real gridspec
    # geometry, so they cannot drift onto the panels when sizes change.
    pos_c = gs[1, :].get_position(fig)
    t_c, handles_c = panel_conversion_pies(fig, gs[1, :], fd)
    S.panel_label_fig(fig, "c", 0.012, pos_c.y1 + 0.028)
    fig.text(0.145, pos_c.y1 + 0.030, "Conversion of the parent compound",
             fontsize=plt.rcParams["font.size"] + 0.5, va="bottom")
    fig.legend(handles=handles_c, loc="upper center", ncol=len(handles_c),
               bbox_to_anchor=(0.56, pos_c.y0 - 0.004), frameon=False)

    pos_d = gs[2, :].get_position(fig)
    t_d = panel_reaction_counts(fig, gs[2, :], fd, biomim=biomim_for_reactions)
    sub = ("observed in either condition" if biomim_for_reactions is None
           else S.LABEL_BIOMIM[biomim_for_reactions] + " only")
    S.panel_label_fig(fig, "d", 0.012, pos_d.y1 + 0.026)
    fig.text(0.145, pos_d.y1 + 0.028,
             f"Oxidation reaction types  ({sub})",
             fontsize=plt.rcParams["font.size"] + 0.5, va="bottom")

    S.save(fig, "figure1_descriptive", outdir=outdir, formats=formats)

    D.export_table(t_a, "fig1a_study_flow")
    D.export_table(t_b, "fig1b_metabolites_per_parent")
    D.export_table(t_c, "fig1c_conversion_categories")
    D.export_table(t_d, "fig1d_reaction_counts_by_ph")


if __name__ == "__main__":
    build(D.load())

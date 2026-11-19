# fig5_ablation.py
# Figure 5 -- machine-learning ablation summary.
#
#   (a) CV PR-AUC with 95% CI for every arm, per reaction type
#   (b) prevalence-adjusted lift averaged across reaction types, per arm
#
# Two deliberate choices:
#
# * The metric is repeated-CV out-of-fold PR-AUC, not the held-out value. With
#   ~26 test compounds and as few as 1-4 positives for the rare reactions, a
#   single held-out PR-AUC moves visibly when one compound flips; the CV column
#   carries a spread across repeats and is what the comparison rests on.
#
# * Panel (b) plots PR-AUC minus prevalence. Prevalence runs from 0.024 (S_ox)
#   to 0.378 (percent), so a raw PR-AUC of 0.40 is excellent for one assay and
#   below chance for another. Subtracting prevalence does not change the
#   ranking within an assay -- it makes the average across assays mean
#   something. The dashed zero line is the class prior.

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import biomimox_figdata as D
import figstyle as S


def arm_colour(arm: str, family_group: str) -> str:
    return S.arm_colour(arm, family_group)


def arm_order(tab: pd.DataFrame) -> List[str]:
    return sorted(tab["arm"].unique())


# -----------------------------------------------------------------------------
# (a) per-assay dot-and-CI
# -----------------------------------------------------------------------------
def panel_per_assay(ax, tab: pd.DataFrame,
                    target_order: Optional[Sequence[str]] = None) -> pd.DataFrame:
    arms = arm_order(tab)
    if target_order is None:
        prev = tab.groupby("target")["prevalence"].first().sort_values()
        target_order = prev.index.tolist()

    n_arms = len(arms)
    span = 0.78
    offsets = np.linspace(-span / 2, span / 2, n_arms)

    for ti, target in enumerate(target_order):
        sub = tab[tab["target"] == target]
        if sub.empty:
            continue
        prev = float(sub["prevalence"].iloc[0])
        # class prior for this assay: the line every arm has to beat
        ax.plot([prev, prev], [ti - 0.44, ti + 0.44], color=S.C_GREY,
                lw=0.8, ls=":", zorder=1)

        for ai, arm in enumerate(arms):
            r = sub[sub["arm"] == arm]
            if r.empty:
                continue
            r = r.iloc[0]
            y = ti + offsets[ai]
            colour = arm_colour(arm, str(r.get("family_group", "other")))
            lo, hi = r.get("cv_pr_auc_lo"), r.get("cv_pr_auc_hi")
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [y, y], color=colour, lw=0.9, alpha=0.85,
                        zorder=2, solid_capstyle="butt")
            ax.scatter([r["cv_pr_auc_mean"]], [y], s=9, color=colour,
                       zorder=3, linewidths=0)

    ax.set_yticks(np.arange(len(target_order)))
    ax.set_yticklabels([f"{t}  ({tab.loc[tab['target'] == t, 'prevalence'].iloc[0]:.2f})"
                        for t in target_order])
    ax.set_ylim(-0.7, len(target_order) - 0.3)
    ax.set_xlabel("cross-validated PR-AUC (mean, 95% CI)")
    ax.set_ylabel("reaction type  (prevalence)")
    ax.set_xlim(0, min(1.0, float(tab["cv_pr_auc_hi"].max(skipna=True)) * 1.05))
    ax.xaxis.grid(True, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Per reaction type", loc="left")
    ax.text(0.995, 0.012, "dotted line = class prior", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=plt.rcParams["font.size"] - 2,
            color=S.C_GREY)
    return tab[tab["target"].isin(target_order)]


# -----------------------------------------------------------------------------
# (b) averaged lift
# -----------------------------------------------------------------------------
def panel_lift(ax, tab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, sub in tab.groupby("arm"):
        vals = pd.to_numeric(sub["lift"], errors="coerce").dropna().values
        if vals.size == 0:
            continue
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if vals.size > 1 else np.nan
        sem = sd / np.sqrt(vals.size) if np.isfinite(sd) else np.nan
        from scipy import stats as st
        ci = (st.t.ppf(0.975, vals.size - 1) * sem
              if vals.size > 1 and np.isfinite(sem) else np.nan)
        rows.append({
            "arm": arm,
            "arm_pretty": sub["arm_pretty"].iloc[0],
            "family_group": sub["family_group"].iloc[0],
            "n_assays": int(vals.size),
            "mean_lift": mean, "sd_lift": sd, "ci95": ci,
            "mean_pr_auc": float(pd.to_numeric(sub["cv_pr_auc_mean"],
                                               errors="coerce").mean()),
        })
    summ = pd.DataFrame(rows).sort_values("mean_lift")
    y = np.arange(len(summ))
    colours = [arm_colour(a, f) for a, f in zip(summ["arm"],
                                                summ["family_group"])]
    ax.barh(y, summ["mean_lift"], color=colours, height=0.68, zorder=2)
    err = summ["ci95"].values
    ax.errorbar(summ["mean_lift"], y, xerr=err, fmt="none", ecolor="black",
                elinewidth=0.7, capsize=1.5, zorder=3)
    ax.axvline(0, color="black", lw=0.7, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(summ["arm_pretty"])
    ax.set_xlabel("mean PR-AUC − prevalence\n(across reaction types, 95% CI)")
    ax.set_title("Averaged over reaction types", loc="left")
    ax.xaxis.grid(True, alpha=0.5)
    ax.set_axisbelow(True)
    # place the value past the end of the error bar, not on top of its cap
    for i, (_, r) in enumerate(summ.iterrows()):
        reach = r["mean_lift"] + (r["ci95"] if np.isfinite(r["ci95"]) else 0.0)
        ax.text(reach + 0.010, i, f"{r['mean_lift']:+.3f}", va="center",
                ha="left", fontsize=plt.rcParams["font.size"] - 1.5)
    lim = float(np.nanmax(np.abs(summ["mean_lift"].values) +
                          np.nan_to_num(err, nan=0.0)))
    ax.set_xlim(-0.1, 0.4)
    return summ


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
def build(tab: pd.DataFrame, outdir: Path = S.FIGDIR,
          formats=S.SAVE_FORMATS) -> None:
    S.apply_style()
    fig = plt.figure(figsize=(S.COL2, 5.2))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 0.82],
                  left=0.145, right=0.985, top=0.905, bottom=0.220,
                  wspace=0.52)

    ax_a = fig.add_subplot(gs[0, 0])
    t_a = panel_per_assay(ax_a, tab)
    S.panel_label(ax_a, "a", dx=-0.30, dy=1.015)

    ax_b = fig.add_subplot(gs[0, 1])
    t_b = panel_lift(ax_b, tab)
    S.panel_label(ax_b, "b", dx=-0.42, dy=1.015)

    arms = arm_order(tab)
    handles = []
    for arm in arms:
        sub = tab[tab["arm"] == arm]
        colour = arm_colour(arm, str(sub["family_group"].iloc[0]))
        handles.append(plt.Line2D([0], [0], marker="o", color="none",
                                  markerfacecolor=colour, markersize=6,
                                  label=sub["arm_pretty"].iloc[0]))
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.52, -0.005), frameon=False,
               fontsize=plt.rcParams["font.size"] + 0.5,
               labelspacing=0.625)

    S.save(fig, "figure5_ablation", outdir=outdir, formats=formats)
    D.export_table(t_a, "fig5a_per_assay_metrics")
    D.export_table(t_b, "fig5b_mean_lift_by_arm")


if __name__ == "__main__":
    build(D.load_ablation())

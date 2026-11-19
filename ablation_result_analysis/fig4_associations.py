# fig4_associations.py
# Figure 4 -- does a HotSMARTS pattern actually predict the reaction, and which
# molecular descriptors track conversion?
#
#   (a) Fisher exact test for every (HotSMARTS pattern x observed reaction)
#       pair at compound level, shown as log2 odds ratio with BH-FDR stars.
#       The full matrix rather than just the diagonal: off-diagonal signal
#       means a pattern is not specific to the reaction it was written for.
#   (b) effect size vs significance for the diagonal pairs only -- the rules as
#       actually used.
#   (c) volcano of RDKit descriptors against biomimetic conversion (Spearman).
#
# Multiplicity: (a) runs ~10 x 13 tests, so p values are corrected with
# Benjamini-Hochberg across the whole matrix, and (c) across all descriptors.
# Stars mark the corrected values; raw p values are in the exported tables.

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm

from scipy import stats

import biomimox_figdata as D
import figstyle as S

FDR_ALPHA = 0.05


def bh_fdr(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values, NaN-safe."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    try:
        from statsmodels.stats.multitest import multipletests
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
        return out
    except Exception:
        pass
    vals = p[ok]
    order = np.argsort(vals)
    n = vals.size
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, vals[i] * n / (rank + 1))
        adj[i] = prev
    out[ok] = np.clip(adj, 0, 1)
    return out


# -----------------------------------------------------------------------------
# Association testing
# -----------------------------------------------------------------------------
def pattern_reaction_tests(fd: D.FigData) -> pd.DataFrame:
    """2x2 Fisher test per (pattern, reaction) over parent compounds."""
    obs = fd.metab_grid[fd.metab_grid["observed"]]
    per_parent = (obs.groupby("name")[fd.rx_focus].max()
                  .reindex(fd.parents["name"]).fillna(0).astype(int))

    hot_cols = [h for h in D.HOT_NAMES if h in fd.parents.columns]
    rows = []
    for hot in hot_cols:
        pat = fd.parents[hot].values.astype(int)
        for rx in fd.rx_focus:
            rxv = per_parent[rx].values.astype(int)
            a = int(((pat == 1) & (rxv == 1)).sum())   # pattern + reaction
            b = int(((pat == 1) & (rxv == 0)).sum())
            c = int(((pat == 0) & (rxv == 1)).sum())
            d = int(((pat == 0) & (rxv == 0)).sum())
            try:
                orr, p = stats.fisher_exact([[a, b], [c, d]],
                                            alternative="two-sided")
            except Exception:
                orr, p = np.nan, np.nan
            # Haldane-Anscombe correction so zero cells still give a finite
            # effect size to plot
            or_h = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
            rows.append({
                "pattern": hot, "reaction": rx,
                "a_pat_rx": a, "b_pat_norx": b, "c_nopat_rx": c,
                "d_nopat_norx": d,
                "odds_ratio": orr, "odds_ratio_haldane": or_h,
                "log2_or": np.log2(or_h), "p_value": p,
                "is_matching_pair": D.ASSAY_TO_HOT.get(rx) == hot,
            })
    out = pd.DataFrame(rows)
    out["p_adj"] = bh_fdr(out["p_value"].values)
    return out


def descriptor_conversion_tests(fd: D.FigData, biomim: int = 1) -> pd.DataFrame:
    """Spearman correlation of each descriptor with conversion percent.

    One value per compound: the mean conversion across the pH values tested,
    for the chosen condition. Pooling by mean keeps each compound contributing
    once, so the n used by the test is the number of compounds, not the number
    of compound x pH rows.
    """
    conv = fd.conversions
    conv = conv[conv["biomim"] == biomim]
    per_cmp = (conv.groupby("name")["percent"]
               .agg(conv_mean="mean", conv_max="max", n_ph="size").reset_index())
    merged = fd.parents.merge(per_cmp, on="name", how="inner")

    rows = []
    for desc in D.DESCRIPTOR_NAMES:
        x = pd.to_numeric(merged[desc], errors="coerce").values
        y = merged["conv_mean"].values
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 8 or np.unique(x[ok]).size < 3:
            rows.append({"descriptor": desc, "rho": np.nan, "p_value": np.nan,
                         "n": int(ok.sum())})
            continue
        rho, p = stats.spearmanr(x[ok], y[ok])
        rows.append({"descriptor": desc, "rho": float(rho), "p_value": float(p),
                     "n": int(ok.sum())})
    out = pd.DataFrame(rows)
    out["p_adj"] = bh_fdr(out["p_value"].values)
    out["neglog10_p"] = -np.log10(out["p_value"].clip(lower=1e-300))
    return out


# -----------------------------------------------------------------------------
# (a) heatmap
# -----------------------------------------------------------------------------
def panel_heatmap(ax, tests: pd.DataFrame, fd: D.FigData) -> pd.DataFrame:
    piv = tests.pivot(index="pattern", columns="reaction", values="log2_or")
    padj = tests.pivot(index="pattern", columns="reaction", values="p_adj")
    cols = [r for r in fd.rx_focus if r in piv.columns]
    # order rows so the pattern written for a reaction sits next to it
    rows_order = [D.ASSAY_TO_HOT[r] for r in cols
                  if r in D.ASSAY_TO_HOT and D.ASSAY_TO_HOT[r] in piv.index]
    seen: set = set()
    rows_order = [h for h in rows_order if not (h in seen or seen.add(h))]
    rows_order += [h for h in piv.index if h not in rows_order]
    piv = piv.loc[rows_order, cols]
    padj = padj.loc[rows_order, cols]

    vmax = float(np.nanmax(np.abs(piv.values)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(piv.values, cmap=S.diverging_cmap(), norm=norm,
                   aspect="auto")

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(rows_order)))
    ax.set_yticklabels([h.replace("hot_", "") for h in rows_order])
    ax.set_xlabel("observed reaction")
    ax.set_ylabel("HotSMARTS pattern")
    ax.set_title("Pattern–reaction association (log$_2$ odds ratio)", loc="left")

    for i, hot in enumerate(rows_order):
        for j, rx in enumerate(cols):
            star = S.sig_stars(padj.loc[hot, rx])
            if star:
                ax.text(j, i, star, ha="center", va="center",
                        fontsize=plt.rcParams["font.size"] - 1.5,
                        color="black")
            if D.ASSAY_TO_HOT.get(rx) == hot:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor="black",
                                           linewidth=1.0, zorder=4))
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    cb = plt.colorbar(im, ax=ax, fraction=0.030, pad=0.015)
    cb.set_label("log$_2$ OR", fontsize=plt.rcParams["font.size"] - 0.5)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=plt.rcParams["font.size"] - 1.5)
    return tests


# -----------------------------------------------------------------------------
# (b) matched pairs
# -----------------------------------------------------------------------------
def panel_matched_pairs(ax, tests: pd.DataFrame) -> pd.DataFrame:
    m = tests[tests["is_matching_pair"]].copy()
    m = m.sort_values("log2_or")
    y = np.arange(len(m))
    colours = [S.C_GREEN if (np.isfinite(p) and p < FDR_ALPHA) else S.C_LGREY
               for p in m["p_adj"]]
    ax.barh(y, m["log2_or"], color=colours, height=0.7)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(m["reaction"])
    ax.set_xlabel("log$_2$ odds ratio")
    ax.set_title("Rule as written", loc="left")

    span = float(np.nanmax(np.abs(m["log2_or"]))) if len(m) else 1.0
    for i, (_, r) in enumerate(m.iterrows()):
        star = S.sig_stars(r["p_adj"])
        if star:
            off = 0.06 * span * (1 if r["log2_or"] >= 0 else -1)
            ax.text(r["log2_or"] + off, i, star, va="center",
                    ha="left" if r["log2_or"] >= 0 else "right",
                    fontsize=plt.rcParams["font.size"] - 1.5)
    ax.set_xlim(-span * 1.35, span * 1.35)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, facecolor=S.C_GREEN,
                      label=f"FDR < {FDR_ALPHA:g}"),
        plt.Rectangle((0, 0), 1, 1, facecolor=S.C_LGREY, label="n.s."),
    ], loc="lower right")
    return m


# -----------------------------------------------------------------------------
# (c) descriptor volcano
# -----------------------------------------------------------------------------
def panel_volcano(ax, desc: pd.DataFrame, biomim: int = 1) -> pd.DataFrame:
    sig = (desc["p_adj"] < FDR_ALPHA) & np.isfinite(desc["p_adj"])
    ax.scatter(desc.loc[~sig, "rho"], desc.loc[~sig, "neglog10_p"],
               s=14, color=S.C_LGREY, linewidths=0, zorder=2)
    ax.scatter(desc.loc[sig, "rho"], desc.loc[sig, "neglog10_p"],
               s=18, color=S.C_VERM, linewidths=0, zorder=3)

    # significance threshold on the raw scale that corresponds to FDR = alpha
    if sig.any():
        thr = desc.loc[sig, "p_value"].max()
        ax.axhline(-np.log10(thr), color=S.C_GREY, lw=0.6, ls="--")
        ax.text(ax.get_xlim()[1], -np.log10(thr), f" FDR {FDR_ALPHA:g}",
                va="bottom", ha="right", fontsize=plt.rcParams["font.size"] - 2,
                color=S.C_GREY)
    ax.axvline(0, color=S.C_GREY, lw=0.5)

    # label the strongest effects only, to keep the panel readable; labels on
    # the right half are right-anchored so they cannot run off the axis
    finite = desc[np.isfinite(desc["rho"])]
    if len(finite):
        pad = 0.18 * (finite["rho"].abs().max() or 0.1)
        ax.set_xlim(finite["rho"].min() - pad, finite["rho"].max() + pad)
    lab = desc.reindex(desc["neglog10_p"].sort_values(ascending=False).index)
    xmid = np.mean(ax.get_xlim())
    for _, r in lab.head(6).iterrows():
        if not np.isfinite(r["rho"]):
            continue
        right = r["rho"] > xmid
        ax.annotate(r["descriptor"], (r["rho"], r["neglog10_p"]),
                    textcoords="offset points",
                    xytext=(-4 if right else 4, 3),
                    ha="right" if right else "left",
                    fontsize=plt.rcParams["font.size"] - 2, color="black")
    ax.set_xlabel("Spearman ρ  (descriptor vs conversion %)")
    ax.set_ylabel("−log$_{10}$ p")
    ax.set_title(f"Descriptors vs {S.LABEL_BIOMIM[biomim]} conversion",
                 loc="left")
    ax.yaxis.grid(True, alpha=0.5)
    ax.set_axisbelow(True)
    return desc


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
def build(fd: D.FigData, outdir: Path = S.FIGDIR, biomim: int = 1,
          formats=S.SAVE_FORMATS) -> None:
    S.apply_style()
    tests = pattern_reaction_tests(fd)
    desc = descriptor_conversion_tests(fd, biomim=biomim)

    fig = plt.figure(figsize=(S.COL2, 5.9))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.22, 1.0],
                  width_ratios=[1.0, 0.78],
                  left=0.110, right=0.965, top=0.935, bottom=0.095,
                  wspace=0.40, hspace=0.72)

    ax_a = fig.add_subplot(gs[0, :])
    panel_heatmap(ax_a, tests, fd)
    S.panel_label(ax_a, "a", dx=-0.075, dy=1.02)

    ax_b = fig.add_subplot(gs[1, 0])
    t_b = panel_matched_pairs(ax_b, tests)
    S.panel_label(ax_b, "b", dx=-0.21, dy=1.03)

    ax_c = fig.add_subplot(gs[1, 1])
    panel_volcano(ax_c, desc, biomim=biomim)
    S.panel_label(ax_c, "c", dx=-0.22, dy=1.03)

    S.save(fig, "figure4_associations", outdir=outdir, formats=formats)
    D.export_table(tests, "fig4a_pattern_reaction_fisher")
    D.export_table(t_b, "fig4b_matched_pairs")
    D.export_table(desc, "fig4c_descriptor_conversion_spearman")


if __name__ == "__main__":
    build(D.load())

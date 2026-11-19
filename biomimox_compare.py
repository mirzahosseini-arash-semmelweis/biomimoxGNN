# biomimox_compare.py
# Assembles every ablation arm into one table and runs the paired comparisons
# that actually answer the question a reviewer will ask: does the GNN do
# anything a fingerprint and fifty lines of scikit-learn does not?
#
# Reads whatever is present in ablation_results/:
#   baselines_summary.csv          arms 0-4
#   gnn_summary_*.csv              arm 5, one file per backbone / variant
#   multitask_summary.csv          arm 6
#
# Writes:
#   ablation_results/ablation_table.csv        tidy: one row per assay x arm
#   ablation_results/ablation_table_wide.csv   CV PR-AUC matrix, assays x arms
#   ablation_results/comparison_stats.csv      paired tests between arms
#   ablation_results/ABLATION_REPORT.md        the table in readable form
#
# The primary comparison metric is repeated-CV out-of-fold PR-AUC, not the
# held-out metric: with ~20 test compounds a single compound flipping moves
# test PR-AUC visibly, while the CV column carries a spread across repeats.
#
# Usage:
#   python biomimox_compare.py

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import biomimox_common as C

PRIMARY = "cv_pr_auc_mean"
PRIMARY_SD = "cv_pr_auc_sd"
SECONDARY = ["cv_roc_auc_mean", "cv_mcc_mean", "cv_balanced_accuracy_mean",
             "test_pr_auc", "test_roc_auc"]

BASELINE_ARMS = ("0_prior", "1_hot_logreg", "2a_ecfp4_logreg", "2b_ecfp4_rf",
                 "3_ecfp4_desc_rf", "4_all_rf")


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def load_all_summaries(results_dir: Path = C.RESULTS_DIR) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    p = results_dir / "baselines_summary.csv"
    if p.exists():
        frames.append(pd.read_csv(p))
        print(f"loaded {p}")

    for path in sorted(glob.glob(str(results_dir / "gnn_summary_*.csv"))):
        frames.append(pd.read_csv(path))
        print(f"loaded {path}")
    if not any("gnn_summary_" in str(f) for f in glob.glob(str(results_dir / "gnn_summary_*.csv"))):
        p = results_dir / "gnn_summary.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
            print(f"loaded {p}")

    p = results_dir / "multitask_summary.csv"
    if p.exists():
        frames.append(pd.read_csv(p))
        print(f"loaded {p}")

    if not frames:
        raise RuntimeError(
            f"No summary files found in {results_dir}. Run biomimox_baselines.py, "
            "biomimox_gnn.py and/or biomimox_multitask.py first."
        )

    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df.drop_duplicates(subset=["target", "arm"], keep="last")
    return df


# -----------------------------------------------------------------------------
# Paired statistics
# -----------------------------------------------------------------------------
def paired_test(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Wilcoxon signed-rank where available, otherwise an exact sign test.

    a and b are paired per-assay scores (a = arm A, b = arm B).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = int(a.size)
    out: Dict[str, float] = {
        "n_pairs": n,
        "mean_diff": float(np.mean(a - b)) if n else float("nan"),
        "median_diff": float(np.median(a - b)) if n else float("nan"),
        "n_a_better": int(np.sum(a > b)),
        "n_b_better": int(np.sum(b > a)),
        "n_ties": int(np.sum(a == b)),
    }
    if n < 3:
        out["test"] = "none (too few pairs)"
        out["p_value"] = float("nan")
        return out

    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        out["test"] = "wilcoxon_signed_rank"
        out["statistic"] = float(stat)
        out["p_value"] = float(p)
        return out
    except Exception:
        pass

    # Exact two-sided sign test, no scipy needed.
    from math import comb
    k = int(np.sum(a > b))
    m = int(np.sum(a != b))
    if m == 0:
        out["test"] = "sign_test"
        out["p_value"] = 1.0
        return out
    tail = sum(comb(m, i) for i in range(0, min(k, m - k) + 1)) / (2 ** m)
    out["test"] = "sign_test_exact"
    out["p_value"] = float(min(1.0, 2.0 * tail))
    return out


def arm_scores(df: pd.DataFrame, arm: str, metric: str = PRIMARY) -> pd.Series:
    sub = df[df["arm"] == arm]
    return sub.set_index("target")[metric] if len(sub) else pd.Series(dtype=float)


def best_baseline_per_target(df: pd.DataFrame, metric: str = PRIMARY) -> pd.DataFrame:
    sub = df[df["arm"].isin(BASELINE_ARMS) & (df["arm"] != "0_prior")].copy()
    if sub.empty:
        return pd.DataFrame(columns=["target", "arm", "arm_label", metric])
    idx = sub.groupby("target")[metric].idxmax()
    return sub.loc[idx, ["target", "arm", "arm_label", metric]].reset_index(drop=True)


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------
def fmt(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def fmt_pm(mean: float, sd: float) -> str:
    if not np.isfinite(mean):
        return "n/a"
    if not np.isfinite(sd):
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {sd:.3f}"


def load_run_config() -> Dict[str, object]:
    """Read the configuration the run actually used, rather than whatever the
    environment happens to say now."""
    import json
    path = C.SPLIT_DIR / "prepare_config.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return C.describe_config()


def build_report(df: pd.DataFrame,
                 wide: pd.DataFrame,
                 stats: pd.DataFrame,
                 negative_transfer: pd.DataFrame) -> str:
    cfg = load_run_config()
    n_assays = int(df["target"].nunique())
    lines: List[str] = []
    lines.append("# Biomimetic oxidation: ablation results\n")
    lines.append(f"Split mode: `{cfg.get('split_mode', C.SPLIT_MODE)}` | CV: "
                 f"{cfg.get('cv_splits', C.CV_SPLITS)}-fold x "
                 f"{cfg.get('cv_repeats', C.CV_REPEATS)} repeats | conformers: "
                 f"{cfg.get('n_conformers', C.N_CONFORMERS)}\n")
    lines.append("Primary metric: out-of-fold PR-AUC from repeated CV on the "
                 "training pool, mean ± SD across repeats. The held-out columns "
                 "are a sanity check, not the comparison: the test set is too "
                 "small for a single PR-AUC to be stable.\n")

    arms = (df[["arm", "arm_label"]].drop_duplicates()
            .sort_values("arm").reset_index(drop=True))
    lines.append("## Arms\n")
    for _, r in arms.iterrows():
        lines.append(f"- `{r['arm']}` — {r['arm_label']}")
    lines.append("")

    lines.append("## CV PR-AUC by assay\n")
    header = "| assay | prevalence | " + " | ".join(arms["arm"].tolist()) + " | winner |"
    lines.append(header)
    lines.append("|" + "---|" * (len(arms) + 3))
    for target in wide.index:
        prev = df[df["target"] == target]["prevalence"]
        prev_s = fmt(float(prev.iloc[0]) if len(prev) else np.nan, 2)
        cells = []
        for arm in arms["arm"]:
            sub = df[(df["target"] == target) & (df["arm"] == arm)]
            if len(sub) == 0:
                cells.append("—")
            else:
                cells.append(fmt_pm(float(sub[PRIMARY].iloc[0]),
                                    float(sub[PRIMARY_SD].iloc[0])
                                    if PRIMARY_SD in sub.columns else np.nan))
        row = wide.loc[target]
        winner = row.idxmax() if row.notna().any() else "—"
        lines.append(f"| {target} | {prev_s} | " + " | ".join(cells) + f" | {winner} |")
    lines.append("")

    if not stats.empty:
        lines.append("## Paired comparisons across assays\n")
        lines.append("| comparison | n | median Δ PR-AUC | A better | B better | test | p |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in stats.iterrows():
            lines.append(
                f"| {r['arm_a']} vs {r['arm_b']} | {int(r['n_pairs'])} | "
                f"{fmt(r['median_diff'])} | {int(r['n_a_better'])} | "
                f"{int(r['n_b_better'])} | {r['test']} | {fmt(r['p_value'], 4)} |"
            )
        lines.append("")
        lines.append(f"Δ is arm A minus arm B, so a positive median favours A. "
                     f"With {n_assays} assays these tests have little power "
                     f"(the smallest attainable two-sided p is around "
                     f"{2.0 / (2 ** n_assays):.3g}); treat them as a direction "
                     f"indicator, not proof.\n")

    if not negative_transfer.empty:
        lines.append("## Multitask vs single-task, per task\n")
        lines.append("Reported per task on purpose: an improved average can hide "
                     "individual tasks getting worse.\n")
        lines.append("| task | single-task | multitask | Δ | verdict |")
        lines.append("|---|---|---|---|---|")
        for _, r in negative_transfer.iterrows():
            lines.append(f"| {r['target']} | {fmt(r['single'])} | {fmt(r['multi'])} | "
                         f"{fmt(r['delta'])} | {r['verdict']} |")
        n_worse = int((negative_transfer["verdict"] == "negative transfer").sum())
        n_better = int((negative_transfer["verdict"] == "positive transfer").sum())
        lines.append("")
        lines.append(f"Positive transfer on {n_better} task(s), negative transfer "
                     f"on {n_worse} task(s), the rest within noise.\n")

    # Headline reading of the numbers, stated plainly.
    lines.append("## How to read this\n")
    bb = best_baseline_per_target(df)
    gnn_arms = [a for a in df["arm"].unique() if str(a).startswith("5_")]
    mt_arms = [a for a in df["arm"].unique() if str(a).startswith("6_")]
    if len(bb) and (gnn_arms or mt_arms):
        best_gnn_arm = None
        best_mean = -np.inf
        for a in gnn_arms + mt_arms:
            s = arm_scores(df, a)
            m = float(np.nanmean(s.values)) if len(s) else np.nan
            if np.isfinite(m) and m > best_mean:
                best_mean, best_gnn_arm = m, a
        bb_mean = float(np.nanmean(bb[PRIMARY].values))
        lines.append(f"- Mean CV PR-AUC across assays: best baseline per assay "
                     f"{fmt(bb_mean)}, best neural arm (`{best_gnn_arm}`) {fmt(best_mean)}.")
        if np.isfinite(bb_mean) and np.isfinite(best_mean) and bb_mean >= best_mean:
            lines.append("- On this dataset the fingerprint baseline is at least as good "
                         "as the graph models. That is a legitimate and useful finding: "
                         "at ~80 training compounds per task the chemistry is learnable "
                         "from topology alone, and the curated dataset is the contribution "
                         "to lead with.")
        else:
            lines.append("- The graph models lead on average. Check the per-assay column "
                         "and the SD before claiming the difference is real; a gap smaller "
                         "than the repeat-to-repeat spread is not evidence.")
    lines.append("- `0_prior` is the floor. Any arm near it is not learning this assay.")
    lines.append("- Compare `1_hot_logreg` with `2b_ecfp4_rf` to see how much of the "
                 "signal the hand-coded SMARTS already carry, and `3_ecfp4_desc_rf` "
                 "with `4_all_rf` to see whether they add anything over fingerprints.")
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    C.set_workdir()
    C.ensure_dirs()

    df = load_all_summaries()
    print(f"\n{len(df)} assay x arm rows, "
          f"{df['target'].nunique()} assays, {df['arm'].nunique()} arms")

    keep_cols = [c for c in ["target", "arm", "arm_label", "family", "backbone",
                             "n_train", "n_test", "prevalence", "n_features",
                             "n_params", PRIMARY, PRIMARY_SD] + SECONDARY
                 if c in df.columns]
    tidy = df[keep_cols].sort_values(["target", "arm"]).reset_index(drop=True)
    tidy.to_csv(C.RESULTS_DIR / "ablation_table.csv", index=False)
    print(f"Wrote {C.RESULTS_DIR / 'ablation_table.csv'}")

    wide = tidy.pivot_table(index="target", columns="arm", values=PRIMARY, aggfunc="first")
    # Keep the original assay order rather than alphabetical.
    order = [t for t in C.TARGET_COLUMNS if t in wide.index]
    wide = wide.loc[order + [t for t in wide.index if t not in order]]
    wide.to_csv(C.RESULTS_DIR / "ablation_table_wide.csv")
    print(f"Wrote {C.RESULTS_DIR / 'ablation_table_wide.csv'}")

    # ---- paired comparisons ----
    comparisons: List[Tuple[str, str]] = []
    gnn_arms = sorted([a for a in df["arm"].unique() if str(a).startswith("5_")])
    mt_arms = sorted([a for a in df["arm"].unique() if str(a).startswith("6_")])

    bb = best_baseline_per_target(df)
    stats_rows: List[Dict] = []

    if len(bb) and gnn_arms:
        bb_series = bb.set_index("target")[PRIMARY]
        for a in gnn_arms + mt_arms:
            s = arm_scores(df, a)
            common = s.index.intersection(bb_series.index)
            if len(common) == 0:
                continue
            res = paired_test(s.loc[common].values, bb_series.loc[common].values)
            res.update({"arm_a": a, "arm_b": "best_baseline_per_assay",
                        "metric": PRIMARY})
            stats_rows.append(res)

    for a in ("4_all_rf", "2b_ecfp4_rf"):
        for b in ("1_hot_logreg", "2a_ecfp4_logreg", "0_prior"):
            if a in df["arm"].values and b in df["arm"].values:
                comparisons.append((a, b))
    for a in mt_arms:
        for b in gnn_arms:
            comparisons.append((a, b))
    for i, a in enumerate(gnn_arms):
        for b in gnn_arms[i + 1:]:
            comparisons.append((a, b))

    for a, b in comparisons:
        sa, sb = arm_scores(df, a), arm_scores(df, b)
        common = sa.index.intersection(sb.index)
        if len(common) == 0:
            continue
        res = paired_test(sa.loc[common].values, sb.loc[common].values)
        res.update({"arm_a": a, "arm_b": b, "metric": PRIMARY})
        stats_rows.append(res)

    stats = pd.DataFrame(stats_rows)
    if not stats.empty:
        front = ["arm_a", "arm_b", "metric", "n_pairs", "median_diff", "mean_diff",
                 "n_a_better", "n_b_better", "n_ties", "test", "p_value"]
        stats = stats[[c for c in front if c in stats.columns]
                      + [c for c in stats.columns if c not in front]]
        stats.to_csv(C.RESULTS_DIR / "comparison_stats.csv", index=False)
        print(f"Wrote {C.RESULTS_DIR / 'comparison_stats.csv'}")

    # ---- negative transfer, per task ----
    nt_rows: List[Dict] = []
    if mt_arms and gnn_arms:
        mt = arm_scores(df, mt_arms[0])
        st = arm_scores(df, gnn_arms[0])
        st_sd = df[df["arm"] == gnn_arms[0]].set_index("target")[PRIMARY_SD] \
            if PRIMARY_SD in df.columns else None
        for target in mt.index.intersection(st.index):
            d = float(mt.loc[target]) - float(st.loc[target])
            noise = float(st_sd.loc[target]) if st_sd is not None and target in st_sd.index else np.nan
            tol = noise if np.isfinite(noise) else 0.02
            verdict = ("positive transfer" if d > tol else
                       "negative transfer" if d < -tol else "within noise")
            nt_rows.append({"target": target, "single": float(st.loc[target]),
                            "multi": float(mt.loc[target]), "delta": d,
                            "single_sd": noise, "verdict": verdict})
    negative_transfer = pd.DataFrame(nt_rows)
    if not negative_transfer.empty:
        negative_transfer.to_csv(C.RESULTS_DIR / "multitask_vs_singletask.csv", index=False)
        print(f"Wrote {C.RESULTS_DIR / 'multitask_vs_singletask.csv'}")

    report = build_report(tidy, wide, stats, negative_transfer)
    report_path = C.RESULTS_DIR / "ABLATION_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}\n")

    print("CV PR-AUC matrix (assays x arms):")
    print(wide.round(3).to_string())

    if not bb.empty:
        print("\nBest baseline per assay:")
        for _, r in bb.iterrows():
            print(f"  {r['target']:<10s} {r['arm_label']:<38s} {fmt(r[PRIMARY])}")

    if not stats.empty:
        print("\nPaired comparisons:")
        for _, r in stats.iterrows():
            print(f"  {r['arm_a']:<24s} vs {r['arm_b']:<26s} "
                  f"median Δ={fmt(r['median_diff']):>7s}  "
                  f"{int(r['n_a_better'])}:{int(r['n_b_better'])}  "
                  f"p={fmt(r['p_value'], 4)}")


if __name__ == "__main__":
    main()

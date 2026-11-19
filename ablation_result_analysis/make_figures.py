#!/usr/bin/env python
# make_figures.py -- build every manuscript figure in one go.
#
#   python make_figures.py                 # all five figures
#   python make_figures.py --only 1 3      # just those
#   python make_figures.py --formats pdf tiff svg
#   python make_figures.py --data /path/to/data.csv --outdir figures_v2
#
# Each figure also writes the numbers behind its panels to figure_tables/, so
# every value in the manuscript can be traced back to a CSV.

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

import biomimox_figdata as D
import figstyle as S


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the manuscript figures")
    p.add_argument("--data", default=str(D.DATA_FILE),
                   help="path to data_20260611.csv")
    p.add_argument("--fda", default=str(D.FDA_FILE),
                   help="optional FDA reference set (columns: name, smiles)")
    p.add_argument("--ablation", default=str(D.ABLATION_TABLE),
                   help="path to ablation_table.csv")
    p.add_argument("--outdir", default=str(S.FIGDIR),
                   help="output directory for the figures")
    p.add_argument("--tables", default=str(D.TABLEDIR),
                   help="output directory for the per-panel CSVs")
    p.add_argument("--formats", nargs="+", default=list(S.SAVE_FORMATS),
                   help="output formats, e.g. pdf png tiff svg")
    p.add_argument("--only", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                   choices=[1, 2, 3, 4, 5], help="which figures to build")
    p.add_argument("--biomim-reactions", type=int, default=None,
                   choices=[0, 1],
                   help="restrict Figure 1d to one condition "
                        "(default: observed in either)")
    p.add_argument("--fontsize", type=float, default=7.0,
                   help="base font size in points")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    D.TABLEDIR = Path(args.tables)
    outdir = Path(args.outdir)
    S.apply_style(args.fontsize)

    print("=" * 72)
    print("Biomimetic oxidation -- manuscript figures")
    print("=" * 72)

    fd = D.load(args.data)
    fda = None
    if 2 in args.only:
        fda = D.load_fda(args.fda)

    tab = None
    if 5 in args.only:
        try:
            tab = D.load_ablation(args.ablation)
        except FileNotFoundError as exc:
            print(f"  skipping Figure 5: {exc}")

    counts, placeholder = fd.workflow_counts()
    if placeholder and 1 in args.only:
        print("\n" + "!" * 72)
        print("! Figure 1a uses PLACEHOLDER HRMS/NMR counts.")
        print("! Set WORKFLOW_COUNTS in biomimox_figdata.py before submission.")
        print("!" * 72)

    failures = []
    for n in sorted(args.only):
        t0 = time.time()
        print(f"\n--- Figure {n} ---")
        try:
            if n == 1:
                import fig1_descriptive as f1
                f1.build(fd, outdir=outdir,
                         biomim_for_reactions=args.biomim_reactions,
                         formats=args.formats)
            elif n == 2:
                import fig2_chemspace as f2
                f2.build(fd, fda, outdir=outdir, formats=args.formats)
            elif n == 3:
                import fig3_expected_observed as f3
                f3.build(fd, outdir=outdir, formats=args.formats)
            elif n == 4:
                import fig4_associations as f4
                f4.build(fd, outdir=outdir, formats=args.formats)
            elif n == 5:
                if tab is None:
                    print("    no ablation table; skipped")
                    continue
                import fig5_ablation as f5
                f5.build(tab, outdir=outdir, formats=args.formats)
            print(f"    done in {time.time() - t0:.1f}s")
        except Exception:
            failures.append(n)
            print(f"    FAILED after {time.time() - t0:.1f}s")
            traceback.print_exc()

    print("\n" + "=" * 72)
    if failures:
        print(f"Finished with failures in figure(s): {failures}")
        return 1
    print(f"All requested figures written to {outdir}/ "
          f"(panel data in {D.TABLEDIR}/)")
    if placeholder and 1 in args.only:
        print("Reminder: Figure 1a HRMS/NMR counts are still placeholders.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Manuscript figures — biomimetic oxidation

Eight Python files that build five manuscript-ready figures from
`data_20260611.csv` and `ablation_results/ablation_table.csv`.

```bash
pip install rdkit umap-learn statsmodels scipy scikit-learn matplotlib pandas pillow

python make_figures.py                      # all five, PDF + PNG
python make_figures.py --only 1 3           # just those
python make_figures.py --formats pdf tiff   # journal asked for TIFF
python make_figures.py --data /path/to/data.csv --outdir figures_v2
```

Outputs land in `figures/`, and **the numbers behind every panel** go to
`figure_tables/` as CSVs, so any value in the manuscript traces back to a file.

| file | role |
|---|---|
| `figstyle.py` | one place for fonts, palette, panel labels, structure rendering |
| `biomimox_figdata.py` | data model and every derived count — the single source of truth |
| `fig1_descriptive.py` | study flow, metabolites/compound, conversion categories, reaction counts |
| `fig2_chemspace.py` | UMAP, HotSMARTS structures, scaffolds, alluvial |
| `fig3_expected_observed.py` | expected vs observed vs in vivo (replaces the sunburst) |
| `fig4_associations.py` | HotSMARTS→reaction Fisher tests, descriptor volcano |
| `fig5_ablation.py` | ablation summary with confidence intervals |
| `make_figures.py` | runner |
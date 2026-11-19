# Methods — statistics, scaffold analysis, machine learning

## Dataset processing and definitions

The curated dataset contains two row types. Conversion rows record the fractional conversion of the parent compound, one per (compound, pH, catalyst) combination. Metabolite rows record the abundance of each catalogued metabolite on the full (compound, metabolite) × pH × catalyst grid; reaction-type annotations, metabolite generation and in vivo status are properties of the (compound, metabolite) pair and are constant across that grid, while abundance varies. A metabolite was counted as **observed** under a given condition when its abundance exceeded zero, and as **catalogued** when the pair exists at all. Unless stated otherwise, metabolite counts are distinct (compound, metabolite) pairs rather than grid rows.

Structures were parsed with RDKit, reduced to the largest covalent fragment to remove counter-ions, and converted to canonical SMILES. Metabolite generation was unassigned for 257 of 435 observed metabolites. In vivo status was recorded as confirmed, probable (occurrence likely but exact structure unresolved) or no data, with the field empty for conversion rows by construction; absence of annotation was treated as missing rather than as evidence against in vivo occurrence.

## Molecular representations, reactivity patterns and scaffold analysis

**Descriptors and fingerprints.** Morgan fingerprints (ECFP4; radius 2, 2,048 bits) were generated with the RDKit Morgan generator. Two descriptor sets were used and should not be conflated. The descriptor–conversion analysis used fifteen RDKit descriptors: molecular weight, cLogP, TPSA, hydrogen-bond acceptor and donor counts, rotatable-bond count, ring count, aromatic-ring count, heavy-atom count, fraction sp³ carbon, heteroatom count, saturated-ring count, Labute approximate surface area, Balaban J and Bertz complexity. The machine-learning arms used a 25-dimensional global feature vector: nine of those RDKit descriptors (molecular weight, cLogP, TPSA, acceptors, donors, rotatable bonds, rings, heavy atoms, fraction sp³), twelve HotSMARTS indicator flags and four ACD/Percepta-calculated properties (molar refractivity, logD, and counts of negatively and positively ionizable centers). Missing Percepta values were imputed with the column mean for the continuous properties and zero for the ionization counts.

**Reactivity patterns.** Each oxidation type was encoded as one or more SMARTS patterns describing the substructure at which the transformation can occur (HotSMARTS; full definitions in biomimox_common.py). Twelve patterns were defined, covering aromatic and aliphatic C–H oxidation, benzylic oxidation, dehydrogenation, N- and O-dealkylation, N- and O-dearylation, N-, S- and alcohol oxidation, and a composite unspecified-oxidation pattern formed as the union of the aromatic, benzylic, N-oxidation, aliphatic and S-oxidation patterns. A compound was scored positive for a pattern when any of its constituent SMARTS matched. Oxidative cleavage has no pattern and is excluded from pattern-based analyses. The identical pattern definitions were used for the figures and for every model arm, so matches are directly comparable across analyses.

**Chemical space.** ECFP4 bit vectors for the study compounds and the FDA approved-drug reference set were reduced by principal component analysis to at most 50 components and embedded in two dimensions with UMAP (n\_neighbors = 25, min\_dist = 0.15, Euclidean metric, random\_state = 42). PCA preprocessing was applied to stabilize the embedding of sparse binary vectors.

**Scaffold analysis.** Bemis–Murcko scaffolds were computed per compound in their standard (non-generic) form, retaining heteroatoms and bond orders. The generic form, which collapses every ring to a carbocycle, was not used: on a drug-like set of this size it merges most compounds into a single benzene-shaped group and makes scaffold-based splitting impossible. Fused ring systems were additionally extracted by merging rings sharing at least one atom and retained when they contained five or more heavy atoms. Scaffold diversity is reported as the number of distinct scaffolds, the number occurring in exactly one compound, and the number of distinct ring systems.

## Statistical analysis

**Pattern–reaction association.** For every HotSMARTS pattern and every observed reaction type, compounds were cross-tabulated by pattern match and by whether the compound yielded any metabolite carrying that reaction, pooling across pH and catalyst condition. Association was tested with Fisher's exact test (two-sided) at compound level. For display, odds ratios were computed with the Haldane–Anscombe correction (0.5 added to each cell) so that tables containing a zero remain plottable; the uncorrected odds ratios, which are infinite in five cases, are given in the source data. P values were adjusted across the full pattern × reaction matrix by the Benjamini–Hochberg procedure, and significance is reported at a false discovery rate of 0.05.

**Descriptor–conversion correlation.** Conversion was summarized as the mean across the four pH values for each compound under biomimetic conditions, so that each compound contributes once and the test n is the number of compounds rather than the number of compound × pH observations. Each descriptor was correlated with this value by Spearman's rank correlation, with Benjamini–Hochberg correction across the fifteen descriptors.

**Comparison of model arms.** Arms were compared on cross-validated out-of-fold PR-AUC, paired by reaction type, using the Wilcoxon signed-rank test across the twelve endpoints. With twelve pairs the smallest attainable two-sided P is 4.9 × 10⁻⁴, and these tests are reported as directional evidence rather than as confirmatory inference.

**Uncertainty.** Confidence intervals on cross-validated metrics are t-based 95% intervals over independent cross-validation repeats (mean ± t₀.₉₇₅,ₙ₋₁ × s/√n). Because the held-out test set contains 25–26 compounds and as few as one positive for the rarest endpoints, single held-out metrics are reported only as a consistency check; all model comparisons rest on the repeated cross-validation estimates. Prevalence-adjusted lift is PR-AUC minus the positive-class prevalence of that endpoint; it leaves rankings within an endpoint unchanged and makes averages across endpoints interpretable. Analyses used two-sided tests throughout; no adjustment was made for comparisons across the separate analyses described above.

## Machine-learning benchmark

**Endpoints.** Twelve binary endpoints were modelled: eleven reaction types and overall parent conversion. A compound was labelled positive for a reaction type when it yielded at least one metabolite carrying that annotation under any condition. Endpoints with fewer than two compounds in either class were excluded.

**Data splitting.** A single compound-level train/test partition was computed once and inherited by every arm, so that a compound falls on the same side of the split in all comparisons. Compounds were grouped by Bemis–Murcko scaffold and whole groups were assigned to an approximately 20% test set, stratified on the number of reaction types each compound exhibits (binned at 0, 1, 2, ≥3). This guarantees that no scaffold appears in both partitions. Where the shared split left an endpoint with no positive on one side, that endpoint received its own scaffold split and is flagged in the source data. Random splitting was implemented and is available as an option, but all reported results use scaffold splitting; random splits place near-neighbors of every training compound in the test set and give optimistic estimates.

**Cross-validation.** Within the training partition, models were evaluated by repeated stratified group k-fold cross-validation (five folds, five repeats, grouped by scaffold so that fold assignment also respects scaffold boundaries). Decision thresholds were chosen to maximise F₁ on pooled out-of-fold training predictions only, never on validation or test data. Each arm performed its own nested hyperparameter selection on inner folds (three-fold, average-precision scoring) so that no arm benefits from tuning the others did not receive.

**Arms.** Eight arms were compared, all on identical compounds, splits and folds:

| Arm | Features | Model |
| --- | --- | --- |
| 0 | — | stratified dummy classifier (class prior) |
| 1 | 12 HotSMARTS flags | L2 logistic regression |
| 2a | ECFP4 (2,048 bits) | L2 logistic regression |
| 2b | ECFP4 (2,048 bits) | random forest |
| 3 | ECFP4 + 9 RDKit descriptors + 4 Percepta properties | random forest |
| 4 | arm 3 + 12 HotSMARTS flags | random forest |
| 5 | molecular graph and 3D conformers | single-task graph neural network |
| 6 | molecular graph and 3D conformers | multitask graph neural network |

Logistic regression used the liblinear solver with balanced class weights and an inner grid over C ∈ {0.01, 0.1, 1, 10}. Random forests used 500 trees with balanced subsample weighting and an inner grid over maximum features {√, 0.1} and minimum leaf size {1, 3}. Continuous features were standardised within the cross-validation fold for the linear models; tree models used raw values.

**Metrics.** The primary metric is PR-AUC, which is appropriate for the low and variable prevalence of these endpoints; ROC-AUC, F₁, Matthews correlation coefficient, balanced accuracy, sensitivity and specificity were also recorded and are reported in the source data.

## Graph neural network architecture and training

**Conformers.** Up to five conformers per molecule were generated with ETKDGv3 (RMS pruning threshold 0.5 Å), optimised with MMFF94s where parameterisation succeeded and UFF otherwise, and retained in order of increasing force-field energy. During training one conformer was sampled per molecule per epoch as data augmentation; at evaluation and prediction the predicted probability was averaged over the full conformer ensemble. A single conformer per molecule is available as an option and was not used for the reported results.

**Graph construction.** Each conformer gave one graph. Node features (32 dimensions) encode element one-hot over the ten elements present, hybridization, aromaticity, total degree, attached hydrogen count, ring membership including ring sizes 3, 5 and 6, formal charge and radical electron count. Edges are the covalent bonds of the molecule, with 9-dimensional bond features encoding bond order, conjugation, ring membership and stereochemistry. Hydrogens were retained. The three-dimensional backbones construct their own radius-based neighbour lists from atomic coordinates internally; the explicit edge list is the bond graph, so bond order and conjugation are not discarded in favour of interatomic distances.

**Architecture.** The model is a dual encoder. A three-dimensional branch (DimeNet++; 4 interaction blocks, 5 Å cutoff, maximum 32 neighbours, 7 spherical and 6 radial basis functions, interaction embedding 64, basis embedding 8) consumes atomic numbers and coordinates. A two-dimensional branch of three GINE convolution layers with residual connections and batch normalisation consumes the node and bond features over the bond graph and is pooled by concatenated sum and mean pooling. The two embeddings and the layer-normalised 25-dimensional global feature vector are concatenated and passed to a two-layer prediction head. SchNet and a two-dimensional-only variant are implemented as alternative backbones and were not used for the reported results.

**Training.** Models were trained with AdamW, gradient-norm clipping at 5.0, and binary cross-entropy with positive-class weighting computed from the training folds. Class imbalance was additionally addressed by a weighted random sampler in the single-task arm. Global features were standardised using statistics computed on the inner training fold only. The hyperparameter grid comprised three configurations spanning hidden width (32, 64), batch size (8, 16), learning rate (5 × 10⁻⁴, 10⁻³), weight decay and dropout.

**Avoiding optimistic bias.** Configuration selection and reported evaluation were separated. The first cross-validation repeat selected the configuration and, by early stopping on validation PR-AUC, an epoch budget. The remaining repeats — which use different fold assignments — were trained for exactly that fixed number of epochs with no reference to the validation fold, so no reported out-of-fold prediction comes from a model whose stopping point was tuned on the fold being scored. On permuted labels this protocol returns PR-AUC at prevalence and ROC-AUC near 0.5, whereas early stopping on the scored fold does not.

**Multitask variant.** The multitask arm shares one encoder across eleven reaction-type heads, excluding overall conversion, which is a different kind of endpoint. The loss is masked binary cross-entropy, so a compound lacking an annotation for a given reaction contributes nothing to that head rather than a spurious negative. Positive-class weights and decision thresholds are computed per head. Results are reported per task, since an improved average across heads can conceal negative transfer on individual tasks.

## Software and reproducibility

Analyses were performed in Python 3.11.9 with package versions in requirements.txt. Physicochemical properties were calculated with ACD/Percepta version 2024. Random seeds were fixed at 42 for structure embedding, data splitting, cross-validation and model initialisation; cross-validation repeats use derived seeds so that repeats are independent while the whole pipeline remains deterministic.

The splitting, model-fitting, statistical and figure-generation code is available in the provided files, together with the per-panel source-data tables from which every reported value and every figure can be regenerated. Conformer generation, model training and figure rendering are driven by a single entry point per stage, so the full analysis reproduces from the curated dataset without manual steps.

# Biomimetic oxidation: ML pipeline and ablation suite

The following scripts parse and preprocess the input file `raw_data/data_20260611.csv`

```r
convert_data_to_inputCSV.R          clean, restructure, and mutate data
merge_percepta_data.R               reads raw files from percepta_files/ and creates percepta_desc.csv
```

The subsequent scripts create the full ablation suite (arms 0–6) wired up on identical splits.
As input they take:

```
input_files/SMILESforGNN.csv        column: SMILES
input_files/percepta_desc.csv       refractivity, logD, neg, pos  (SMILES optional)
input_files/<assay>_outcomes.csv    column: <assay>, 0/1
predict_new/SMILES_new.csv          for prediction
predict_new/percepta_new.csv
```

## Run order

```bash
# 0. once: conformers + the one global split every arm inherits
python biomimox_prepare.py

# 1. the bar the GNN has to clear  (arms 0-4; the most valuable single step)
python biomimox_baselines.py

# 2. single-task 3-D GNN  (arm 5)
python biomimox_gnn.py

# 3. multitask, one head per reaction type  (arm 6)
python biomimox_multitask.py

# 4. the comparison table and the paired statistics
python biomimox_compare.py

# 5. predict new compounds from either bundle type
python biomimox_predict.py
```

Set the working directory once:

```bash
export BIOMIMOX_DIR="C:/path/to/your/workdir"
```

## Files

| file | role |
|---|---|
| `biomimox_common.py` | featurisation, SMARTS, descriptors, ECFP4, splits, repeated CV, metrics, applicability domain |
| `biomimox_graphs.py` | PyG `Data` construction, conformer datasets, graph cache |
| `biomimox_models.py` | dual encoder, three backbones, masked losses |
| `biomimox_prepare.py` | step 0: conformer cache + the authoritative splits |
| `biomimox_baselines.py` | arms 0–4 (sklearn only, no torch needed) |
| `biomimox_gnn.py` | arm 5, single-task |
| `biomimox_multitask.py` | arm 6, multitask |
| `biomimox_compare.py` | the ablation table, paired tests, negative-transfer count |
| `biomimox_predict.py` | prediction for new compounds |

## The ablation suite

| arm | features | model |
|---|---|---|
| `0_prior` | — | stratified dummy (the floor) |
| `1_hot_logreg` | 12 HOT_SMARTS flags | L2 logistic |
| `2a_ecfp4_logreg` | ECFP4, 2048 bit | L2 logistic |
| `2b_ecfp4_rf` | ECFP4, 2048 bit | random forest |
| `3_ecfp4_desc_rf` | ECFP4 + 9 RDKit + 4 Percepta | random forest |
| `4_all_rf` | arm 3 + HOT_SMARTS | random forest |
| `5_gnn_*` | full graph | single-task DimeNet++ / SchNet / GINE |
| `6_multitask_*` | full graph | multitask, one head per reaction |

Every arm gets the same molecules, the same split, the same repeated-CV
protocol, its own nested hyper-parameter selection, and its own F1-optimal
threshold chosen on out-of-fold training predictions only.

## Multitask

Shared encoder, one sigmoid head per reaction type, masked BCE so a molecule
with no label for a given reaction contributes nothing to that head. Per-head
`pos_weight` and per-head thresholds, both computed on training data only.
`percent` is excluded by default (overall turnover, not one reaction type;
`BIOMIMOX_MT_INCLUDE_PERCENT=1` includes it). Tasks with fewer than 3 positives
or 3 negatives are dropped with a printed reason.

Results are reported **per task, never as a single average** — with 11 heads and
~100 compounds an improved mean can easily hide two tasks getting worse.
`biomimox_compare.py` writes `multitask_vs_singletask.csv` labelling each task
positive transfer / negative transfer / within noise, using the single-task
repeat SD as the noise threshold.

## Backbones

`BACKBONE` is DimeNet++ by default.

```bash
python biomimox_gnn.py                             # DimeNet++  (default)
BIOMIMOX_BACKBONE=schnet python biomimox_gnn.py    # faster
BIOMIMOX_BACKBONE=none   python biomimox_gnn.py    # 2-D GINE only, seconds
BIOMIMOX_USE_2D=0 python biomimox_gnn.py           # geometry only, no bond branch
```

## Environment variables

| variable | default | effect |
|---|---|---|
| `BIOMIMOX_DIR` | finds parent path | working directory |
| `BIOMIMOX_SPLIT` | `scaffold` | `random` restores the original split |
| `BIOMIMOX_REPEATS` | `5` | CV repeats; repeat 0 selects, the rest report |
| `BIOMIMOX_NCONF` | `5` | conformers per molecule |
| `BIOMIMOX_BACKBONE` | `dimenet++` | `schnet`, `none` |
| `BIOMIMOX_USE_2D` | `1` | `0` disables the bond-graph branch |
| `BIOMIMOX_EPOCHS` | `120` / `150` | epoch ceiling |
| `BIOMIMOX_MAX_CONFIGS` | all 4 | trim the grid for a smoke test |
| `BIOMIMOX_LOSS` | `bce` | `focal` |
| `BIOMIMOX_N_JOBS` | `-1` | baseline parallelism |
| `BIOMIMOX_RF_TREES` | `500` | forest size |
| `BIOMIMOX_GENERIC_SCAFFOLD` | `0` | `1` uses generic Murcko scaffolds |
| `BIOMIMOX_TIME_BUDGET_H` | `8` | when to suggest SchNet |

Smoke test before committing to a full run:

```bash
BIOMIMOX_NCONF=2 BIOMIMOX_REPEATS=2 BIOMIMOX_MAX_CONFIGS=1 \
BIOMIMOX_EPOCHS=10 BIOMIMOX_BACKBONE=none python biomimox_prepare.py && python biomimox_gnn.py
```
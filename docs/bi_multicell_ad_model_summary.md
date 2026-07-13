# BI multicell AD representation model summary

Remote source:

`${OMNICELL_NVU_RESULTS_ROOT}/04_NVU_ad_model/02_Result/multicell_integrated_ad_model`

BI copy:

`${OMNICELL_NVU_ROOT}/projects/BI/results/multicell_integrated_ad_model`

Local copy:

`${LOCAL_USER_HOME}\Documents\链接武超-NVU AI\projects\BI\results\multicell_integrated_ad_model`

## Core conclusion

The defensible BI figure claim should be:

`Sample-level multicell/NVU aggregation of CPT representations separates AD from Control with strong held-out performance, and the AD axis can be interpreted through glial, vascular, mural, neuronal, LR, region, and gene-module features.`

Do not frame this as cell-level AUROC. The statistical unit is sample/chip. Individual NVUs/cells are aggregated into sample-level feature distributions.

## Held-out AD/Control discrimination

Two validation sets were reported:

- `all44`: all 44 sample/tissue entries from `all_results_v2.pkl`, balanced AD/Control 22/22.
- `qc36_risk_available`: QC-filtered 36 samples matching the existing Figure7 risk embedding/prediction table, AD/Control 21/15.

Main AUROC results:

| dataset | feature set | AUROC | note |
|---|---:|---:|---|
| all44 | CPT latent | 0.893 | strongest all-sample baseline |
| all44 | AD-informed static | 0.893 | CPT + scalar modules + curated genes/LR/regions |
| all44 | AD-informed CPT | 0.872 | CPT + curated modules + fold-internal selected genes |
| all44 | raw gene SVD | 0.731 | raw expression baseline |
| qc36 | CPT latent | 0.981 | strongest held-out result |
| qc36 | AD-informed static | 0.917 | interpretable integrated model |
| qc36 | AD-informed CPT | 0.914 | interpretable plus fold-selected genes |
| qc36 | scalar + region + risk | 0.848 | risk-augmented interpretable layer |
| qc36 | raw gene SVD | 0.698 | raw expression baseline |

Figure wording recommendation:

`AD-informed CPT features retained high held-out AD/Control discrimination while adding interpretable NVU/glial disease modules; on the QC-filtered cohort, CPT latent features achieved AUROC 0.981 and the AD-informed interpretable model achieved AUROC 0.914-0.917, outperforming raw gene SVD.`

## Interpretable drivers

### Cell/module layer

Module annotation from marker overlap:

- `green`: microglia/immune AD-response module, enriched for `APOE`, `C1QA/B/C`, `CD74`, `HLA-DRA`, `AIF1`, `FCER1G`, `TYROBP`.
- `blue`: astrocyte/glial module with neuronal/VLMC hints, including `GFAP`, `AQP4`, `SLC1A3`, `CLU`, `CHI3L1`, `VIM`, `DAB1`, `FGF14`.
- `magenta`: endothelial/pericyte/mural module, including `SLC2A1`, `ADIRF`, `RGS5`.
- `yellow`: astrocyte-stress/iron module, including `FTL`, `FTH1`, `CRYAB`, `GFAP`, `HSPB1`, `SLC1A3`.
- `turquoise`: AD/stress/metabolic module, including `SPP1`, `BEST1`, `CRYAB`, `NDRG1`, `APOD`.

QC36 AD-informed CPT top groups:

- CPT latent distribution: importance 1.417.
- Cell composition: importance 0.416, driven by endothelial ratio distribution.
- Module blue: importance 0.370, with Oligo/OPC/Endo-associated blue-module features.
- Fold-internal selected genes: importance 0.248.
- Curated AD/NVU genes: importance 0.191.
- Node LR signaling: importance 0.189.
- LR signaling: importance 0.148.

### Gene layer

High-frequency fold-internal selected genes after confounder filtering:

- all44: `APOC1`, `AZGP1`, `CD163`, `DDIT4`, `FCER1G`, `GSTP1`, `LY86`, `NDRG1`, `NUPR1`, `RGS1`, `SPP1`, `TYROBP`, `C1QC`, `FTL`, `FTH1`, `GPNMB`.
- qc36: `ADIRF`, `BEST1`, `CAPS`, `FABP5`, `FCER1G`, `FTH1`, `FTL`, `GSTP1`, `HCST`, `LGALS1`, `MT1E`, `MT1X`, `TRIP6`, `TYROBP`, `HLA-DQB1`, `VIM`, `LGALS3`, `SERPING1`.

Curated AD/NVU gene evidence is strongest for:

`VIM`, `SPP1`, `FTL`, `CLU`, `APOE`, `ADIRF`, `HSPB1`, `CRYAB`, `RBFOX2`, `MAPT`, `GFAP`, `APOD`, `SLC1A3`, `FTH1`, `NDRG1`, `HLA-DRB1`, `BEST1`, `GSTP1`, `AIF1`, `C1QC`, `CHI3L1`, `HLA-DRA`, `SLC2A1`, `AQP4`, `TYROBP`, `HCST`.

### LR/region layer

Strong interpretable LR signals include:

- `SPP1-CD44`
- `VIM-CD44`
- `APP-CD74`
- `PSAP-GPR37`
- `NCAM1-NCAM1`
- `NCAM1-NCAM2`
- `APP-SORL1`
- `APP-GPC1`

Region features repeatedly selected include `L1` and `FAS` fractions, supporting a spatial localization panel, but the figure should still report statistics at sample/chip level.

## Suggested figure panel

A. Workflow schematic: NVUs/cells per sample -> CPT latent summaries + curated AD/NVU modules + fold-internal selected genes -> sample-level AD/Control prediction.

B. AUROC bar/dot plot: compare `raw gene SVD`, `CPT latent`, `curated/scalar modules`, `AD-informed static`, `AD-informed CPT`; show both all44 and qc36.

C. Interpretable feature-group stacked bars: CPT latent, module blue/green/turquoise/magenta, cell composition, LR signaling, curated genes, fold-selected genes.

D. Cell/module heatmap: rows as modules or feature groups, columns as neuron, astrocyte, microglia/immune, endothelial, pericyte/mural, VLMC/fibroblast; annotate blue/green/magenta/yellow/turquoise module meanings.

E. Gene dot plot: selected frequency and coefficient direction for `SPP1`, `TYROBP`, `FCER1G`, `C1QC`, `APOE`, `CLU`, `GFAP`, `VIM`, `ADIRF`, `RGS5`, `NDRG1`, `BEST1`, `TRIP6`, `CAPS`, `AZGP1`.

F. LR/region interpretability: LR pairs `SPP1-CD44`, `APP-CD74`, `VIM-CD44`, `PSAP-GPR37`, plus region features such as `L1`/`FAS`.

## Reviewer-risk notes

- Main AUC must be described as leave-one-sample/chip-out, not cell-level random split.
- `CPT latent` is strongest for pure prediction; `AD-informed CPT/static` is the better main biological panel because it adds explainable modules and genes.
- Avoid claiming curated genes alone classify AD well; curated genes are primarily explanatory and improve interpretability.
- Risk features should be discussed only for the QC36 subset where risk predictions exist.
- Technical/confounder genes were filtered from selected gene pools, including sex-linked, mitochondrial/mitochondrial-like, ribosomal, hemoglobin, `MALAT1`, and `NEAT1`.

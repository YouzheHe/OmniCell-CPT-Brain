# Analysis Plan: AD Representation Learning and Interpretability

## Goal

Build a BI-focused figure showing that OmniCell representations provide an
interpretable AD disease-state readout across single-cell/snRNA-seq and spatial
transcriptomics.

## Minimal Successful Analysis

1. Load AD_Hip all-cell embedding metadata and `embedding.npy`.
2. Keep cells or anchors with `condition_inferred` equal to `AD` or `Control`.
3. Train a simple AD/control probe using group-held-out splits by sample/chip.
4. Compare OmniCell embedding performance with raw-expression SVD/HVG PCA.
5. Compute an AD-axis score for each cell.
6. Summarize the score by broad cell type and by sample/chip.
7. Link the AD-axis score to interpretable module genes.
8. Validate spatially at chip level, with anchor maps only as visualization.

## Preferred Statistical Unit

- Single-cell/snRNA: donor or sample if available; otherwise sample/batch.
- Spatial transcriptomics: chip/sample.
- Do not treat individual cells/spots as independent replicates for final
  disease statistics.

## Recommended Panel Evidence

### Panel A: Embedding Context

Show UMAP or low-dimensional projection colored by:

- broad cell type;
- AD versus Control;
- modality.

Use this as context, not as the main quantitative evidence.

### Panel B: Disease Probe

Use grouped cross-validation:

- model: ridge/logistic probe or linear SVM;
- groups: sample/chip;
- labels: AD versus Control;
- output: AUROC, balanced accuracy, macro F1.

Compare:

- raw-expression SVD/HVG PCA;
- native OmniCell if available;
- OmniCell-CPT/fine-tuned representation if available.

### Panel C: Cell-Type-Resolved AD Axis

For each broad cell type, summarize AD-axis scores per sample/chip. The best
biology-facing comparison is usually:

- endothelial/pericyte/VLMC shifts for NVU involvement;
- astrocyte/microglia shifts for AD inflammatory/reactive state;
- excitatory/inhibitory neurons as disease-context support.

### Panel D: Interpretability Modules

Use one or more of the following:

- linear probe coefficients mapped back to genes/modules when feature mapping is
  available;
- module scores computed from raw expression and correlated with AD-axis score;
- leave-module-out or mask-module sensitivity if gene-token perturbation is
  available.

Show the final result as a compact module-by-cell-type heatmap.

### Panel E: Spatial Validation

Project the AD-axis or module scores to AD_Hip spatial chips. If A-beta plaque
coordinates are available, compare:

- Control;
- AD distal/non-plaque;
- AD plaque-neighbor regions.

Keep the statistical test at chip/sample level.

### Panel F: Robustness

Include a compact audit:

- group-held-out split design;
- number of chips/samples per group;
- batch/modality leakage check;
- age/region confounding check when metadata are present.

## Claims That Are Safe

- The representation contains AD-associated information under grouped
  validation.
- The AD axis is enriched in specific NVU and glial modules.
- Spatial AD samples show a consistent shift in the same direction when
  summarized at chip level.

## Claims To Avoid Unless Further Validated

- The embedding proves causality.
- Individual spots/cells are independent statistical replicates.
- Plaque-neighborhood effects are valid without calibrated plaque masks and
  coordinate-scale confirmation.
- A cortex age trend proves AD disease effect without separating age, region,
  and cohort.

## Immediate Next Step

Choose the plotting/implementation backend before writing the final figure
script: Python or R.

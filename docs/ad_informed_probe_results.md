# AD-Informed Probe Results and Figure Decision

Remote root:

`${OMNICELL_NVU_ROOT}/projects/BI`

## Updated Feature Strategy

The BI AD analysis now compares:

- raw-expression SVD;
- CPT embedding;
- curated AD/NVU module scores;
- AD-informed CPT features: low-dimensional CPT representation plus curated
  modules plus fold-internal AD-associated genes.

The fold-internal selected genes are selected only from the training fold before
predicting held-out sample/chip groups. By default, likely confounder genes are
excluded from this selected-gene pool, including sex-linked, mitochondrial,
ribosomal, hemoglobin, MALAT1/NEAT1, and related technical/demographic genes.

## Current Smoke Finding

Simple concatenation of CPT features with selected genes is not yet the best
main-figure claim. In this smoke test, the cleaner and more defensible result is:

- curated AD/NVU modules improve disease readout in several single-cell
  cell-type strata;
- AD-informed features improve over CPT-only in selected spatial cell types,
  especially endothelial, microglia/immune, and inhibitory neurons;
- raw-expression SVD remains the strongest spatial baseline in the tiny smoke
  split, so the final figure should not claim that the current AD-informed CPT
  model universally beats raw SVD yet.

## Single-Cell Smoke AUROC

| scope | raw SVD | CPT | curated modules | AD-informed CPT+modules+genes |
|---|---:|---:|---:|---:|
| all cells | 0.371 | 0.349 | 0.369 | 0.306 |
| astrocyte | 0.397 | 0.382 | 0.528 | 0.381 |
| endothelial | 0.253 | 0.316 | 0.496 | 0.263 |
| excitatory neuron | 0.328 | 0.353 | 0.335 | 0.293 |
| inhibitory neuron | 0.406 | 0.395 | 0.447 | 0.338 |
| microglia/immune | 0.327 | 0.398 | 0.485 | 0.416 |
| pericyte/mural | 0.262 | 0.374 | 0.306 | 0.363 |

Interpretation:

- For single-cell, curated modules are currently the strongest interpretable
  signal in astrocytes, endothelial cells, inhibitory neurons, and
  microglia/immune cells.
- The genome-wide fold-internal selected-gene block is too noisy in the current
  cell-level setup and should not be the main single-cell panel yet.

## Spatial Smoke AUROC

| scope | raw SVD | CPT | curated modules | AD-informed CPT+modules+genes |
|---|---:|---:|---:|---:|
| all cells | 0.966 | 0.001 | 0.721 | 0.149 |
| astrocyte | 0.932 | 0.000 | 0.655 | 0.360 |
| endothelial | 0.879 | 0.525 | 0.712 | 0.817 |
| excitatory neuron | 0.971 | 0.005 | 0.749 | 0.305 |
| inhibitory neuron | 0.952 | 0.598 | 0.786 | 0.853 |
| microglia/immune | 0.919 | 0.643 | 0.717 | 0.836 |
| pericyte/mural | 0.816 | 0.000 | 0.667 | 0.428 |

Interpretation:

- Spatial raw SVD is very strong in smoke, but only two AD and two Control chip
  groups are present, so this may be chip separability.
- AD-informed CPT features are useful in selected spatial strata:
  endothelial, microglia/immune, and inhibitory neurons.
- CPT-only is directionally unstable in this smoke setup; it should be
  calibrated or re-fine-tuned before being used as a headline disease-probe
  model.

## Selected Gene Sanity Check

The current filtered selected-gene table excludes likely confounders. The
following suspicious genes had zero selections after filtering:

`XIST`, `UTY`, `USP9Y`, `NLGN4Y`, `TTTY14`, `MT-ND1`, `MT-CO2`, `RPL41`,
`RPS13`, `MALAT1`.

Biologically interpretable selected genes and modules include:

- microglia/immune: `SPP1`, `FTL`, `CD74`, `APOC1`, `APOE`, `SRGN`, `CD83`;
- astrocyte: `GFAP`, `SPP1`, `VIM`, `CRYAB`, `SPARCL1`, `MAOB`;
- endothelial: `CLU`, `APOE`, `NDRG1`, `SLC38A5`, `MT2A`, `ACSL5`;
- pericyte/mural: `A2M`, `ABCG2`, `RGS5`, `SYNM`, `NTRK3`, `GRM3`;
- excitatory neuron: `ZBTB20`, `COL6A3`, `CSMD1`, `DMD`, `FGF14`, `DAB1`;
- inhibitory neuron: `DLGAP1`, `GRIP1`, `CCSER1`, `NRG3`, `DAB1`, `KCNQ3`.

## Figure Recommendation

For the BI panel, use the following framing:

`AD-informed feature engineering improves interpretable disease readout in
cell-type-specific strata, while raw SVD remains the conservative spatial
baseline until full chip-level validation is completed.`

Best panel layout:

1. Feature design schematic: CPT representation, curated AD/NVU modules,
   fold-internal selected AD genes, and confounder-gene filtering.
2. AUROC comparison by cell type: raw SVD, CPT, curated modules, AD-informed
   features.
3. Module-axis interpretability heatmap: glial, vascular, mural, and neuronal
   modules across cell types.
4. Fold-internal selected gene dot plot: show recurrent genes by cell type,
   with selected frequency and AD-Control direction.
5. Spatial validation: chip-level AD module shifts plus cell-type-specific
   AD-informed performance.

## Next Improvement Before Final Figure

The next model improvement should not be another naive feature concatenation.
Use one of these two safer routes:

1. Late fusion: train CPT, curated module, and selected-gene probes separately
   inside each training fold, then combine their training-fold predictions with
   a small calibrated meta-model.
2. AD-weighted CPT fine-tuning: add curated AD/NVU and neuronal modules as
   auxiliary biological supervision during CPT training, then re-evaluate with
   group-held-out sample/chip splits.

The safest immediate figure claim is cell-type-specific improvement over
CPT-only, with raw SVD retained as a conservative baseline.

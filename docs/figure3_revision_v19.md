# Figure 3 revision v19

This revision keeps Figure 3 focused on vascular-cell biology while adding two
review/validation layers:

1. Single-cell subtype marker audit with top10 markers per cluster.
2. AD_Hip spatial deconvolution with an explicit `Other` class for non-vascular
   reference profiles.

## 1. Marker audit

Run on the remote project root:

```bash
cd ${OMNICELL_NVU_ROOT}
dsub -s projects/nvu_vascular/scripts/dsub_figure3_vascular_marker_audit_v19.sh
```

Useful smoke/full knobs:

```bash
AUDIT_TOP_N=10 MERGE_JACCARD_THRESHOLD=0.35 \
  dsub -s projects/nvu_vascular/scripts/dsub_figure3_vascular_marker_audit_v19.sh
```

Main outputs:

- `figure3_vascular_marker_audit_v19_manual_review_workbook.xlsx`
- `figure3_vascular_marker_audit_v19_cluster_top10_manual_review.csv`
- `figure3_vascular_marker_audit_v19_pairwise_marker_overlap.csv`
- `Figure3_vascular_marker_audit_v19_report.md`

Use the workbook to decide which clusters should be merged or relabeled as
`Other`.

Optional merge map for the spatial rerun:

```csv
cluster,merged_cluster,merged_class
EC00,EC_capillary,Endothelial
EC01,EC_capillary,Endothelial
SMC00,Mural_contractile,SMC
```

Clusters not listed in the map keep their original cluster and class.

## 2. Spatial vascular + Other deconvolution

Run a smoke test first:

```bash
cd ${OMNICELL_NVU_ROOT}
MAX_QUERY_ROWS=500 dsub -s projects/nvu_vascular/scripts/dsub_figure3_adhip_vascular_other_deconv_v19.sh
```

Run the full version without `MAX_QUERY_ROWS`:

```bash
dsub -s projects/nvu_vascular/scripts/dsub_figure3_adhip_vascular_other_deconv_v19.sh
```

Run with a manual merge map after cluster review:

```bash
CLUSTER_MERGE_MAP_CSV=/path/to/figure3_cluster_merge_map.csv \
  dsub -s projects/nvu_vascular/scripts/dsub_figure3_adhip_vascular_other_deconv_v19.sh
```

Main outputs:

- `figure3_adhip_vascular_other_v19_source.csv.gz`
- `figure3_adhip_vascular_other_v19_chip_summary.csv`
- `figure3_adhip_vascular_other_v19_label_agreement.csv`
- `figure3_adhip_vascular_other_v19_marker_deconv_correlations.csv`
- `Figure3_ADHip_vascular_other_deconvolution_v19_report.md`

Interpretation rule: use `deconv_other_probability` and
`pass_vascular_filter` as validation/QC. Do not treat `Other` as a new vascular
cell state in Figure 3.

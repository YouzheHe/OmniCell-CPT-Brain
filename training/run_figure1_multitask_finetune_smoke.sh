#!/usr/bin/env bash
set -euo pipefail

PROJECT=${OMNICELL_NVU_ROOT:-$PWD}/projects/nvu_vascular
PY=${PYTHON:-python}
DATASET=${DATA_ROOT:-${OMNICELL_NVU_ROOT:-$PWD}/NVU_hyz}
MODEL=${OMNICELL_NVU_ROOT}/projects/nvu_vascular/external_vascular_reference/processed/GSE256490_ucsc_adult_control_unsorted/omnicell_cpt_gse256490_adult_control_vascular_supcon_strong/backbone
OUT=${PROJECT}/results/figure1_multitask_cpt_alignment_smoke

cd "$PROJECT"
"$PY" "$PROJECT/scripts/train_memmap_multitask_alignment.py" \
  --dataset-root "$DATASET" \
  --model-name-or-path "$MODEL" \
  --output-dir "$OUT" \
  --sample-ids "AD_Hip_sc,Cortex_sc,AD_Hip_Saptial/AD2.1,AD_Hip_Saptial/AD2.2,AD_Hip_Saptial/Con2.1,AD_Hip_Saptial/Con2.2,Cortex_Spatial/T1001,31435019,39402379_DLPFC" \
  --sequence-length 1500 \
  --token-per-cell 1498 \
  --n-cells-per-sample 1 \
  --selection-strategy top_expression \
  --sample-weight-mode uniform \
  --unsupervised-loss-on nonzero \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 1 \
  --max-steps 8 \
  --logging-steps 1 \
  --save-steps 8 \
  --learning-rate 2e-6 \
  --bf16 \
  --dataloader-num-workers 0

#!/usr/bin/env bash
set -euo pipefail

PROJECT=${OMNICELL_NVU_ROOT:-$PWD}/projects/nvu_vascular
PY=${PYTHON:-python}
DATASET=${DATA_ROOT:-${OMNICELL_NVU_ROOT:-$PWD}/NVU_hyz}
MODEL=${PROJECT}/results/figure1_multitask_cpt_alignment_full/backbone
ANCHOR_DIR=${PROJECT}/results/figure1_agebalanced_training_anchors
ANCHORS=${ANCHOR_DIR}/agebalanced_training_anchors.csv.gz
OUT=${PROJECT}/results/figure1_multitask_cpt_alignment_agebalanced_smoke

cd "$PROJECT"
if [[ ! -s "$ANCHORS" ]]; then
  "$PY" "$PROJECT/scripts/prepare_figure1_agebalanced_training_anchors.py" \
    --dataset-root "$DATASET" \
    --output-dir "$ANCHOR_DIR" \
    --max-per-sample 35000 \
    --max-per-stratum 5000 \
    --n-age-bins 8 \
    --force
fi

"$PY" "$PROJECT/scripts/train_memmap_multitask_alignment.py" \
  --dataset-root "$DATASET" \
  --model-name-or-path "$MODEL" \
  --output-dir "$OUT" \
  --anchor-csv "$ANCHORS" \
  --anchor-strata-column age_condition_modality_cell_class_stratum \
  --anchor-weight-mode stratified \
  --sequence-length 1500 \
  --token-per-cell 1498 \
  --n-cells-per-sample 1 \
  --selection-strategy top_expression \
  --unsupervised-loss-on nonzero \
  --reconstruction-loss-weight 0.75 \
  --disease-loss-weight 0.30 \
  --age-loss-weight 0.85 \
  --cell-class-loss-weight 0.55 \
  --cell-supcon-loss-weight 0.06 \
  --cohort-adversarial-loss-weight 0.08 \
  --modality-adversarial-loss-weight 0.08 \
  --domain-grl-lambda 0.50 \
  --per-device-train-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --max-steps 8 \
  --logging-steps 1 \
  --save-steps 8 \
  --learning-rate 1.2e-6 \
  --bf16 \
  --dataloader-num-workers 0

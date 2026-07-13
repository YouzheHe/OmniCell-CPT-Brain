#!/usr/bin/env python3
"""Prepare Cortex all-cell pseudo-spots for broad/fine deconvolution supervision."""

from __future__ import annotations
import os

import argparse
import json
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

WORK_ROOT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}"))
PROJECT_ROOT = WORK_ROOT / "projects/nvu_vascular"
DEFAULT_INPUT = PROJECT_ROOT / "results/cortex_t906_task_inputs/cortex_sc_subset.h5ad"
DEFAULT_OUT = PROJECT_ROOT / "results/cortex_allcell_pseudospot_deconv_supervision"
DEFAULT_GENE_VOCAB = WORK_ROOT / "NVU_hyz/gene_vocab.txt"
DEFAULT_MODEL = WORK_ROOT / "checkpoint/OmniCell_CPT_336687/checkpoint-245000"

sys.path.insert(0, str(WORK_ROOT / "cellfm-datasets/src"))
from cellfm_dataset.convert import convert_h5ad_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--label-column", default="cell_type")
    parser.add_argument("--label-space", choices=["broad", "fine", "both"], default="both")
    parser.add_argument("--n-pseudospots", type=int, default=50000)
    parser.add_argument("--min-cells-per-spot", type=int, default=2)
    parser.add_argument("--max-cells-per-spot", type=int, default=10)
    parser.add_argument("--min-labels-per-spot", type=int, default=1)
    parser.add_argument("--max-labels-per-spot", type=int, default=4)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.7)
    parser.add_argument("--n-hvg", type=int, default=15000)
    parser.add_argument("--gene-vocab", type=Path, default=DEFAULT_GENE_VOCAB)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[prepare-cortex-pseudospot] {message}", flush=True)


def subset_nonzero_hvg(adata: ad.AnnData, n_hvg: int) -> ad.AnnData:
    if n_hvg <= 0 or adata.n_vars <= n_hvg:
        log(f"using all genes: n_vars={adata.n_vars}")
        return adata
    x = adata.X
    if not sparse.issparse(x):
        x = sparse.csr_matrix(x)
    x = x.tocsr()
    mean = np.asarray(x.mean(axis=0)).ravel()
    mean_sq = np.asarray(x.power(2).mean(axis=0)).ravel()
    var = np.maximum(mean_sq - mean * mean, 0.0)
    nonzero_counts = np.diff(x.tocsc().indptr)
    keep = np.flatnonzero((nonzero_counts > 0) & np.isfinite(var))
    if len(keep) <= n_hvg:
        selected = keep
    else:
        # Prefer variable genes among genes observed in at least one cell.
        order = keep[np.argsort(var[keep])[-n_hvg:]]
        selected = np.sort(order)
    log(f"nonzero-HVG selected {len(selected)} / {adata.n_vars} genes")
    return adata[:, selected].copy()


def safe_label(label: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z]+", "_", str(label)).strip("_")
    return clean or "Unknown"


def broad_label(fine: str) -> str:
    text = str(fine)
    low = text.lower()
    if "astro" in low:
        return "Astrocyte"
    if "oligodendrocyte precursor" in low or low == "opc" or "opc" in low:
        return "OPC"
    if "oligodendrocyte" in low:
        return "Oligodendrocyte"
    if "microglia" in low or "macroph" in low or "immune" in low:
        return "Microglia"
    if "vascular" in low or "endothelial" in low or "pericyte" in low or "smooth muscle" in low:
        return "Vascular cells"
    if "ependymal" in low or "choroid" in low:
        return "Ependymal/choroid"
    inhibitory = ("vip", "pvalb", "sst", "lamp5", "reln")
    if any(token in low for token in inhibitory):
        return "Inhibitory neuron"
    if "neuron" in low or " it " in f" {low} " or low.startswith("l") or any(x in low for x in (" car3", " np", " ct", " et", "l6b")):
        return "Excitatory neuron"
    return "Other"


BROAD_ORDER = [
    "Excitatory neuron",
    "Inhibitory neuron",
    "Astrocyte",
    "Oligodendrocyte",
    "OPC",
    "Microglia",
    "Vascular cells",
    "Ependymal/choroid",
    "Other",
]


def ordered_labels(labels: pd.Series, label_space: str) -> list[str]:
    values = labels.astype(str)
    if label_space == "broad":
        present = set(values)
        return [x for x in BROAD_ORDER if x in present] + sorted(present.difference(BROAD_ORDER))
    counts = values.value_counts()
    return list(counts.index)


def build_pseudospots(
    adata: ad.AnnData,
    *,
    labels: pd.Series,
    label_order: list[str],
    n_pseudospots: int,
    min_cells: int,
    max_cells: int,
    min_labels: int,
    max_labels: int,
    alpha: float,
    seed: int,
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    label_array = labels.astype(str).to_numpy()
    pools = {label: np.flatnonzero(label_array == label) for label in label_order}
    pools = {label: idx for label, idx in pools.items() if len(idx)}
    label_order = [label for label in label_order if label in pools]
    if not label_order:
        raise ValueError("No labels with cells were available.")
    safe_map = {label: safe_label(label) for label in label_order}
    if len(set(safe_map.values())) != len(safe_map):
        raise ValueError(f"Safe label collision: {safe_map}")

    indicator_rows: list[int] = []
    indicator_cols: list[int] = []
    indicator_vals: list[float] = []
    props = np.zeros((n_pseudospots, len(label_order)), dtype=np.float32)
    obs_records = []
    max_labels = min(max_labels, len(label_order), max_cells)
    min_labels = max(1, min(min_labels, max_labels))
    label_sampling_weights = np.ones(len(label_order), dtype=np.float64) / len(label_order)

    for spot_idx in range(n_pseudospots):
        n_cells = int(rng.integers(min_cells, max_cells + 1))
        k = int(rng.integers(min_labels, max_labels + 1))
        k = min(k, n_cells, len(label_order))
        chosen_label_idx = rng.choice(len(label_order), size=k, replace=False, p=label_sampling_weights)
        raw_p = rng.dirichlet(np.full(k, alpha, dtype=np.float32))
        # Give every selected label one cell first; distribute the remainder by
        # Dirichlet weights. This avoids slow rejection sampling when one label
        # has a very small draw.
        counts = np.ones(k, dtype=np.int64)
        remainder = max(0, n_cells - k)
        if remainder:
            counts += rng.multinomial(remainder, raw_p)
        n_chosen = 0
        for local_i, count in enumerate(counts):
            if count <= 0:
                continue
            label_idx = int(chosen_label_idx[local_i])
            pool = pools[label_order[label_idx]]
            selected = rng.choice(pool, size=int(count), replace=count > len(pool))
            indicator_rows.extend([spot_idx] * len(selected))
            indicator_cols.extend(selected.astype(int).tolist())
            indicator_vals.extend([1.0] * len(selected))
            n_chosen += int(len(selected))
            props[spot_idx, label_idx] += float(count)
        props[spot_idx] = props[spot_idx] / max(props[spot_idx].sum(), 1.0)
        dominant_idx = int(props[spot_idx].argmax())
        record = {
            "spot_id": f"pseudo_{spot_idx:06d}",
            "n_cells_mixed": int(n_chosen),
            "dominant_label": label_order[dominant_idx],
            "dominant_label_safe": safe_map[label_order[dominant_idx]],
        }
        for i, label in enumerate(label_order):
            record[f"target_prop_{safe_map[label]}"] = float(props[spot_idx, i])
        obs_records.append(record)

    indicator = sparse.coo_matrix(
        (
            np.asarray(indicator_vals, dtype=np.float32),
            (np.asarray(indicator_rows, dtype=np.int32), np.asarray(indicator_cols, dtype=np.int32)),
        ),
        shape=(n_pseudospots, adata.n_obs),
    ).tocsr()
    x_out = indicator @ adata.X
    pseudo = ad.AnnData(
        X=sparse.csr_matrix(x_out),
        obs=pd.DataFrame(obs_records).set_index("spot_id"),
        var=adata.var.copy(),
    )
    pseudo.obsm["deconv_target_proportions"] = props
    pseudo.uns["deconv_target_labels"] = label_order
    pseudo.uns["deconv_target_safe_labels"] = [safe_map[label] for label in label_order]
    pseudo.uns["deconv_label_mapping"] = safe_map
    return pseudo


def convert_memmap(h5ad_path: Path, output_dir: Path, safe_labels: list[str], gene_vocab: Path, sample_id: str, overwrite: bool) -> Path:
    memmap_dir = output_dir / "memmap"
    obs_columns = ["n_cells_mixed", "dominant_label", "dominant_label_safe"] + [
        f"target_prop_{label}" for label in safe_labels
    ]
    manifest = [
        {
            "h5ad": str(h5ad_path.resolve()),
            "sample_id": sample_id,
            "obs_columns": obs_columns,
            "expression_transform": "normalize_total_log1p",
        }
    ]
    manifest_path = output_dir / f"{sample_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if overwrite or not (memmap_dir / "dataset_manifest.json").exists():
        convert_h5ad_manifest(
            manifest_json=manifest_path,
            output_dir=memmap_dir,
            gene_vocab_path=gene_vocab.resolve(),
            default_obs_columns=None,
            expression_transform="normalize_total_log1p",
            normalize_target_sum=1e4,
            overwrite=overwrite,
        )
    return memmap_dir


def write_train_script(output_dir: Path, memmap_dir: Path, safe_labels: list[str], model_path: Path, label_space: str) -> Path:
    script = output_dir / f"run_cortex_allcell_{label_space}_deconv_supervised.sh"
    target_classes = ",".join(safe_labels)
    max_steps = 1200 if label_space == "broad" else 2200
    deconv_weight = 1.2 if label_space == "broad" else 1.8
    text = f"""#!/usr/bin/env bash
set -euo pipefail

ROOT=${OMNICELL_NVU_ROOT}
PROJECT=$ROOT/projects/nvu_vascular
PY=${PYTHON}

export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-3}}
export LD_LIBRARY_PATH=${SHARED_SOFTWARE_ROOT}/miniconda3/envs/GenOmics2/lib:/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.6/lib64:/usr/local/cuda-12.4/lib64:/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.6/lib64:/usr/local/cuda-11.6/extras/CUPTI/lib64:${{LD_LIBRARY_PATH:-}}
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

$PY $PROJECT/scripts/train_memmap_deconv_regression.py \\
  --dataset-root {memmap_dir} \\
  --model-name-or-path {model_path} \\
  --output-dir {output_dir / ("omnicell_cpt_cortex_allcell_" + label_space + "_deconv_supervised")} \\
  --sequence-length 1500 \\
  --n-cells-per-sample 1 \\
  --selection-strategy top_expression \\
  --sample-weight-mode uniform \\
  --with-replacement \\
  --unsupervised-loss smooth_l1 \\
  --unsupervised-loss-on nonzero \\
  --reconstruction-loss-weight 0.35 \\
  --deconv-loss-weight {deconv_weight:.2f} \\
  --deconv-loss mse \\
  --target-classes {target_classes} \\
  --per-device-train-batch-size 8 \\
  --gradient-accumulation-steps 4 \\
  --learning-rate 4e-6 \\
  --weight-decay 0.01 \\
  --warmup-ratio 0.08 \\
  --max-steps {max_steps} \\
  --logging-steps 20 \\
  --save-steps 400 \\
  --save-total-limit 3 \\
  --dataloader-num-workers 0 \\
  --bf16 \\
  --report-to ""
"""
    script.write_text(text, encoding="utf-8")
    script.chmod(0o755)
    return script


def prepare_one(args: argparse.Namespace, label_space: str) -> dict:
    out = args.output_dir / label_space
    out.mkdir(parents=True, exist_ok=True)
    h5ad_path = out / f"cortex_allcell_{label_space}_pseudospot_deconv_supervision.h5ad"
    if h5ad_path.exists() and not args.overwrite:
        pseudo = ad.read_h5ad(h5ad_path, backed="r")
        safe_labels = list(pseudo.uns["deconv_target_safe_labels"])
        label_order = list(pseudo.uns["deconv_target_labels"])
    else:
        log(f"reading input h5ad: {args.input_h5ad}")
        adata = ad.read_h5ad(args.input_h5ad)
        log(f"loaded matrix shape={adata.shape}")
        adata = subset_nonzero_hvg(adata, args.n_hvg)
        fine = adata.obs[args.label_column].astype(str)
        labels = fine.map(broad_label) if label_space == "broad" else fine
        label_order = ordered_labels(labels, label_space)
        log(f"building {label_space} pseudo-spots: n={args.n_pseudospots}, labels={len(label_order)}")
        pseudo = build_pseudospots(
            adata,
            labels=labels,
            label_order=label_order,
            n_pseudospots=args.n_pseudospots,
            min_cells=args.min_cells_per_spot,
            max_cells=args.max_cells_per_spot,
            min_labels=args.min_labels_per_spot,
            max_labels=args.max_labels_per_spot,
            alpha=args.dirichlet_alpha,
            seed=args.seed + (0 if label_space == "broad" else 1000),
        )
        pseudo.uns["source_h5ad"] = str(args.input_h5ad)
        pseudo.uns["label_column"] = args.label_column
        pseudo.uns["label_space"] = label_space
        pseudo.uns["n_hvg"] = int(args.n_hvg)
        log(f"writing {label_space} h5ad: {h5ad_path}")
        pseudo.write_h5ad(h5ad_path, compression="lzf")
        safe_labels = list(pseudo.uns["deconv_target_safe_labels"])
        label_order = list(pseudo.uns["deconv_target_labels"])

    sample_id = f"cortex_allcell_{label_space}_pseudospot"
    log(f"converting {label_space} h5ad to memmap")
    memmap_dir = convert_memmap(h5ad_path, out, safe_labels, args.gene_vocab, sample_id, args.overwrite)
    train_script = write_train_script(out, memmap_dir, safe_labels, args.model_name_or_path, label_space)
    summary = {
        "label_space": label_space,
        "input_h5ad": str(args.input_h5ad),
        "output_h5ad": str(h5ad_path),
        "memmap_dir": str(memmap_dir),
        "train_script": str(train_script),
        "n_pseudospots": int(ad.read_h5ad(h5ad_path, backed="r").n_obs),
        "target_labels": label_order,
        "target_safe_labels": safe_labels,
        "model_name_or_path": str(args.model_name_or_path),
        "note": "Hierarchical spatial deconvolution supervision entry point; train broad first, then fine.",
    }
    (out / f"cortex_allcell_{label_space}_pseudospot_deconv_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spaces = ["broad", "fine"] if args.label_space == "both" else [args.label_space]
    summaries = [prepare_one(args, space) for space in spaces]
    (args.output_dir / "cortex_allcell_hierarchical_deconv_supervision_manifest.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

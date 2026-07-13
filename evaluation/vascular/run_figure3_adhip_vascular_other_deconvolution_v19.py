#!/usr/bin/env python
"""Figure 3 AD_Hip spatial deconvolution with an explicit Other class.

This is a validation-focused rerun: non-vascular single-cell profiles are
collapsed into the class ``Other`` so spatial anchors that are not convincingly
vascular are no longer forced into EC/pericyte/SMC/VLMC. CPT is not retrained
here; the output is intended to decide whether reference cleanup is sufficient
before considering a CPT adapter.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import nnls
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, normalized_mutual_info_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_figure3_adhip_full_spatial_deconvolution_v14 as v14  # noqa: E402


DEFAULT_RESULT_DIR = v14.DEFAULT_BASE / "figure3_adhip_vascular_other_deconvolution_v19/spatial"
DEFAULT_FIGURE_DIR = (
    v14.PROJECT
    / "figures/vascular_omnicell_cpt_gse256490_adult_control_supcon_backbone_nonzero_hvg_all_data/figure3_adhip_vascular_other_deconvolution_v19/spatial"
)

VASCULAR_CLASSES = list(v14.CLASS_ORDER)
OUTPUT_CLASSES = VASCULAR_CLASSES + ["Other"]
OTHER_COMPONENT_ORDER = [
    "Excitatory neuron",
    "Inhibitory neuron",
    "Astrocyte",
    "Oligodendrocyte",
    "OPC",
    "Microglia/immune",
    "Ependymal/choroid",
    "Other/unknown",
]
VASCULAR_BROAD_LABELS = {"Endothelial", "Pericyte/mural", "VLMC/fibroblast"}
CLASS_PALETTE = {
    **v14.CLASS_PALETTE,
    "Other": "#8A8F98",
    "Low_confidence": "#BFC5CC",
}

LABEL_COLUMNS = [
    "subcelltype",
    "celltype_unit",
    "CellType",
    "celltype",
    "cell_type",
    "Subclass",
    "subclass.v4",
    "CellType_m",
    "Class",
    "Supertype",
]

OTHER_MARKERS = {
    "Other": ["GFAP", "AQP4", "PLP1", "MBP", "P2RY12", "CX3CR1", "RBFOX3", "SNAP25", "HBB", "HBA1", "TTR", "FOXJ1"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-h5ad", type=Path, default=v14.DEFAULT_REF)
    p.add_argument("--annotation-csv", type=Path, default=v14.DEFAULT_ANNOTATION)
    p.add_argument("--memmap-root", type=Path, default=v14.MEMMAP_ROOT)
    p.add_argument("--alias-csv", type=Path, default=v14.ALIAS_CSV)
    p.add_argument("--single-cell-sample-id", default="AD_sc")
    p.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    p.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    p.add_argument("--matrix-key", default="expanded_marker_log1p")
    p.add_argument("--gene-key", default="expanded_marker_genes")
    p.add_argument("--cluster-key", default="v11_clean_cluster")
    p.add_argument("--class-key", default="v11_marker_class")
    p.add_argument("--cluster-merge-map-csv", type=Path, default=None, help="Optional manual map with cluster, merged_cluster, merged_class columns.")
    p.add_argument("--max-other-ref-cells-per-class", type=int, default=6000)
    p.add_argument("--min-other-ref-cells", type=int, default=80)
    p.add_argument("--other-prob-threshold", type=float, default=0.30)
    p.add_argument("--vascular-prob-threshold", type=float, default=0.50)
    p.add_argument("--confidence-threshold", type=float, default=0.30)
    p.add_argument("--residual-threshold", type=float, default=0.95)
    p.add_argument("--max-query-rows", type=int, default=0, help="Optional smoke-test row limit per chip.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=650)
    return p.parse_args()


def clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>", "na", "n/a"} else text


def map_broad_celltype(label: object) -> str:
    text = clean_text(label).lower().replace("_", " ").replace("-", " ")
    if not text:
        return "Other/unknown"
    if re.search(r"(^|\b)ex($|\b| )|glutamatergic|pyramidal|granule|dentate|ca1|ca2|ca3|ca4|subiculum|\bdg\b", text):
        return "Excitatory neuron"
    if re.search(r"(^|\b)in($|\b| )|inhibitory|gaba|interneuron|sst|pvalb|vip|lamp5|sncg", text):
        return "Inhibitory neuron"
    if re.search(r"opc|oligodendrocyte precursor|precursor cell", text):
        return "OPC"
    if re.search(r"oligo|oligodendro", text):
        return "Oligodendrocyte"
    if re.search(r"microglia|\bmicro\b|macrophage|immune|monocyte|blood|t cell|b cell|lymphocyte|myeloid", text):
        return "Microglia/immune"
    if re.search(r"astro|bergmann|radial glia", text):
        return "Astrocyte"
    if re.search(r"endothelial|endothelium|endo|capillary|arter|venous|vascular endothelial", text):
        return "Endothelial"
    if re.search(r"pericyte|smooth muscle|smc|vsmc|mural", text):
        return "Pericyte/mural"
    if re.search(r"vlmc|fibro|leptomeningeal|perivascular fibroblast", text):
        return "VLMC/fibroblast"
    if re.search(r"ependymal|ependy|choroid", text):
        return "Ependymal/choroid"
    if re.search(r"neuron|neuronal", text):
        return "Excitatory neuron"
    return "Other/unknown"


def first_label(obs: pd.DataFrame) -> pd.Series:
    labels = pd.Series([""] * len(obs), index=obs.index, dtype=object)
    for col in LABEL_COLUMNS:
        if col not in obs:
            continue
        vals = obs[col].map(clean_text)
        mask = labels.eq("") & vals.ne("")
        labels.loc[mask] = vals.loc[mask]
    return labels.where(labels.ne(""), "Unknown")


def single_cell_obs_path(memmap_root: Path, requested: str) -> tuple[Path, str]:
    candidates = [requested, "AD_sc", "AD_Hip_sc"]
    for sample_id in candidates:
        obs_path = memmap_root / sample_id / "obs.parquet"
        if obs_path.exists():
            return obs_path, sample_id
    raise FileNotFoundError("Could not locate single-cell obs.parquet under " + str(memmap_root))


def read_single_cell_other_obs(memmap_root: Path, requested_sample_id: str, max_per_class: int, min_cells: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs_path, disk_sample_id = single_cell_obs_path(memmap_root, requested_sample_id)
    obs = pd.read_parquet(obs_path)
    labels = first_label(obs)
    broad = labels.map(map_broad_celltype)
    table = pd.DataFrame(
        {
            "sample_id": requested_sample_id,
            "fallback_sample_id": disk_sample_id,
            "cell_index": obs.index.to_numpy(dtype=np.int64),
            "ground_truth_label": labels.astype(str).to_numpy(),
            "ground_truth_celltype": broad.astype(str).to_numpy(),
        }
    )
    other = table[~table["ground_truth_celltype"].isin(VASCULAR_BROAD_LABELS)].copy()
    rng = np.random.default_rng(seed)
    selected = []
    diag_rows = []
    for label in OTHER_COMPONENT_ORDER:
        sub = other[other["ground_truth_celltype"].eq(label)]
        n_total = int(len(sub))
        if n_total < min_cells:
            diag_rows.append({"ground_truth_celltype": label, "n_total": n_total, "n_selected": 0, "used": False})
            continue
        if len(sub) > max_per_class:
            idx = np.sort(rng.choice(sub.index.to_numpy(), size=max_per_class, replace=False))
            sub = sub.loc[idx]
        selected.append(sub)
        diag_rows.append({"ground_truth_celltype": label, "n_total": n_total, "n_selected": int(len(sub)), "used": True})
    if not selected:
        raise ValueError("No non-vascular cells passed the minimum-cell filter for Other reference profiles.")
    return pd.concat(selected, ignore_index=True), pd.DataFrame(diag_rows)


def fetch_expression(obs: pd.DataFrame, memmap_root: Path, gene_ids: list[int]) -> np.ndarray:
    if v14.MemmapDataset is None:
        raise ImportError("cellfm_dataset.memmap is required. Run this in the remote OmniCell_NVU environment.")
    ds = v14.MemmapDataset(memmap_root)
    out = np.zeros((len(obs), len(gene_ids)), dtype=np.float32)
    gene_to_pos = {int(g): i for i, g in enumerate(gene_ids)}
    for sample_id, part in obs.groupby("sample_id", sort=False):
        sample_key = resolve_sample_key(ds, str(sample_id), part.get("fallback_sample_id", pd.Series([""])).iloc[0])
        sample = ds.samples[sample_key]
        rows = part["cell_index"].to_numpy(dtype=np.int64)
        _, ptr, indices, values = sample.fetch_rows(rows)
        target_rows = part["_row_order"].to_numpy(dtype=np.int64)
        for local_i, global_i in enumerate(target_rows):
            start, end = int(ptr[local_i]), int(ptr[local_i + 1])
            for gid, val in zip(indices[start:end], values[start:end]):
                pos = gene_to_pos.get(int(gid))
                if pos is not None and np.isfinite(val):
                    out[global_i, pos] = max(float(val), 0.0)
    return out


def resolve_sample_key(ds: object, requested: str, fallback: object = "") -> str:
    keys = set(getattr(ds, "samples", {}).keys())
    for candidate in [requested, str(fallback), "AD_sc", "AD_Hip_sc"]:
        if candidate in keys:
            return candidate
    raise KeyError(f"None of the candidate sample IDs are available in MemmapDataset: {requested}, {fallback}")


def shared_query_genes_keep_other(ref_genes: list[str], gene_symbols: list[str]) -> tuple[list[str], list[int], list[int]]:
    symbol_to_id: dict[str, int] = {}
    for idx, symbol in enumerate(gene_symbols):
        symbol_to_id.setdefault(str(symbol).upper(), idx)
    ref_first_idx = {g: i for i, g in enumerate(ref_genes)}
    genes = [g for g in ref_genes if g in symbol_to_id and not nuisance_gene(g)]
    return genes, [ref_first_idx[g] for g in genes], [symbol_to_id[g] for g in genes]


def nuisance_gene(gene: str) -> bool:
    g = str(gene).upper()
    if g.startswith(("MT-", "RPL", "RPS", "MRPL", "MRPS")):
        return True
    if g in {"MALAT1", "NEAT1", "XIST", "FTX"}:
        return True
    return bool(re.match(r"^(AC|AL|AP|RP11|LINC|MIR|SNHG|RN7|RNU|SCARNA|C\d+ORF)", g))


def scale_three(ref_x: np.ndarray, query_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    joined = np.vstack([ref_x, query_x, other_x])
    scale = np.nanpercentile(joined, 99, axis=0).astype(np.float32)
    scale = np.maximum(scale, 1e-3)
    return np.clip(ref_x / scale, 0, 5), np.clip(query_x / scale, 0, 5), np.clip(other_x / scale, 0, 5), scale


def apply_merge_map(ref: object, cluster_key: str, class_key: str, merge_csv: Path | None) -> tuple[str, str]:
    if merge_csv is None:
        return cluster_key, class_key
    if not merge_csv.exists():
        raise FileNotFoundError(str(merge_csv))
    merge = pd.read_csv(merge_csv)
    required = {"cluster", "merged_cluster"}
    if not required.issubset(merge.columns):
        raise ValueError("merge map must contain at least cluster and merged_cluster columns")
    cluster_map = merge.set_index("cluster")["merged_cluster"].astype(str).to_dict()
    ref.obs["figure3_v19_merged_cluster"] = ref.obs[cluster_key].astype(str).map(lambda x: cluster_map.get(x, x)).astype("category")
    if "merged_class" in merge.columns:
        class_map = merge.set_index("cluster")["merged_class"].astype(str).to_dict()
        ref.obs["figure3_v19_merged_class"] = ref.obs[cluster_key].astype(str).map(lambda x: class_map.get(x, ""))
        ref.obs.loc[ref.obs["figure3_v19_merged_class"].eq(""), "figure3_v19_merged_class"] = ref.obs[class_key].astype(str)
    else:
        ref.obs["figure3_v19_merged_class"] = ref.obs[class_key].astype(str)
    ref.obs["figure3_v19_merged_class"] = ref.obs["figure3_v19_merged_class"].astype("category")
    return "figure3_v19_merged_cluster", "figure3_v19_merged_class"


def safe_component(label: str) -> str:
    return "Other__" + re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def build_other_profiles(other_obs: pd.DataFrame, other_scaled: np.ndarray) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    profile_class = {}
    for label in OTHER_COMPONENT_ORDER:
        mask = other_obs["ground_truth_celltype"].astype(str).eq(label).to_numpy()
        if not np.any(mask):
            continue
        name = safe_component(label)
        rows.append(pd.Series(other_scaled[mask].mean(axis=0), name=name))
        profile_class[name] = "Other"
    return pd.DataFrame(rows), pd.Series(profile_class, name="class")


def run_nnls(query_x: np.ndarray, profile: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = profile.to_numpy(dtype=np.float32).T
    weights = np.zeros((query_x.shape[0], profile.shape[0]), dtype=np.float32)
    residual = np.zeros(query_x.shape[0], dtype=np.float32)
    fitted_norm = np.zeros(query_x.shape[0], dtype=np.float32)
    for i in range(query_x.shape[0]):
        y = query_x[i].astype(np.float32)
        w, _ = nnls(a, y)
        pred = a @ w
        denom = max(float(np.linalg.norm(y)), 1e-6)
        residual[i] = float(np.linalg.norm(y - pred) / denom)
        fitted_norm[i] = float(np.linalg.norm(pred) / denom)
        total = float(w.sum())
        if total > 0:
            weights[i] = (w / total).astype(np.float32)
        if (i + 1) % 10000 == 0:
            print(f"NNLS {i + 1}/{query_x.shape[0]}", flush=True)
    return weights, residual, fitted_norm


def entropy_rows(weights: np.ndarray) -> np.ndarray:
    p = np.clip(weights, 1e-8, 1.0)
    return (-(p * np.log(p)).sum(axis=1) / np.log(weights.shape[1])).astype(np.float32)


def add_prediction_columns(query: pd.DataFrame, weights: np.ndarray, profile: pd.DataFrame, profile_class: pd.Series, residual: np.ndarray, fitted_norm: np.ndarray, args: argparse.Namespace) -> pd.DataFrame:
    out = query.drop(columns=["_row_order"], errors="ignore").copy()
    components = profile.index.astype(str).tolist()
    class_map = profile_class.reindex(components).astype(str).to_dict()
    for j, component in enumerate(components):
        out[f"deconv_component_{component}"] = weights[:, j]
    for klass in OUTPUT_CLASSES:
        idx = [j for j, component in enumerate(components) if class_map.get(component) == klass]
        out[f"deconv_class_{klass}"] = weights[:, idx].sum(axis=1) if idx else 0.0

    raw_idx = weights.argmax(axis=1)
    raw_component = np.asarray([components[i] for i in raw_idx], dtype=object)
    raw_class = np.asarray([class_map.get(component, "Other") for component in raw_component], dtype=object)
    vascular_idx = [j for j, component in enumerate(components) if class_map.get(component) in VASCULAR_CLASSES]
    vascular_weights = weights[:, vascular_idx]
    vascular_components = [components[j] for j in vascular_idx]
    vascular_top_idx = vascular_weights.argmax(axis=1)
    top_vascular_component = np.asarray([vascular_components[i] for i in vascular_top_idx], dtype=object)
    top_vascular_class = np.asarray([class_map.get(component, "Other") for component in top_vascular_component], dtype=object)
    top_vascular_prob = vascular_weights.max(axis=1)

    other_prob = out["deconv_class_Other"].to_numpy(float)
    vascular_prob = np.vstack([out[f"deconv_class_{klass}"].to_numpy(float) for klass in VASCULAR_CLASSES]).sum(axis=0)
    pass_filter = (
        (other_prob <= args.other_prob_threshold)
        & (vascular_prob >= args.vascular_prob_threshold)
        & (top_vascular_prob >= args.confidence_threshold)
        & (residual <= args.residual_threshold)
    )
    out["deconv_raw_component"] = raw_component
    out["deconv_raw_class"] = raw_class
    out["deconv_top_vascular_component"] = top_vascular_component
    out["deconv_top_vascular_class"] = top_vascular_class
    out["deconv_top_vascular_probability"] = top_vascular_prob
    out["deconv_other_probability"] = other_prob
    out["deconv_vascular_probability"] = vascular_prob
    out["deconv_entropy"] = entropy_rows(weights)
    out["deconv_residual"] = residual
    out["deconv_fitted_norm"] = fitted_norm
    out["pass_vascular_filter"] = pass_filter
    out["deconv_dominant_class"] = np.where(pass_filter, top_vascular_class, "Other")
    out["deconv_dominant_component"] = np.where(pass_filter, top_vascular_component, "Other")
    return out


def harmonize_truth_five(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    vascular = v14.harmonize_label(raw)
    low = raw.str.lower()
    other_mask = low.str.contains(
        r"astro|neuron|oligo|micro|immune|blood|rbc|epend|choroid|contaminant|mixed|unknown|vascular_unknown|nan|none",
        regex=True,
        na=False,
    )
    out = vascular.copy()
    out[other_mask & ~vascular.isin(VASCULAR_CLASSES)] = "Other"
    out[~out.isin(OUTPUT_CLASSES)] = "Unknown"
    return out


def label_agreement(table: pd.DataFrame, result_dir: Path) -> pd.DataFrame:
    rows = []
    for pred_col in ["deconv_raw_class", "deconv_dominant_class"]:
        pred = table[pred_col].astype(str)
        for truth_col in ["annotation", "vascular_class", "cell_label_original"]:
            if truth_col not in table:
                continue
            truth = harmonize_truth_five(table[truth_col])
            mask = truth.isin(OUTPUT_CLASSES) & pred.isin(OUTPUT_CLASSES)
            if int(mask.sum()) < 10:
                continue
            rows.append(
                {
                    "prediction": pred_col,
                    "truth_column": truth_col,
                    "n": int(mask.sum()),
                    "coverage_fraction": float(mask.mean()),
                    "accuracy": float(accuracy_score(truth[mask], pred[mask])),
                    "balanced_accuracy": float(balanced_accuracy_score(truth[mask], pred[mask])),
                    "macro_f1": float(f1_score(truth[mask], pred[mask], labels=OUTPUT_CLASSES, average="macro", zero_division=0)),
                    "nmi": float(normalized_mutual_info_score(truth[mask], pred[mask])),
                    "other_pred_fraction": float(pred.eq("Other").mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / "figure3_adhip_vascular_other_v19_label_agreement.csv", index=False)
    return out


def chip_summary(table: pd.DataFrame, components: list[str], result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for sample, sub in table.groupby("sample_id", observed=False):
        row = {
            "sample_id": sample,
            "disease_group": sub["disease_group"].astype(str).mode().iloc[0],
            "n_anchors": int(len(sub)),
            "pass_vascular_filter_fraction": float(sub["pass_vascular_filter"].mean()),
            "dominant_other_fraction": float(sub["deconv_dominant_class"].astype(str).eq("Other").mean()),
            "mean_other_probability": float(sub["deconv_other_probability"].mean()),
            "mean_vascular_probability": float(sub["deconv_vascular_probability"].mean()),
            "mean_residual": float(sub["deconv_residual"].mean()),
            "mean_entropy": float(sub["deconv_entropy"].mean()),
        }
        for klass in OUTPUT_CLASSES:
            row[f"mean_prop_{klass}"] = float(sub[f"deconv_class_{klass}"].mean())
            row[f"dominant_frac_{klass}"] = float(sub["deconv_dominant_class"].astype(str).eq(klass).mean())
        for component in components:
            col = f"deconv_component_{component}"
            row[f"mean_component_{component}"] = float(sub[col].mean())
        rows.append(row)
    comp = pd.DataFrame(rows).sort_values(["disease_group", "sample_id"])
    comp.to_csv(result_dir / "figure3_adhip_vascular_other_v19_chip_summary.csv", index=False)
    long_rows = []
    for _, row in comp.iterrows():
        for klass in OUTPUT_CLASSES:
            long_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "disease_group": row["disease_group"],
                    "level": "class",
                    "label": klass,
                    "mean_proportion": row[f"mean_prop_{klass}"],
                    "dominant_fraction": row[f"dominant_frac_{klass}"],
                    "n_anchors": row["n_anchors"],
                }
            )
    comp_long = pd.DataFrame(long_rows)
    comp_long.to_csv(result_dir / "figure3_adhip_vascular_other_v19_chip_summary_long.csv", index=False)
    return comp, comp_long


def cohen_d(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def ad_stats(comp: pd.DataFrame, result_dir: Path) -> pd.DataFrame:
    features = [
        "pass_vascular_filter_fraction",
        "dominant_other_fraction",
        "mean_other_probability",
        "mean_vascular_probability",
        "mean_residual",
        "mean_entropy",
    ]
    features += [f"mean_prop_{klass}" for klass in OUTPUT_CLASSES]
    features += [f"dominant_frac_{klass}" for klass in OUTPUT_CLASSES]
    rows = []
    for feature in features:
        ad = pd.to_numeric(comp.loc[comp["disease_group"].eq("AD"), feature], errors="coerce").dropna()
        control = pd.to_numeric(comp.loc[comp["disease_group"].eq("Control"), feature], errors="coerce").dropna()
        if len(ad) < 2 or len(control) < 2:
            continue
        welch_p = np.nan
        if v14.scipy_stats is not None:
            welch_p = float(v14.scipy_stats.ttest_ind(ad, control, equal_var=False, nan_policy="omit").pvalue)
        rows.append(
            {
                "feature": feature,
                "n_ad_chips": int(len(ad)),
                "n_control_chips": int(len(control)),
                "mean_ad": float(ad.mean()),
                "mean_control": float(control.mean()),
                "delta_ad_minus_control": float(ad.mean() - control.mean()),
                "cohens_d_ad_minus_control": cohen_d(ad, control),
                "welch_t_p": welch_p,
            }
        )
    out = pd.DataFrame(rows)
    out["abs_delta"] = out["delta_ad_minus_control"].abs()
    out = out.sort_values("abs_delta", ascending=False).drop(columns="abs_delta")
    out.to_csv(result_dir / "figure3_adhip_vascular_other_v19_ad_vs_control_stats.csv", index=False)
    return out


def marker_deconv_correlations(log_expr: np.ndarray, genes: list[str], table: pd.DataFrame, result_dir: Path) -> pd.DataFrame:
    module_map = {
        "Endothelial": ["CLDN5", "PECAM1", "VWF", "CDH5", "FLT1", "KDR", "RAMP2", "SLC2A1"],
        "Pericyte": ["PDGFRB", "RGS5", "KCNJ8", "ABCC9", "NOTCH3", "CSPG4", "MCAM"],
        "SMC": ["ACTA2", "MYH11", "TAGLN", "CNN1", "MYLK", "MYOCD", "SMTN"],
        "Fibroblast_VLMC": ["COL1A1", "COL1A2", "COL3A1", "COL6A1", "COL6A2", "DCN", "LUM", "APOD"],
        **OTHER_MARKERS,
    }
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    rows = []
    for klass, module_genes in module_map.items():
        idx = [gene_to_idx[g] for g in module_genes if g in gene_to_idx]
        if not idx or f"deconv_class_{klass}" not in table:
            continue
        score = log_expr[:, idx].mean(axis=1)
        prop = table[f"deconv_class_{klass}"].to_numpy(float)
        rows.append(
            {
                "class": klass,
                "n_genes": len(idx),
                "genes": ",".join([g for g in module_genes if g in gene_to_idx]),
                "pearson_r": np.nan if np.std(score) == 0 or np.std(prop) == 0 else float(np.corrcoef(score, prop)[0, 1]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / "figure3_adhip_vascular_other_v19_marker_deconv_correlations.csv", index=False)
    return out


def selected_samples(comp: pd.DataFrame) -> list[str]:
    out = []
    for disease in ["Control", "AD"]:
        out.extend(comp[comp["disease_group"].eq(disease)].sort_values("n_anchors", ascending=False)["sample_id"].head(4).tolist())
    return out


def plot_spatial_grid(table: pd.DataFrame, samples: list[str], color_key: str, prefix: Path, dpi: int) -> None:
    n = len(samples)
    if n == 0:
        return
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    cats = [c for c in OUTPUT_CLASSES if c in set(table[color_key].astype(str))]
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.25 * ncols, 2.0 * nrows), squeeze=False)
    for ax, sample in zip(axes.ravel(), samples):
        sub = table[table["sample_id"].astype(str).eq(sample)]
        vals = sub[color_key].astype(str).to_numpy()
        x = pd.to_numeric(sub["coord_x"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(sub["coord_y"], errors="coerce").to_numpy(float)
        for cat in cats:
            idx = vals == cat
            if np.any(idx):
                ax.scatter(x[idx], y[idx], s=3.6, lw=0, c=[CLASS_PALETTE.get(cat, "#808080")], alpha=0.82, rasterized=True)
        ax.set_title(sample.split("/")[-1], fontsize=6.8, pad=1.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["left", "bottom", "right", "top"]].set_visible(False)
    for ax in axes.ravel()[len(samples) :]:
        ax.axis("off")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=CLASS_PALETTE.get(c, "#808080"), markeredgewidth=0, markersize=4, label=c) for c in cats]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=5.8)
    fig.subplots_adjust(left=0.02, right=0.86, bottom=0.02, top=0.92, wspace=0.06, hspace=0.13)
    v14.save_all(fig, prefix, dpi)


def plot_metrics(metrics: pd.DataFrame, prefix: Path, dpi: int) -> None:
    if metrics.empty:
        return
    sub = metrics[metrics["prediction"].eq("deconv_dominant_class")].copy()
    if sub.empty:
        return
    cols = ["accuracy", "balanced_accuracy", "macro_f1", "nmi"]
    fig, ax = plt.subplots(figsize=(4.7, 2.7))
    x = np.arange(len(cols))
    width = 0.22
    for i, (_, row) in enumerate(sub.iterrows()):
        vals = [row[c] for c in cols]
        ax.bar(x + (i - (len(sub) - 1) / 2) * width, vals, width=width, label=row["truth_column"], edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Balanced\nacc.", "Macro F1", "NMI"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Figure 3 vascular + Other validation", loc="left", fontsize=8.3, fontweight="bold")
    ax.legend(fontsize=5.8)
    ax.grid(axis="y", lw=0.35, color="#D8DEE8")
    v14.save_all(fig, prefix, dpi)


def write_report(result_dir: Path, summary: dict[str, object], metrics: pd.DataFrame, stats: pd.DataFrame) -> None:
    lines = [
        "# Figure 3 AD_Hip vascular + Other deconvolution v19",
        "",
        "Non-vascular reference profiles are modeled as Other. CPT is not retrained in this run.",
        "",
        "## Summary",
        "",
        f"- Query anchors: {summary['n_query_anchors']}",
        f"- Chips: {summary['n_chips']}",
        f"- Vascular profiles: {summary['n_vascular_profiles']}",
        f"- Other profiles: {summary['n_other_profiles']}",
        f"- Shared genes: {summary['n_genes']}",
        f"- Mean Other probability: {summary['mean_other_probability']:.4f}",
        f"- Mean vascular probability: {summary['mean_vascular_probability']:.4f}",
        f"- Pass vascular filter fraction: {summary['pass_vascular_filter_fraction']:.4f}",
        "",
        "## Label Agreement",
        "",
    ]
    if metrics.empty:
        lines.append("No usable label-agreement records.")
    else:
        cols = ["prediction", "truth_column", "n", "accuracy", "balanced_accuracy", "macro_f1", "other_pred_fraction"]
        lines.append("|" + "|".join(cols) + "|")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, row in metrics[cols].iterrows():
            lines.append("|" + "|".join(f"{row[c]:.4g}" if isinstance(row[c], float) else str(row[c]) for c in cols) + "|")
    lines.extend(["", "## Top AD-Control QC Changes", ""])
    if not stats.empty:
        cols = ["feature", "mean_ad", "mean_control", "delta_ad_minus_control", "cohens_d_ad_minus_control", "welch_t_p"]
        lines.append("|" + "|".join(cols) + "|")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, row in stats[cols].head(16).iterrows():
            lines.append("|" + "|".join(f"{row[c]:.4g}" if isinstance(row[c], float) else str(row[c]) for c in cols) + "|")
    (result_dir / "Figure3_ADHip_vascular_other_deconvolution_v19_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    figure_dir = Path(args.figure_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    required = [args.reference_h5ad, args.annotation_csv, args.memmap_root / "gene_vocab.txt"]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    if v14.sc is None:
        raise ImportError("scanpy is required. Run this in the remote BrainST environment.")

    ref = v14.sc.read_h5ad(args.reference_h5ad)
    ref_x, ref_genes = v14.get_marker_matrix(ref, args.matrix_key, args.gene_key)
    n_memmap_genes = len((args.memmap_root / "gene_vocab.txt").read_text().splitlines())
    gene_symbols = v14.load_gene_symbols(args.alias_csv, n_memmap_genes, args.memmap_root)
    genes, ref_idx, query_gene_ids = shared_query_genes_keep_other(ref_genes, gene_symbols)
    if len(genes) < 20:
        raise ValueError(f"Too few shared genes for vascular+Other deconvolution: {len(genes)}")

    query = v14.load_adhip_query(args.annotation_csv, args.max_query_rows)
    other_obs, other_diag = read_single_cell_other_obs(
        args.memmap_root,
        args.single_cell_sample_id,
        args.max_other_ref_cells_per_class,
        args.min_other_ref_cells,
        args.seed,
    )
    query["_row_order"] = np.arange(len(query), dtype=np.int64)
    other_obs["_row_order"] = np.arange(len(other_obs), dtype=np.int64)
    print(f"[INFO] query anchors={len(query):,}; chips={query['sample_id'].nunique()}", flush=True)
    print(f"[INFO] other reference cells={len(other_obs):,}", flush=True)

    query_raw = fetch_expression(query, args.memmap_root, query_gene_ids)
    other_raw = fetch_expression(other_obs, args.memmap_root, query_gene_ids)
    query_log = np.log1p(np.nan_to_num(np.clip(query_raw, 0, None), nan=0.0)).astype(np.float32)
    other_log = np.log1p(np.nan_to_num(np.clip(other_raw, 0, None), nan=0.0)).astype(np.float32)
    ref_x = ref_x[:, ref_idx]
    ref_scaled, query_scaled, other_scaled, scale = scale_three(ref_x, query_log, other_log)

    profile_cluster_key, profile_class_key = apply_merge_map(ref, args.cluster_key, args.class_key, args.cluster_merge_map_csv)
    vascular_profile, vascular_class = v14.build_profiles(ref, ref_scaled, profile_cluster_key, profile_class_key)
    vascular_profile.columns = genes
    other_profile, other_class = build_other_profiles(other_obs, other_scaled)
    other_profile.columns = genes
    profile = pd.concat([vascular_profile, other_profile], axis=0)
    profile_class = pd.concat([vascular_class.astype(str), other_class.astype(str)])

    profile.to_csv(result_dir / "figure3_adhip_vascular_other_v19_reference_profiles.csv")
    profile_class.to_csv(result_dir / "figure3_adhip_vascular_other_v19_reference_profile_classes.csv", header=True)
    other_diag.to_csv(result_dir / "figure3_adhip_vascular_other_v19_other_reference_diagnostics.csv", index=False)
    pd.DataFrame({"gene": genes, "scale_p99": scale, "query_gene_id": query_gene_ids}).to_csv(result_dir / "figure3_adhip_vascular_other_v19_gene_scaling.csv", index=False)

    weights, residual, fitted_norm = run_nnls(query_scaled, profile)
    source = add_prediction_columns(query, weights, profile, profile_class, residual, fitted_norm, args)
    source.to_csv(result_dir / "figure3_adhip_vascular_other_v19_source.csv.gz", index=False, compression="gzip")

    metrics = label_agreement(source, result_dir)
    comp, _ = chip_summary(source, profile.index.astype(str).tolist(), result_dir)
    stats = ad_stats(comp, result_dir)
    marker_corr = marker_deconv_correlations(query_log, genes, source, result_dir)

    samples = selected_samples(comp)
    plot_spatial_grid(source, samples, "deconv_raw_class", figure_dir / "figure3_adhip_vascular_other_v19_selected_chips_raw_class", args.dpi)
    plot_spatial_grid(source, samples, "deconv_dominant_class", figure_dir / "figure3_adhip_vascular_other_v19_selected_chips_filtered_class", args.dpi)
    plot_metrics(metrics, figure_dir / "figure3_adhip_vascular_other_v19_validation_metrics", args.dpi)

    summary = {
        "method": "NNLS marker-profile deconvolution with explicit Other background; no CPT retraining",
        "reference_h5ad": str(args.reference_h5ad),
        "annotation_csv": str(args.annotation_csv),
        "single_cell_sample_id": args.single_cell_sample_id,
        "cluster_merge_map_csv": str(args.cluster_merge_map_csv) if args.cluster_merge_map_csv else "",
        "n_reference_cells": int(ref.n_obs),
        "n_query_anchors": int(len(source)),
        "n_chips": int(source["sample_id"].nunique()),
        "n_genes": int(len(genes)),
        "n_vascular_profiles": int(len(vascular_profile)),
        "n_other_profiles": int(len(other_profile)),
        "other_prob_threshold": float(args.other_prob_threshold),
        "vascular_prob_threshold": float(args.vascular_prob_threshold),
        "confidence_threshold": float(args.confidence_threshold),
        "residual_threshold": float(args.residual_threshold),
        "mean_other_probability": float(source["deconv_other_probability"].mean()),
        "mean_vascular_probability": float(source["deconv_vascular_probability"].mean()),
        "pass_vascular_filter_fraction": float(source["pass_vascular_filter"].mean()),
        "mean_residual": float(source["deconv_residual"].mean()),
        "metrics": metrics.to_dict(orient="records"),
        "marker_deconv_correlations": marker_corr.to_dict(orient="records"),
        "result_dir": str(result_dir),
        "figure_dir": str(figure_dir),
    }
    (result_dir / "figure3_adhip_vascular_other_v19_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(result_dir, summary, metrics, stats)
    print(json.dumps(summary, indent=2), flush=True)
    if not metrics.empty:
        print("label_agreement", flush=True)
        print(metrics.to_string(index=False), flush=True)
    if not stats.empty:
        print("top_ad_vs_control_qc_stats", flush=True)
        print(stats.head(16).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

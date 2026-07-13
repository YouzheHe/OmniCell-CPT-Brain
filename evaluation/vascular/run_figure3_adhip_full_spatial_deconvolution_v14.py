#!/usr/bin/env python
"""Full AD_Hip spatial vascular deconvolution and chip-level statistics.

This reruns Figure 3 AD hippocampus spatial deconvolution from the full
vascular anchor table, not from the sampled v12b spatial h5ad.  Disease labels
for AD_Hip are inferred from sample_id (/AD or /Con), because the synchronized
metadata stores AD hippocampus cases as condition_inferred == Unknown.
"""

from __future__ import annotations
import os

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import nnls

try:
    import scanpy as sc
except Exception:  # pragma: no cover - allows --help on lightweight local hosts.
    sc = None

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - remote environment should have scipy.
    scipy_stats = None


WORK_ROOT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}"))
PROJECT = WORK_ROOT / "projects/nvu_vascular"
MEMMAP_ROOT = WORK_ROOT / "NVU_hyz"
ALIAS_CSV = WORK_ROOT / "OmniCell-HF/assets/vocab/new_genes_homo_sapiens.csv"
DEFAULT_BASE = PROJECT / "results/vascular_omnicell_cpt_gse256490_adult_control_supcon_backbone_nonzero_hvg_all_data"
DEFAULT_REF = DEFAULT_BASE / "vascular_clean_diagonal_v11/single_cell/single_cell_clean_vascular_v11.h5ad"
DEFAULT_ANNOTATION = PROJECT / "results/vascular_recluster_by_modality/vascular_cells_clustered_annotated.csv.gz"
DEFAULT_RESULT_DIR = DEFAULT_BASE / "figure3_adhip_full_spatial_deconvolution_v14/spatial"
DEFAULT_FIGURE_DIR = PROJECT / "figures/vascular_omnicell_cpt_gse256490_adult_control_supcon_backbone_nonzero_hvg_all_data/figure3_adhip_full_spatial_deconvolution_v14/spatial"

sys.path.insert(0, str(WORK_ROOT / "cellfm-datasets/src"))

try:
    from cellfm_dataset.memmap import MemmapDataset  # noqa: E402
except Exception:  # pragma: no cover - allows --help on local hosts.
    MemmapDataset = None


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
    }
)

CLASS_ORDER = ["Endothelial", "Pericyte", "SMC", "Fibroblast_VLMC"]
CLASS_PALETTE = {
    "Endothelial": "#4E79A7",
    "Pericyte": "#B07AA1",
    "SMC": "#E15759",
    "Fibroblast_VLMC": "#59A14F",
    "Low_confidence": "#BFC5CC",
}
DISEASE_PALETTE = {"Control": "#7E8FA6", "AD": "#C86054"}
DISTINCT_CLUSTER_COLORS = [
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#76B7B2",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
]
EXCLUDE_GENES = {
    "GFAP",
    "AQP4",
    "PLP1",
    "MBP",
    "P2RY12",
    "CX3CR1",
    "RBFOX3",
    "SNAP25",
    "HBB",
    "HBA1",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--reference-h5ad", type=Path, default=DEFAULT_REF)
    p.add_argument("--annotation-csv", type=Path, default=DEFAULT_ANNOTATION)
    p.add_argument("--memmap-root", type=Path, default=MEMMAP_ROOT)
    p.add_argument("--alias-csv", type=Path, default=ALIAS_CSV)
    p.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    p.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    p.add_argument("--matrix-key", default="expanded_marker_log1p")
    p.add_argument("--gene-key", default="expanded_marker_genes")
    p.add_argument("--cluster-key", default="v11_clean_cluster")
    p.add_argument("--class-key", default="v11_marker_class")
    p.add_argument("--confidence-threshold", type=float, default=0.38)
    p.add_argument("--residual-threshold", type=float, default=0.78)
    p.add_argument("--max-query-rows", type=int, default=0, help="Optional smoke-test row limit.")
    p.add_argument("--dpi", type=int, default=650)
    return p.parse_args()


def save_all(fig: plt.Figure, prefix: Path, dpi: int) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(prefix.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def infer_adhip_disease(sample_id: object) -> str:
    sid = str(sample_id).lower()
    if "/ad" in sid:
        return "AD"
    if "/con" in sid:
        return "Control"
    return "Unknown"


def load_gene_symbols(alias_csv: Path, n_genes: int, memmap_root: Path) -> list[str]:
    gene_ids = [x.strip().split(".")[0].upper() for x in (memmap_root / "gene_vocab.txt").read_text().splitlines()[:n_genes]]
    if alias_csv.exists():
        df = pd.read_csv(alias_csv, header=None)
        alias = (
            df[[0, 1]]
            .dropna()
            .assign(
                ensembl=lambda x: x[0].astype(str).str.split(".").str[0].str.upper(),
                symbol=lambda x: x[1].astype(str).str.upper(),
            )
            .drop_duplicates("ensembl")
            .set_index("ensembl")["symbol"]
            .to_dict()
        )
        return [alias.get(gid, gid) for gid in gene_ids]
    return gene_ids


def fetch_expression(obs: pd.DataFrame, memmap_root: Path, gene_ids: list[int]) -> np.ndarray:
    if MemmapDataset is None:
        raise ImportError("cellfm_dataset.memmap is required to fetch full spatial expression.")
    ds = MemmapDataset(memmap_root)
    out = np.zeros((len(obs), len(gene_ids)), dtype=np.float32)
    gene_to_pos = {int(g): i for i, g in enumerate(gene_ids)}
    for sample_id, part in obs.groupby("sample_id", sort=False):
        sample = ds.samples[str(sample_id)]
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


def read_annotation_table(annotation_path: Path, columns: list[str]) -> pd.DataFrame:
    if annotation_path.suffix == ".parquet":
        return pd.read_parquet(annotation_path, columns=columns)
    return pd.read_csv(annotation_path, usecols=lambda c: c in columns, low_memory=False)


def load_adhip_query(annotation_csv: Path, max_query_rows: int = 0) -> pd.DataFrame:
    wanted = [
        "sample_id",
        "cell_index",
        "cohort",
        "modality",
        "coord_x",
        "coord_y",
        "cell_label_original",
        "vascular_class",
        "condition_inferred",
        "annotation",
        "annotation_confidence",
    ]
    obs = read_annotation_table(annotation_csv, wanted)
    sub = obs[
        obs["cohort"].astype(str).eq("AD_Hip_Saptial")
        & obs["modality"].astype(str).eq("spatial")
    ].copy()
    sub["disease_group"] = sub["sample_id"].map(infer_adhip_disease)
    sub = sub[sub["disease_group"].isin(["AD", "Control"])].copy()
    sub["cell_index"] = pd.to_numeric(sub["cell_index"], errors="raise").astype(np.int64)
    if max_query_rows and max_query_rows > 0:
        sub = sub.groupby("sample_id", group_keys=False, observed=False).head(max_query_rows)
    sub = sub.reset_index(drop=True)
    sub["_row_order"] = np.arange(len(sub), dtype=np.int64)
    if sub.empty:
        raise ValueError("No full AD_Hip spatial anchors were found in annotation table.")
    return sub


def get_marker_matrix(adata: sc.AnnData, matrix_key: str, gene_key: str) -> tuple[np.ndarray, list[str]]:
    if matrix_key not in adata.obsm:
        raise KeyError(f"{matrix_key} missing from reference obsm")
    if gene_key not in adata.uns:
        raise KeyError(f"{gene_key} missing from reference uns")
    return np.asarray(adata.obsm[matrix_key], dtype=np.float32), [str(x).upper() for x in adata.uns[gene_key]]


def shared_query_genes(ref_genes: list[str], gene_symbols: list[str]) -> tuple[list[str], list[int], list[int]]:
    symbol_to_id: dict[str, int] = {}
    for idx, symbol in enumerate(gene_symbols):
        symbol_to_id.setdefault(str(symbol).upper(), idx)
    ref_first_idx = {g: i for i, g in enumerate(ref_genes)}
    genes = [g for g in ref_genes if g in symbol_to_id and g not in EXCLUDE_GENES]
    return genes, [ref_first_idx[g] for g in genes], [symbol_to_id[g] for g in genes]


def scale_matrices(ref_x: np.ndarray, query_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample = np.vstack([ref_x, query_x])
    scale = np.nanpercentile(sample, 99, axis=0).astype(np.float32)
    scale = np.maximum(scale, 1e-3)
    return np.clip(ref_x / scale, 0, 5), np.clip(query_x / scale, 0, 5), scale


def build_profiles(ref: sc.AnnData, ref_x: np.ndarray, cluster_key: str, class_key: str) -> tuple[pd.DataFrame, pd.Series]:
    clusters = list(ref.obs[cluster_key].cat.categories) if hasattr(ref.obs[cluster_key], "cat") else sorted(ref.obs[cluster_key].astype(str).unique())
    labels = ref.obs[cluster_key].astype(str).to_numpy()
    rows = []
    cluster_class = {}
    for cluster in clusters:
        mask = labels == str(cluster)
        if not np.any(mask):
            continue
        rows.append(pd.Series(ref_x[mask].mean(axis=0), name=str(cluster)))
        klass = ref.obs.loc[mask, class_key].astype(str).mode()
        cluster_class[str(cluster)] = klass.iloc[0] if len(klass) else "Unknown"
    profile = pd.DataFrame(rows)
    return profile, pd.Series(cluster_class, name="class")


def run_nnls(query_x: np.ndarray, profile: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = profile.to_numpy(dtype=np.float32).T
    n_query = query_x.shape[0]
    n_clusters = profile.shape[0]
    weights = np.zeros((n_query, n_clusters), dtype=np.float32)
    residual = np.zeros(n_query, dtype=np.float32)
    fitted_norm = np.zeros(n_query, dtype=np.float32)
    for i in range(n_query):
        y = query_x[i].astype(np.float32)
        w, _ = nnls(a, y)
        pred = a @ w
        denom = max(float(np.linalg.norm(y)), 1e-6)
        residual[i] = float(np.linalg.norm(y - pred) / denom)
        fitted_norm[i] = float(np.linalg.norm(pred) / denom)
        s = float(w.sum())
        if s > 0:
            weights[i] = (w / s).astype(np.float32)
        if (i + 1) % 10000 == 0:
            print(f"NNLS {i + 1}/{n_query}", flush=True)
    return weights, residual, fitted_norm


def entropy_rows(weights: np.ndarray) -> np.ndarray:
    p = np.clip(weights, 1e-8, 1.0)
    ent = -(p * np.log(p)).sum(axis=1)
    return ent / np.log(weights.shape[1])


def add_deconv_columns(
    query: pd.DataFrame,
    weights: np.ndarray,
    profile: pd.DataFrame,
    cluster_class: pd.Series,
    residual: np.ndarray,
    fitted_norm: np.ndarray,
    confidence_threshold: float,
    residual_threshold: float,
) -> pd.DataFrame:
    out = query.drop(columns=["_row_order"]).copy()
    clusters = profile.index.astype(str).tolist()
    class_by_cluster = cluster_class.reindex(clusters).astype(str).to_dict()
    for j, cluster in enumerate(clusters):
        out[f"deconv_cluster_{cluster}"] = weights[:, j]
    for klass in CLASS_ORDER:
        cols = [j for j, cluster in enumerate(clusters) if class_by_cluster.get(cluster) == klass]
        out[f"deconv_class_{klass}"] = weights[:, cols].sum(axis=1) if cols else 0.0
    max_idx = weights.argmax(axis=1)
    max_w = weights[np.arange(weights.shape[0]), max_idx]
    dom_cluster = np.array([clusters[i] for i in max_idx], dtype=object)
    dom_class = np.array([class_by_cluster.get(c, "Unknown") for c in dom_cluster], dtype=object)
    confident = (max_w >= confidence_threshold) & (residual <= residual_threshold)
    out["deconv_dominant_cluster_raw"] = dom_cluster
    out["deconv_dominant_class_raw"] = dom_class
    out["deconv_dominant_cluster"] = np.where(confident, dom_cluster, "Low_confidence")
    out["deconv_dominant_class"] = np.where(confident, dom_class, "Low_confidence")
    out["deconv_confidence"] = max_w
    out["deconv_entropy"] = entropy_rows(weights)
    out["deconv_residual"] = residual
    out["deconv_fitted_norm"] = fitted_norm
    out["deconv_is_confident"] = confident
    return out


def harmonize_label(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    low = s.str.lower()
    out = pd.Series("Unknown", index=s.index, dtype=object)
    out[low.str.contains("endo|endothelial|capillary|arterial|venous", regex=True, na=False)] = "Endothelial"
    out[low.str.contains("peri|pericyte", regex=True, na=False)] = "Pericyte"
    out[low.str.contains("smc|vsmc|smooth|mural|contractile", regex=True, na=False)] = "SMC"
    out[low.str.contains("fibro|vlmc|leptomeningeal", regex=True, na=False)] = "Fibroblast_VLMC"
    out[s.isin(CLASS_ORDER)] = s[s.isin(CLASS_ORDER)]
    out[s.isin(["Possible_contaminant", "Mixed_spot", "nan", "None", "Unknown", "vascular_unknown"])] = "Unknown"
    return out


def label_agreement(table: pd.DataFrame, result_dir: Path) -> pd.DataFrame:
    pred = table["deconv_dominant_class"].astype(str)
    rows = []
    for col in ["annotation", "vascular_class", "cell_label_original"]:
        if col not in table:
            continue
        truth = harmonize_label(table[col])
        mask = truth.isin(CLASS_ORDER) & pred.isin(CLASS_ORDER)
        if int(mask.sum()) < 10:
            continue
        rows.append(
            {
                "truth_column": col,
                "n": int(mask.sum()),
                "accuracy": float((truth[mask].to_numpy() == pred[mask].to_numpy()).mean()),
                "pred_confident_fraction": float(pred.isin(CLASS_ORDER).mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / "figure3_adhip_full_spatial_deconv_label_agreement.csv", index=False)
    return out


def chip_composition(table: pd.DataFrame, clusters: list[str], result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for sample, sub in table.groupby("sample_id", observed=False):
        row = {
            "sample_id": sample,
            "disease_group": sub["disease_group"].astype(str).mode().iloc[0],
            "condition_inferred": sub["condition_inferred"].astype(str).mode().iloc[0] if "condition_inferred" in sub else "",
            "n_anchors": int(len(sub)),
            "mean_confidence": float(sub["deconv_confidence"].mean()),
            "median_confidence": float(sub["deconv_confidence"].median()),
            "mean_entropy": float(sub["deconv_entropy"].mean()),
            "mean_residual": float(sub["deconv_residual"].mean()),
            "confident_fraction": float(sub["deconv_is_confident"].astype(bool).mean()),
        }
        for klass in CLASS_ORDER + ["Low_confidence"]:
            row[f"dominant_class_frac_{klass}"] = float((sub["deconv_dominant_class"].astype(str) == klass).mean())
        for klass in CLASS_ORDER:
            row[f"mean_prop_{klass}"] = float(sub[f"deconv_class_{klass}"].mean())
        for cluster in clusters:
            row[f"dominant_cluster_frac_{cluster}"] = float((sub["deconv_dominant_cluster"].astype(str) == cluster).mean())
            row[f"mean_prop_{cluster}"] = float(sub[f"deconv_cluster_{cluster}"].mean())
        rows.append(row)
    comp = pd.DataFrame(rows).sort_values(["disease_group", "sample_id"])
    comp.to_csv(result_dir / "figure3_adhip_full_spatial_deconv_chip_composition.csv", index=False)

    long_rows = []
    for _, row in comp.iterrows():
        for label in CLASS_ORDER:
            long_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "disease_group": row["disease_group"],
                    "level": "class",
                    "label": label,
                    "mean_proportion": row.get(f"mean_prop_{label}", np.nan),
                    "dominant_fraction": row.get(f"dominant_class_frac_{label}", np.nan),
                    "n_anchors": row["n_anchors"],
                }
            )
        for label in clusters:
            long_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "disease_group": row["disease_group"],
                    "level": "cluster",
                    "label": label,
                    "mean_proportion": row.get(f"mean_prop_{label}", np.nan),
                    "dominant_fraction": row.get(f"dominant_cluster_frac_{label}", np.nan),
                    "n_anchors": row["n_anchors"],
                }
            )
    comp_long = pd.DataFrame(long_rows)
    comp_long.to_csv(result_dir / "figure3_adhip_full_spatial_deconv_chip_composition_long.csv", index=False)
    return comp, comp_long


def cohen_d(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(float)
    out = np.full(len(p), np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return pd.Series(out, index=p_values.index)
    order = np.argsort(p[ok])
    vals = p[ok][order]
    n = len(vals)
    q = vals * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    ok_idx = np.where(ok)[0]
    out[ok_idx[order]] = q
    return pd.Series(out, index=p_values.index)


def permutation_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int = 20000, seed: int = 13) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    observed = abs(float(np.mean(a) - np.mean(b)))
    pooled = np.concatenate([a, b])
    n_a = len(a)
    count = 1
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        diff = abs(float(np.mean(perm[:n_a]) - np.mean(perm[n_a:])))
        if diff >= observed - 1e-12:
            count += 1
    return float(count / (n_perm + 1))


def ad_vs_control_stats(comp: pd.DataFrame, clusters: list[str], result_dir: Path) -> pd.DataFrame:
    features = ["mean_confidence", "mean_entropy", "mean_residual", "confident_fraction"]
    features += [f"mean_prop_{x}" for x in CLASS_ORDER]
    features += [f"dominant_class_frac_{x}" for x in CLASS_ORDER + ["Low_confidence"]]
    features += [f"mean_prop_{x}" for x in clusters]
    features += [f"dominant_cluster_frac_{x}" for x in clusters]
    rows = []
    for feature in features:
        if feature not in comp:
            continue
        ad = pd.to_numeric(comp.loc[comp["disease_group"].eq("AD"), feature], errors="coerce").dropna()
        control = pd.to_numeric(comp.loc[comp["disease_group"].eq("Control"), feature], errors="coerce").dropna()
        if len(ad) < 2 or len(control) < 2:
            continue
        welch_p = np.nan
        mw_p = np.nan
        if scipy_stats is not None:
            welch_p = float(scipy_stats.ttest_ind(ad, control, equal_var=False, nan_policy="omit").pvalue)
            mw_p = float(scipy_stats.mannwhitneyu(ad, control, alternative="two-sided").pvalue)
        rows.append(
            {
                "comparison": "AD_Hip_AD_vs_Control",
                "feature": feature,
                "n_ad_chips": int(len(ad)),
                "n_control_chips": int(len(control)),
                "mean_ad": float(ad.mean()),
                "mean_control": float(control.mean()),
                "delta_ad_minus_control": float(ad.mean() - control.mean()),
                "cohens_d_ad_minus_control": cohen_d(ad, control),
                "welch_t_p": welch_p,
                "mannwhitney_p": mw_p,
                "permutation_p": permutation_pvalue(ad.to_numpy(float), control.to_numpy(float)),
            }
        )
    stats_df = pd.DataFrame(rows)
    for p_col in ["welch_t_p", "mannwhitney_p", "permutation_p"]:
        if p_col in stats_df:
            stats_df[f"{p_col}_bh_fdr"] = bh_fdr(stats_df[p_col])
    stats_df["abs_delta"] = stats_df["delta_ad_minus_control"].abs()
    stats_df = stats_df.sort_values(["abs_delta", "feature"], ascending=[False, True]).drop(columns=["abs_delta"])
    stats_df.to_csv(result_dir / "figure3_adhip_full_spatial_deconv_ad_vs_control_stats.csv", index=False)
    return stats_df


def marker_deconv_correlations(query_log1p: np.ndarray, genes: list[str], table: pd.DataFrame, clusters: list[str], result_dir: Path) -> pd.DataFrame:
    module_map = {
        "Endothelial": ["CLDN5", "PECAM1", "VWF", "CDH5", "FLT1", "KDR", "RAMP2", "SLC2A1"],
        "Pericyte": ["PDGFRB", "RGS5", "KCNJ8", "ABCC9", "NOTCH3", "CSPG4", "MCAM"],
        "SMC": ["ACTA2", "MYH11", "TAGLN", "CNN1", "MYLK", "MYOCD", "SMTN"],
        "Fibroblast_VLMC": ["COL1A1", "COL1A2", "COL3A1", "COL6A1", "COL6A2", "DCN", "LUM", "APOD"],
    }
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    rows = []
    for klass, module_genes in module_map.items():
        idx = [gene_to_idx[g] for g in module_genes if g in gene_to_idx]
        if not idx:
            continue
        module_score = query_log1p[:, idx].mean(axis=1)
        for target in [f"deconv_class_{klass}"] + [f"deconv_cluster_{c}" for c in clusters if str(c).startswith(klass[:2])]:
            if target not in table:
                continue
            x = table[target].to_numpy(float)
            if np.nanstd(x) == 0 or np.nanstd(module_score) == 0:
                r = np.nan
            else:
                r = float(np.corrcoef(module_score, x)[0, 1])
            rows.append(
                {
                    "module": klass,
                    "target": target,
                    "n_genes": len(idx),
                    "genes": ",".join([g for g in module_genes if g in gene_to_idx]),
                    "pearson_r": r,
                }
            )
    out = pd.DataFrame(rows).sort_values(["module", "pearson_r"], ascending=[True, False])
    out.to_csv(result_dir / "figure3_adhip_full_spatial_deconv_marker_correlations.csv", index=False)
    return out


def plot_class_composition(comp_long: pd.DataFrame, prefix: Path, dpi: int) -> None:
    sub = comp_long[(comp_long["level"].eq("class")) & (comp_long["label"].isin(CLASS_ORDER))].copy()
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    x = np.arange(len(CLASS_ORDER), dtype=float)
    width = 0.33
    for i, disease in enumerate(["Control", "AD"]):
        means = []
        sems = []
        for label in CLASS_ORDER:
            vals = sub.loc[sub["disease_group"].eq(disease) & sub["label"].eq(label), "mean_proportion"].dropna()
            means.append(vals.mean())
            sems.append(vals.sem() if len(vals) > 1 else 0)
            jitter = (i - 0.5) * width + np.linspace(-0.045, 0.045, max(len(vals), 1))[: len(vals)]
            ax.scatter(np.full(len(vals), x[CLASS_ORDER.index(label)] + (i - 0.5) * width) + jitter, vals, s=8, lw=0.55, facecolor="white", edgecolor=DISEASE_PALETTE[disease], alpha=0.9, zorder=3)
        ax.bar(x + (i - 0.5) * width, means, width=width, yerr=sems, color=DISEASE_PALETTE[disease], alpha=0.9, edgecolor="none", capsize=2, label=disease)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER, rotation=18, ha="right")
    ax.set_ylabel("Mean deconvolved proportion")
    ax.set_title("AD hippocampus full spatial vascular deconvolution", loc="left", fontsize=8.4, fontweight="bold")
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(bottom=0)
    save_all(fig, prefix, dpi)


def plot_delta_forest(stats_df: pd.DataFrame, clusters: list[str], prefix: Path, dpi: int) -> None:
    keep = stats_df[stats_df["feature"].isin([f"mean_prop_{x}" for x in CLASS_ORDER + clusters])].copy()
    keep["label"] = keep["feature"].str.replace("mean_prop_", "", regex=False)
    keep = keep.sort_values("delta_ad_minus_control")
    if len(keep) > 24:
        top = keep.reindex(keep["delta_ad_minus_control"].abs().sort_values(ascending=False).index).head(24)
        keep = top.sort_values("delta_ad_minus_control")
    fig_h = max(3.2, 0.17 * len(keep) + 1.0)
    fig, ax = plt.subplots(figsize=(4.4, fig_h))
    colors = ["#C86054" if v > 0 else "#4E79A7" for v in keep["delta_ad_minus_control"]]
    ax.barh(np.arange(len(keep)), keep["delta_ad_minus_control"], color=colors, alpha=0.9)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticks(np.arange(len(keep)))
    ax.set_yticklabels(keep["label"])
    ax.set_xlabel("AD - Control mean deconvolved proportion")
    ax.set_title("Chip-level AD hippocampus changes", loc="left", fontsize=8.4, fontweight="bold")
    save_all(fig, prefix, dpi)


def selected_samples(comp: pd.DataFrame) -> list[str]:
    out = []
    for disease in ["Control", "AD"]:
        part = comp[comp["disease_group"].eq(disease)].sort_values("n_anchors", ascending=False)
        out.extend(part["sample_id"].head(4).tolist())
    return out


def plot_spatial_grid(table: pd.DataFrame, samples: list[str], color_key: str, prefix: Path, dpi: int, palette: dict[str, str]) -> None:
    n = len(samples)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    vals_all = table[color_key].astype(str)
    cats = [c for c in list(palette) if c in set(vals_all)] + [c for c in vals_all.value_counts().index if c not in palette]
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.25 * ncols, 2.0 * nrows), squeeze=False)
    for ax, sample in zip(axes.ravel(), samples):
        sub = table[table["sample_id"].astype(str).eq(sample)]
        vals = sub[color_key].astype(str).to_numpy()
        x = pd.to_numeric(sub["coord_x"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(sub["coord_y"], errors="coerce").to_numpy(float)
        for cat in cats:
            idx = vals == cat
            if np.any(idx):
                ax.scatter(x[idx], y[idx], s=3.8, lw=0, c=[palette.get(cat, "#808080")], alpha=0.82, rasterized=True)
        ax.set_title(sample.split("/")[-1], fontsize=6.8, pad=1.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["left", "bottom", "right", "top"]].set_visible(False)
    for ax in axes.ravel()[len(samples) :]:
        ax.axis("off")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=palette.get(c, "#808080"), markeredgewidth=0, markersize=4, label=c) for c in cats[:24]]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=5.8)
    fig.subplots_adjust(left=0.02, right=0.86, bottom=0.02, top=0.92, wspace=0.06, hspace=0.13)
    save_all(fig, prefix, dpi)


def write_report(result_dir: Path, summary: dict[str, object], stats_df: pd.DataFrame, agreement: pd.DataFrame) -> None:
    def markdown_table(df: pd.DataFrame) -> list[str]:
        if df.empty:
            return ["No records."]
        cols = list(df.columns)
        rows = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, rec in df.iterrows():
            values = []
            for col in cols:
                val = rec[col]
                if isinstance(val, float):
                    values.append(f"{val:.4g}" if np.isfinite(val) else "")
                else:
                    values.append(str(val))
            rows.append("|" + "|".join(values) + "|")
        return rows

    class_stats = stats_df[stats_df["feature"].isin([f"mean_prop_{x}" for x in CLASS_ORDER])].copy()
    top_stats = stats_df.head(12).copy()
    lines = [
        "# Figure 3 AD_Hip full spatial deconvolution v14",
        "",
        "This package reran AD hippocampus spatial vascular deconvolution on the full synchronized AD_Hip vascular anchor set. Disease group is inferred from sample_id (/AD and /Con). Statistics use chips as the unit.",
        "",
        "## Run summary",
        "",
        f"- Anchors: {summary['n_query_anchors']}",
        f"- Chips: {summary['n_chips']} ({summary['n_ad_chips']} AD, {summary['n_control_chips']} Control)",
        f"- Reference cells: {summary['n_reference_cells']}",
        f"- Reference clusters: {summary['n_reference_clusters']}",
        f"- Shared marker genes: {summary['n_genes']}",
        f"- Mean confidence: {summary['mean_confidence']:.4f}",
        f"- Mean residual: {summary['mean_residual']:.4f}",
        f"- Confident anchor fraction: {summary['confident_fraction']:.4f}",
        "",
        "## Broad class AD - Control deltas",
        "",
    ]
    if not class_stats.empty:
        lines.extend(markdown_table(class_stats[["feature", "mean_ad", "mean_control", "delta_ad_minus_control", "cohens_d_ad_minus_control", "welch_t_p", "permutation_p"]]))
    lines.extend(["", "## Largest chip-level changes", ""])
    if not top_stats.empty:
        lines.extend(markdown_table(top_stats[["feature", "mean_ad", "mean_control", "delta_ad_minus_control", "cohens_d_ad_minus_control", "welch_t_p", "permutation_p"]]))
    lines.extend(["", "## Label agreement QA", ""])
    if not agreement.empty:
        lines.extend(markdown_table(agreement))
    else:
        lines.append("No usable existing broad-class labels were available for agreement QA.")
    (result_dir / "Figure3_ADHip_full_spatial_deconvolution_v14_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    figure_dir = Path(args.figure_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    required = [
        args.reference_h5ad,
        args.annotation_csv,
        args.memmap_root / "gene_vocab.txt",
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required full-run inputs:\n" + "\n".join(missing))
    if sc is None:
        raise ImportError("scanpy is required to read the v11 reference h5ad. Run this in the remote BrainST environment.")
    if MemmapDataset is None:
        raise ImportError("cellfm_dataset.memmap is required. Run this under the remote OmniCell_NVU environment.")

    ref = sc.read_h5ad(args.reference_h5ad)
    ref_x, ref_genes = get_marker_matrix(ref, args.matrix_key, args.gene_key)
    n_memmap_genes = len((args.memmap_root / "gene_vocab.txt").read_text().splitlines())
    gene_symbols = load_gene_symbols(args.alias_csv, n_memmap_genes, args.memmap_root)
    genes, ref_idx, query_gene_ids = shared_query_genes(ref_genes, gene_symbols)
    if len(genes) < 20:
        raise ValueError(f"Too few shared marker genes for deconvolution: {len(genes)}")

    query = load_adhip_query(args.annotation_csv, args.max_query_rows)
    print(f"Loaded full AD_Hip query anchors: {len(query)} across {query['sample_id'].nunique()} chips", flush=True)
    raw_query = fetch_expression(query, args.memmap_root, query_gene_ids)
    query_log1p = np.log1p(np.nan_to_num(np.clip(raw_query, 0.0, None), nan=0.0, posinf=0.0, neginf=0.0))

    ref_x = ref_x[:, ref_idx]
    ref_scaled, query_scaled, scale = scale_matrices(ref_x, query_log1p)
    profile, cluster_class = build_profiles(ref, ref_scaled, args.cluster_key, args.class_key)
    profile.columns = genes
    clusters = profile.index.astype(str).tolist()

    profile.to_csv(result_dir / "figure3_adhip_full_spatial_deconv_reference_profiles.csv")
    cluster_class.to_csv(result_dir / "figure3_adhip_full_spatial_deconv_reference_cluster_classes.csv", header=True)
    pd.DataFrame({"gene": genes, "scale_p99": scale, "query_gene_id": query_gene_ids}).to_csv(result_dir / "figure3_adhip_full_spatial_deconv_gene_scaling.csv", index=False)

    weights, residual, fitted_norm = run_nnls(query_scaled, profile)
    source = add_deconv_columns(
        query,
        weights,
        profile,
        cluster_class,
        residual,
        fitted_norm,
        args.confidence_threshold,
        args.residual_threshold,
    )
    source.to_csv(result_dir / "figure3_adhip_full_spatial_deconv_source.csv.gz", index=False, compression="gzip")

    agreement = label_agreement(source, result_dir)
    comp, comp_long = chip_composition(source, clusters, result_dir)
    stats_df = ad_vs_control_stats(comp, clusters, result_dir)
    marker_deconv_correlations(query_log1p, genes, source, clusters, result_dir)

    plot_class_composition(comp_long, figure_dir / "figure3_adhip_full_spatial_deconv_class_composition", args.dpi)
    plot_delta_forest(stats_df, clusters, figure_dir / "figure3_adhip_full_spatial_deconv_delta_forest", args.dpi)
    class_palette = dict(CLASS_PALETTE)
    class_palette["Low_confidence"] = "#BFC5CC"
    plot_spatial_grid(source, selected_samples(comp), "deconv_dominant_class", figure_dir / "figure3_adhip_full_spatial_deconv_selected_chips_class", args.dpi, class_palette)
    cluster_palette = {c: DISTINCT_CLUSTER_COLORS[i % len(DISTINCT_CLUSTER_COLORS)] for i, c in enumerate(clusters)}
    cluster_palette["Low_confidence"] = "#BFC5CC"
    plot_spatial_grid(source, selected_samples(comp), "deconv_dominant_cluster", figure_dir / "figure3_adhip_full_spatial_deconv_selected_chips_cluster", args.dpi, cluster_palette)

    summary = {
        "reference_h5ad": str(args.reference_h5ad),
        "annotation_csv": str(args.annotation_csv),
        "memmap_root": str(args.memmap_root),
        "n_reference_cells": int(ref.n_obs),
        "n_reference_clusters": int(len(clusters)),
        "n_query_anchors": int(len(source)),
        "n_chips": int(source["sample_id"].nunique()),
        "n_ad_chips": int(comp["disease_group"].eq("AD").sum()),
        "n_control_chips": int(comp["disease_group"].eq("Control").sum()),
        "n_genes": int(len(genes)),
        "mean_confidence": float(source["deconv_confidence"].mean()),
        "mean_residual": float(source["deconv_residual"].mean()),
        "confident_fraction": float(source["deconv_is_confident"].astype(bool).mean()),
        "result_dir": str(result_dir),
        "figure_dir": str(figure_dir),
    }
    (result_dir / "figure3_adhip_full_spatial_deconv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(result_dir, summary, stats_df, agreement)
    print(json.dumps(summary, indent=2), flush=True)
    print("top_ad_vs_control_stats")
    print(stats_df.head(16).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

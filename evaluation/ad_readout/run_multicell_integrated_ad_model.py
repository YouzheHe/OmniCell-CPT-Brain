#!/usr/bin/env python
"""Sample-level multi-cell/NVU AD classification and interpretation.

This analysis treats each donor/chip sample as the statistical unit. It
aggregates many NVUs/cells per sample into interpretable feature distributions:
module-by-cell-type activity, LR signaling, cell composition, region
composition, latent summaries, and predicted-risk summaries.
"""

from __future__ import annotations
import os

import argparse
from collections import defaultdict
import json
import math
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_ROOT = Path(os.path.expandvars("${OMNICELL_NVU_RESULTS_ROOT}/04_NVU_ad_model"))
REGION_ORDER = ["FAS", "SLRM", "CA1", "CA2", "CA3", "CA4", "DG", "L1", "L23", "L456", "WM"]
CELLTYPE_ORDER = ["Neuron", "Astro", "Micro", "Endo", "Pericyte", "Oligo", "OPC"]
MANUAL_AD_NVU_GENES = {
    "microglia_immune": [
        "APOE",
        "TREM2",
        "TYROBP",
        "C1QA",
        "C1QB",
        "C1QC",
        "SPP1",
        "ITGAX",
        "LPL",
        "CD74",
        "HLA-DRA",
        "HLA-DRB1",
        "AIF1",
        "CSF1R",
        "CTSS",
        "FTL",
        "FTH1",
        "HCST",
        "SRGN",
    ],
    "astrocyte": [
        "GFAP",
        "AQP4",
        "SLC1A2",
        "SLC1A3",
        "ALDH1L1",
        "CLU",
        "APOE",
        "VIM",
        "SERPINA3",
        "SPARCL1",
        "MAOB",
        "S100B",
        "HSPB1",
        "CRYAB",
        "CHI3L1",
        "BEST1",
    ],
    "endothelial": [
        "CLDN5",
        "SLC2A1",
        "ABCB1",
        "MFSD2A",
        "KDR",
        "FLT1",
        "PECAM1",
        "VWF",
        "PLVAP",
        "ABCG2",
        "ICAM1",
        "VCAM1",
        "CLU",
        "NDRG1",
        "SLC38A5",
    ],
    "pericyte_mural": [
        "PDGFRB",
        "RGS5",
        "CSPG4",
        "ABCC9",
        "KCNJ8",
        "ACTA2",
        "TAGLN",
        "MYH11",
        "MCAM",
        "NOTCH3",
        "A2M",
        "SYNM",
        "ADIRF",
    ],
    "vlmc_fibroblast": [
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "COL6A1",
        "COL6A2",
        "COL6A3",
        "DCN",
        "LUM",
        "FN1",
        "MMP2",
        "APOD",
        "VIM",
    ],
    "neuron": [
        "RBFOX2",
        "SYT1",
        "SNAP25",
        "MAP2",
        "NEFL",
        "NRGN",
        "DAB1",
        "GRIN1",
        "APP",
        "PSEN1",
        "BACE1",
        "MAPT",
        "APLP1",
        "FGF14",
        "KCNQ3",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-subdir", default="multicell_integrated_ad_model")
    parser.add_argument("--latent-pcs", type=int, default=12)
    parser.add_argument("--top-features", type=int, default=50)
    parser.add_argument("--raw-gene-pcs", type=int, default=12)
    parser.add_argument("--fold-selected-genes", type=int, default=40)
    parser.add_argument("--max-curated-genes", type=int, default=180)
    parser.add_argument("--min-feature-std", type=float, default=1e-8)
    return parser.parse_args()


def clean_feature_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.:+\\-]+", "_", str(name))


def is_confounder_feature(name: str) -> bool:
    upper = str(name).upper()
    confounders = ["XIST", "TSIX", "UTY", "USP9Y", "DDX3Y", "KDM5D", "NLGN4Y", "TTTY14", "MALAT1", "NEAT1"]
    if any(x in upper for x in confounders):
        return True
    parts = str(name).split("__")
    if len(parts) >= 3 and parts[0] in {"gene", "curated_gene", "selected_gene"} and is_confounder_gene(parts[1]):
        return True
    return bool(re.search(r"(^|__)MT-|(^|__)RPL|(^|__)RPS|(^|__)HB[A-Z0-9]", upper))


def canonical_gene(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "", str(name).upper())


def is_confounder_gene(name: str) -> bool:
    gene = canonical_gene(name)
    confounders = {"XIST", "TSIX", "UTY", "USP9Y", "DDX3Y", "KDM5D", "NLGN4Y", "TTTY14", "MALAT1", "NEAT1"}
    if gene in confounders:
        return True
    return bool(re.match(r"^(MT-|MTCO|MTND|MTCYB|MTATP|MTRNR|RPL|RPS|RP[SL][0-9]|HB[A-Z0-9])", gene))


def load_existing_predictions(root: Path) -> pd.DataFrame:
    frames = []
    for tissue in ["hip", "ctx"]:
        path = root / "02_Result" / f"nvu_pred_{tissue}_with_metadata.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["tissue"] = tissue
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def sample_summary_from_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    rows = []
    for sample_id, grp in pred.groupby("sample_id", sort=False):
        rec = {
            "sample_id": sample_id,
            "group": grp["group"].iloc[0],
            "label": int(float(grp["label"].iloc[0])),
            "tissue": grp["tissue"].iloc[0],
            "n_nvu_pred_rows": int(len(grp)),
            "risk__available": 1.0,
        }
        prob = pd.to_numeric(grp["nvu_pred_prob"], errors="coerce").to_numpy(float)
        rec.update(
            {
                "risk__nvu_pred_prob__mean": float(np.nanmean(prob)),
                "risk__nvu_pred_prob__std": float(np.nanstd(prob)),
                "risk__nvu_pred_prob__max": float(np.nanmax(prob)),
                "risk__nvu_pred_prob__p90": float(np.nanpercentile(prob, 90)),
                "risk__nvu_pred_prob__high_frac_0.5": float(np.nanmean(prob >= 0.5)),
                "risk__nvu_pred_prob__high_frac_0.8": float(np.nanmean(prob >= 0.8)),
            }
        )
        for region, rg in grp.groupby("region"):
            reg = clean_feature_name(region)
            vals = pd.to_numeric(rg["nvu_pred_prob"], errors="coerce").to_numpy(float)
            rec[f"risk_region__{reg}__mean"] = float(np.nanmean(vals))
            rec[f"region_frac__{reg}"] = float(len(rg) / max(len(grp), 1))
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_nvu_distribution(values: np.ndarray, names: list[str], prefix: str) -> dict[str, float]:
    rec: dict[str, float] = {}
    if values.size == 0:
        return rec
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        maxv = np.nanmax(values, axis=0)
        p90 = np.nanpercentile(values, 90, axis=0)
        high = np.nanmean(values > mean + std, axis=0)
    for i, name in enumerate(names):
        clean = clean_feature_name(name)
        if is_confounder_feature(clean):
            continue
        rec[f"{prefix}__{clean}__mean"] = float(mean[i])
        rec[f"{prefix}__{clean}__std"] = float(std[i])
        rec[f"{prefix}__{clean}__max"] = float(maxv[i])
        rec[f"{prefix}__{clean}__p90"] = float(p90[i])
        rec[f"{prefix}__{clean}__high_frac"] = float(high[i])
    return rec


def load_curated_gene_sources(root: Path, available_genes: list[str], max_genes: int) -> tuple[list[str], pd.DataFrame]:
    available_by_upper = {canonical_gene(g): g for g in available_genes if not is_confounder_gene(g)}
    scores: dict[str, float] = defaultdict(float)
    sources: dict[str, set[str]] = defaultdict(set)

    for module, genes in MANUAL_AD_NVU_GENES.items():
        for gene in genes:
            key = canonical_gene(gene)
            if key in available_by_upper and not is_confounder_gene(key):
                scores[key] += 5.0
                sources[key].add(f"manual:{module}")

    data_dir = root / "00_Data"
    abeta_path = data_dir / "Abeta_associated_genes_intersections.csv"
    if abeta_path.exists():
        abeta = pd.read_csv(abeta_path)
        gene_col = "gene" if "gene" in abeta.columns else abeta.columns[0]
        for _, row in abeta.iterrows():
            key = canonical_gene(row[gene_col])
            if key in available_by_upper and not is_confounder_gene(key):
                scores[key] += 3.0
                sources[key].add(f"abeta:{row.get('intersection', 'intersection')}")

    for fn in ["NVU.Module.csv", "Cortex_up_NVU.Module.csv"]:
        path = data_dir / fn
        if not path.exists():
            continue
        mod = pd.read_csv(path)
        if "gene_name" not in mod.columns:
            continue
        for _, row in mod.iterrows():
            key = canonical_gene(row["gene_name"])
            if key in available_by_upper and not is_confounder_gene(key):
                scores[key] += 1.0
                sources[key].add(f"module:{row.get('module', row.get('color', 'unknown'))}")

    result_dir = root / "02_Result"
    for tissue in ["hip", "ctx"]:
        gw_path = result_dir / f"gene_weight_{tissue}.csv"
        if gw_path.exists():
            gw = pd.read_csv(gw_path).sort_values("weight", ascending=False).head(80)
            for _, row in gw.iterrows():
                key = canonical_gene(row["gene"])
                if key in available_by_upper and not is_confounder_gene(key):
                    scores[key] += float(row.get("weight", 0.0)) * 4.0
                    sources[key].add(f"model_weight:{tissue}")
        sens_path = result_dir / f"sensitive_genes_{tissue}.csv"
        if sens_path.exists():
            sg = pd.read_csv(sens_path).sort_values("rank_score", ascending=False).head(80)
            for _, row in sg.iterrows():
                key = canonical_gene(row["feature"])
                if key in available_by_upper and not is_confounder_gene(key):
                    scores[key] += min(float(row.get("rank_score", 0.0)), 10.0) / 2.0
                    sources[key].add(f"sensitive:{tissue}")

    rows = []
    for key, score in scores.items():
        rows.append(
            {
                "gene": available_by_upper[key],
                "gene_upper": key,
                "curated_score": float(score),
                "sources": ";".join(sorted(sources[key])),
            }
        )
    source_df = pd.DataFrame(rows)
    if source_df.empty:
        return [], source_df
    source_df = source_df.sort_values(["curated_score", "gene"], ascending=[False, True])
    curated = source_df["gene"].head(max_genes).tolist()
    source_df["selected_for_model"] = source_df["gene"].isin(curated)
    return curated, source_df


def aggregate_gene_lr_features(
    result: dict,
    curated_genes: set[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    gene_names = list(result.get("node_gene_names", result.get("genes_ok", [])))
    lr_names = list(result.get("node_lr_names", []))
    gene_dim = len(gene_names)
    lr_dim = len(lr_names)
    gene_sum = np.zeros(gene_dim, dtype=np.float64)
    lr_sum = np.zeros(lr_dim, dtype=np.float64)
    n_cells = 0
    for graph in result["nvu_graphs"]:
        x = graph.x.detach().cpu().numpy()
        if gene_dim:
            gene_sum += x[:, :gene_dim].sum(axis=0)
        if lr_dim:
            lr_sum += x[:, -lr_dim:].sum(axis=0)
        n_cells += x.shape[0]
    denom = max(n_cells, 1)

    gene_values = gene_sum / denom if gene_dim else np.array([])
    lr_values = lr_sum / denom if lr_dim else np.array([])
    gene_rec = {}
    curated_rec = {}
    for gene, value in zip(gene_names, gene_values):
        if is_confounder_gene(gene):
            continue
        clean = clean_feature_name(gene)
        gene_rec[f"gene__{clean}__mean_cell"] = float(value)
        if canonical_gene(gene) in curated_genes:
            curated_rec[f"curated_gene__{clean}__mean_cell"] = float(value)

    lr_rec = {}
    for lr_name, value in zip(lr_names, lr_values):
        lr_rec[f"node_lr__{clean_feature_name(lr_name)}__mean_cell"] = float(value)
    return gene_rec, curated_rec, lr_rec


def load_latent_arrays(root: Path) -> dict[str, np.ndarray]:
    data_dir = root / "00_Data"
    return {
        "hip": np.load(data_dir / "nvu_latent_hip.npy", mmap_mode="r"),
        "ctx": np.load(data_dir / "nvu_latent_ctx.npy", mmap_mode="r"),
        "hip_cellrepr": np.load(data_dir / "nvu_latent_hip_cellrepr.npy", mmap_mode="r"),
        "ctx_cellrepr": np.load(data_dir / "nvu_latent_ctx_cellrepr.npy", mmap_mode="r"),
    }


def load_latent_metadata(root: Path) -> dict[str, pd.DataFrame]:
    data_dir = root / "00_Data"
    out = {}
    for tissue in ["hip", "ctx"]:
        path = data_dir / f"nvu_latent_{tissue}_meta_with_metadata.csv"
        if not path.exists():
            path = data_dir / f"nvu_latent_{tissue}_meta.csv"
        out[tissue] = pd.read_csv(path)
    return out


def sample_rows_from_results(
    root: Path,
    latent_pcs: int,
    max_curated_genes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open(root / "00_Data" / "all_results_v2.pkl", "rb") as handle:
        results = pickle.load(handle)
    latent = load_latent_arrays(root)
    latent_meta = load_latent_metadata(root)
    all_gene_names = []
    for result in results:
        all_gene_names.extend(list(result.get("node_gene_names", result.get("genes_ok", []))))
    available_genes = sorted(set(all_gene_names))
    curated_genes, curated_gene_sources = load_curated_gene_sources(root, available_genes, max_curated_genes)
    curated_gene_set = {canonical_gene(g) for g in curated_genes}

    # Build PCA coordinates globally per tissue so latent summaries are comparable.
    pca_models: dict[str, PCA] = {}
    for tissue in ["hip", "ctx"]:
        arr = np.asarray(latent[tissue])
        n_comp = min(latent_pcs, arr.shape[1], arr.shape[0] - 1)
        pca_models[tissue] = PCA(n_components=n_comp, random_state=13).fit(arr)

    rows = []
    scalar_name_rows = []
    for result in results:
        tissue = str(result["tissue"])
        n_nvu = int(result["n_nvu"])

        rec = {
            "sample_id": result["sample_id"],
            "group": "AD" if int(result["label"]) == 1 else "Control",
            "label": int(result["label"]),
            "tissue": tissue,
            "n_nvu": n_nvu,
        }
        scalar_names = list(result["nvu_scalar_names"])
        scalar = np.asarray(result["nvu_scalar_feats"], dtype=float)
        rec.update(summarize_nvu_distribution(scalar, scalar_names, "scalar"))
        gene_rec, curated_rec, lr_rec = aggregate_gene_lr_features(result, curated_gene_set)
        rec.update(gene_rec)
        rec.update(curated_rec)
        rec.update(lr_rec)

        regions = pd.Series(result["nvu_regions"]).fillna("unknown").astype(str)
        for region in REGION_ORDER + sorted(set(regions) - set(REGION_ORDER)):
            rec[f"region_frac__{clean_feature_name(region)}"] = float((regions == region).mean())

        meta = latent_meta[tissue]
        idx = np.flatnonzero(meta["sample_id"].astype(str).to_numpy() == str(result["sample_id"]))
        if len(idx) > 0:
            lat = np.asarray(latent[tissue][idx])
            pcs = pca_models[tissue].transform(lat)
            pc_names = [f"latent_pc{i + 1}" for i in range(pcs.shape[1])]
            rec.update(summarize_nvu_distribution(pcs, pc_names, "latent"))

        rows.append(rec)
        for name in scalar_names:
            scalar_name_rows.append({"tissue": tissue, "feature": name})
    return pd.DataFrame(rows), pd.DataFrame(scalar_name_rows).drop_duplicates(), curated_gene_sources


def feature_group(name: str) -> tuple[str, str, str]:
    parts = str(name).split("__")
    if name.startswith("scalar__"):
        body = parts[1:-1]
        stat = parts[-1]
        base = "__".join(body)
        if base.startswith("LR__") or base.startswith("LR_"):
            return "LR signaling", base, stat
        if base.startswith("ratio__") or base.startswith("ratio_"):
            return "cell composition", base, stat
        if base in {"dist_mean", "n_cells"}:
            return "density/geometry", base, stat
        if len(body) >= 2:
            module = body[0]
            celltype = body[1]
            return f"module:{module}", celltype, stat
        return "scalar other", base, stat
    if name.startswith("risk"):
        return "model-risk distribution", "__".join(parts[:-1]), parts[-1]
    if name.startswith("region_frac"):
        return "region composition", "__".join(parts[1:]), "fraction"
    if name.startswith("latent"):
        return "CPT latent distribution", "__".join(parts[:-1]), parts[-1]
    if name.startswith("raw_gene_svd"):
        return "raw gene SVD", "__".join(parts[:-1]), parts[-1]
    if name.startswith("curated_gene"):
        return "curated AD/NVU genes", "__".join(parts[1:-1]), parts[-1]
    if name.startswith("selected_gene"):
        return "fold-internal selected genes", "__".join(parts[1:-1]), parts[-1]
    if name.startswith("gene__"):
        return "all genes", "__".join(parts[1:-1]), parts[-1]
    if name.startswith("node_lr"):
        return "node LR signaling", "__".join(parts[1:-1]), parts[-1]
    return "other", name, ""


def prepare_feature_matrix(df: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    id_cols = {"sample_id", "group", "label", "tissue"}
    cols = [c for c in df.columns if c not in id_cols]
    if feature_set == "scalar":
        cols = [c for c in cols if c.startswith("scalar__")]
    elif feature_set == "scalar_region":
        cols = [c for c in cols if c.startswith("scalar__") or c.startswith("region_frac__")]
    elif feature_set == "scalar_region_risk":
        cols = [c for c in cols if c.startswith("scalar__") or c.startswith("region_frac__") or c.startswith("risk")]
    elif feature_set == "latent":
        cols = [c for c in cols if c.startswith("latent__")]
    elif feature_set == "raw_gene":
        cols = [c for c in cols if c.startswith("gene__")]
    elif feature_set == "curated_gene":
        cols = [c for c in cols if c.startswith("curated_gene__")]
    elif feature_set == "ad_informed_static":
        cols = [
            c
            for c in cols
            if c.startswith(("latent__", "scalar__", "region_frac__", "curated_gene__", "node_lr__"))
        ]
    elif feature_set == "combined":
        cols = [c for c in cols if c.startswith(("scalar__", "region_frac__", "risk", "latent__"))]
    else:
        raise ValueError(feature_set)

    x = df[cols].copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(0.0)
    std = x.std(axis=0)
    keep = [c for c in cols if float(std[c]) > 1e-8 and not is_confounder_feature(c)]
    return x[keep], keep


def _safe_metric_summary(feature_set: str, y: np.ndarray, oof: np.ndarray) -> pd.DataFrame:
    ok = np.isfinite(oof)
    if ok.sum() and len(np.unique(y[ok])) == 2:
        pred = (oof[ok] >= 0.5).astype(int)
        return pd.DataFrame(
            [
                {
                    "feature_set": feature_set,
                    "n_samples": int(ok.sum()),
                    "n_ad": int(y[ok].sum()),
                    "n_control": int((1 - y[ok]).sum()),
                    "auroc": float(roc_auc_score(y[ok], oof[ok])),
                    "balanced_accuracy": float(balanced_accuracy_score(y[ok], pred)),
                    "macro_f1": float(f1_score(y[ok], pred, average="macro")),
                }
            ]
        )
    return pd.DataFrame()


def make_l1_logistic(random_state: int, c_value: float = 0.35):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            C=c_value,
            max_iter=2000,
            random_state=random_state,
        ),
    )


def select_genes_train_only(x_train: pd.DataFrame, y_train: np.ndarray, top_n: int) -> list[str]:
    scores = []
    for col in x_train.columns:
        if is_confounder_feature(col):
            continue
        ad = x_train.loc[y_train == 1, col].to_numpy(float)
        ctrl = x_train.loc[y_train == 0, col].to_numpy(float)
        pooled = math.sqrt((np.nanvar(ad) + np.nanvar(ctrl)) / 2.0)
        if pooled <= 0 or not np.isfinite(pooled):
            continue
        d = (float(np.nanmean(ad)) - float(np.nanmean(ctrl))) / pooled
        if np.isfinite(d):
            scores.append((abs(d), col, d))
    scores.sort(reverse=True)
    return [col for _, col, _ in scores[:top_n]]


def loo_probe(df: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x, cols = prepare_feature_matrix(df, feature_set)
    y = df["label"].astype(int).to_numpy()
    if len(np.unique(y)) < 2 or len(y) < 4 or len(cols) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    loo = LeaveOneOut()
    oof = np.full(len(df), np.nan, dtype=float)
    fold_rows = []
    coef_rows = []
    for fold, (train, test) in enumerate(loo.split(x), start=1):
        if len(np.unique(y[train])) < 2:
            continue
        pipe = make_l1_logistic(fold)
        pipe.fit(x.iloc[train], y[train])
        prob = float(pipe.predict_proba(x.iloc[test])[:, 1][0])
        oof[test[0]] = prob
        coef = pipe.named_steps["logisticregression"].coef_.ravel()
        nonzero = np.flatnonzero(np.abs(coef) > 1e-10)
        for j in nonzero:
            grp, entity, stat = feature_group(cols[j])
            coef_rows.append(
                {
                    "feature_set": feature_set,
                    "fold": fold,
                    "heldout_sample_id": df.iloc[test[0]]["sample_id"],
                    "feature": cols[j],
                    "feature_group": grp,
                    "entity": entity,
                    "stat": stat,
                    "coef": float(coef[j]),
                    "abs_coef": float(abs(coef[j])),
                }
            )
        fold_rows.append(
            {
                "feature_set": feature_set,
                "fold": fold,
                "heldout_sample_id": df.iloc[test[0]]["sample_id"],
                "heldout_label": int(y[test[0]]),
                "pred_prob": prob,
            }
        )
    summary = _safe_metric_summary(feature_set, y, oof)
    folds = pd.DataFrame(fold_rows)
    coefs = pd.DataFrame(coef_rows)
    return summary, folds, coefs


def loo_raw_gene_svd(df: pd.DataFrame, n_components: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_gene, gene_cols = prepare_feature_matrix(df, "raw_gene")
    y = df["label"].astype(int).to_numpy()
    if len(np.unique(y)) < 2 or len(y) < 4 or len(gene_cols) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    loo = LeaveOneOut()
    oof = np.full(len(df), np.nan, dtype=float)
    fold_rows = []
    coef_rows = []
    for fold, (train, test) in enumerate(loo.split(x_gene), start=1):
        if len(np.unique(y[train])) < 2:
            continue
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_gene.iloc[train])
        x_test_scaled = scaler.transform(x_gene.iloc[test])
        n_comp = max(1, min(n_components, x_train_scaled.shape[0] - 1, x_train_scaled.shape[1]))
        pca = PCA(n_components=n_comp, random_state=fold)
        z_train = pca.fit_transform(x_train_scaled)
        z_test = pca.transform(x_test_scaled)
        clf = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            C=0.5,
            max_iter=2000,
            random_state=fold,
        )
        clf.fit(z_train, y[train])
        prob = float(clf.predict_proba(z_test)[:, 1][0])
        oof[test[0]] = prob
        fold_rows.append(
            {
                "feature_set": "raw_gene_svd",
                "fold": fold,
                "heldout_sample_id": df.iloc[test[0]]["sample_id"],
                "heldout_label": int(y[test[0]]),
                "pred_prob": prob,
                "n_components": int(n_comp),
            }
        )
        raw_importance = np.abs(pca.components_.T @ clf.coef_.ravel())
        top_idx = np.argsort(raw_importance)[::-1][:50]
        for j in top_idx:
            grp, entity, stat = feature_group(gene_cols[j])
            coef_rows.append(
                {
                    "feature_set": "raw_gene_svd",
                    "fold": fold,
                    "heldout_sample_id": df.iloc[test[0]]["sample_id"],
                    "feature": gene_cols[j],
                    "feature_group": grp,
                    "entity": entity,
                    "stat": stat,
                    "coef": float(raw_importance[j]),
                    "abs_coef": float(raw_importance[j]),
                }
            )
    return _safe_metric_summary("raw_gene_svd", y, oof), pd.DataFrame(fold_rows), pd.DataFrame(coef_rows)


def loo_ad_informed_cpt(
    df: pd.DataFrame,
    selected_gene_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_base, _ = prepare_feature_matrix(df, "ad_informed_static")
    x_gene, _ = prepare_feature_matrix(df, "raw_gene")
    y = df["label"].astype(int).to_numpy()
    if len(np.unique(y)) < 2 or len(y) < 4 or x_base.shape[1] == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    loo = LeaveOneOut()
    oof = np.full(len(df), np.nan, dtype=float)
    fold_rows = []
    coef_rows = []
    selected_rows = []
    for fold, (train, test) in enumerate(loo.split(x_base), start=1):
        if len(np.unique(y[train])) < 2:
            continue
        selected = select_genes_train_only(x_gene.iloc[train], y[train], selected_gene_n)
        x_fold = pd.concat(
            [
                x_base,
                x_gene[selected].rename(columns=lambda c: c.replace("gene__", "selected_gene__", 1)),
            ],
            axis=1,
        )
        selected_renamed = [c.replace("gene__", "selected_gene__", 1) for c in selected]
        pipe = make_l1_logistic(fold, c_value=0.22)
        pipe.fit(x_fold.iloc[train], y[train])
        prob = float(pipe.predict_proba(x_fold.iloc[test])[:, 1][0])
        oof[test[0]] = prob
        coef = pipe.named_steps["logisticregression"].coef_.ravel()
        cols = list(x_fold.columns)
        for selected_col in selected_renamed:
            selected_rows.append(
                {
                    "feature_set": "ad_informed_cpt",
                    "fold": fold,
                    "heldout_sample_id": df.iloc[test[0]]["sample_id"],
                    "feature": selected_col,
                    "gene": selected_col.split("__")[1],
                }
            )
        nonzero = np.flatnonzero(np.abs(coef) > 1e-10)
        for j in nonzero:
            grp, entity, stat = feature_group(cols[j])
            coef_rows.append(
                {
                    "feature_set": "ad_informed_cpt",
                    "fold": fold,
                    "heldout_sample_id": df.iloc[test[0]]["sample_id"],
                    "feature": cols[j],
                    "feature_group": grp,
                    "entity": entity,
                    "stat": stat,
                    "coef": float(coef[j]),
                    "abs_coef": float(abs(coef[j])),
                }
            )
        fold_rows.append(
            {
                "feature_set": "ad_informed_cpt",
                "fold": fold,
                "heldout_sample_id": df.iloc[test[0]]["sample_id"],
                "heldout_label": int(y[test[0]]),
                "pred_prob": prob,
                "n_selected_genes": int(len(selected)),
            }
        )
    summary = _safe_metric_summary("ad_informed_cpt", y, oof)
    return summary, pd.DataFrame(fold_rows), pd.DataFrame(coef_rows), pd.DataFrame(selected_rows)


def univariate_feature_stats(df: pd.DataFrame, feature_set: str, top_n: int) -> pd.DataFrame:
    x, cols = prepare_feature_matrix(df, feature_set)
    y = df["label"].astype(int).to_numpy()
    rows = []
    for col in cols:
        ad = x.loc[y == 1, col].to_numpy(float)
        ctrl = x.loc[y == 0, col].to_numpy(float)
        if np.nanstd(x[col]) <= 0:
            continue
        try:
            pvalue = stats.mannwhitneyu(ad, ctrl, alternative="two-sided").pvalue
        except ValueError:
            pvalue = np.nan
        pooled = math.sqrt((np.nanvar(ad) + np.nanvar(ctrl)) / 2.0)
        delta = float(np.nanmean(ad) - np.nanmean(ctrl))
        d = delta / pooled if pooled > 0 else np.nan
        grp, entity, stat = feature_group(col)
        rows.append(
            {
                "feature_set": feature_set,
                "feature": col,
                "feature_group": grp,
                "entity": entity,
                "stat": stat,
                "mean_ad": float(np.nanmean(ad)),
                "mean_control": float(np.nanmean(ctrl)),
                "diff_ad_minus_control": delta,
                "cohens_d": float(d),
                "abs_cohens_d": float(abs(d)) if np.isfinite(d) else np.nan,
                "mw_pvalue": float(pvalue) if np.isfinite(pvalue) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank_score"] = out["abs_cohens_d"].fillna(0) * -np.log10(out["mw_pvalue"].fillna(1).clip(lower=1e-300))
    return out.sort_values("rank_score", ascending=False).head(top_n)


def aggregate_coefficients(coefs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if coefs.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    feature_weights = (
        coefs.groupby(["feature_set", "feature", "feature_group", "entity", "stat"], dropna=False)
        .agg(
            mean_coef=("coef", "mean"),
            mean_abs_coef=("abs_coef", "mean"),
            n_folds_nonzero=("fold", "count"),
        )
        .reset_index()
        .sort_values(["feature_set", "mean_abs_coef"], ascending=[True, False])
    )
    group_weights = (
        feature_weights.groupby(["feature_set", "feature_group"], dropna=False)
        .agg(
            importance=("mean_abs_coef", "sum"),
            n_features=("feature", "count"),
            signed_mean=("mean_coef", "mean"),
        )
        .reset_index()
        .sort_values(["feature_set", "importance"], ascending=[True, False])
    )
    entity_weights = (
        feature_weights.groupby(["feature_set", "feature_group", "entity"], dropna=False)
        .agg(
            importance=("mean_abs_coef", "sum"),
            n_features=("feature", "count"),
            signed_mean=("mean_coef", "mean"),
        )
        .reset_index()
        .sort_values(["feature_set", "importance"], ascending=[True, False])
    )
    return feature_weights, group_weights, entity_weights


def aggregate_coefficients_by_dataset(coefs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if coefs.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    group_cols = ["dataset", "feature_set", "feature", "feature_group", "entity", "stat"]
    feature_weights = (
        coefs.groupby(group_cols, dropna=False)
        .agg(
            mean_coef=("coef", "mean"),
            mean_abs_coef=("abs_coef", "mean"),
            n_folds_nonzero=("fold", "count"),
        )
        .reset_index()
        .sort_values(["dataset", "feature_set", "mean_abs_coef"], ascending=[True, True, False])
    )
    group_weights = (
        feature_weights.groupby(["dataset", "feature_set", "feature_group"], dropna=False)
        .agg(importance=("mean_abs_coef", "sum"), n_features=("feature", "count"), signed_mean=("mean_coef", "mean"))
        .reset_index()
        .sort_values(["dataset", "feature_set", "importance"], ascending=[True, True, False])
    )
    entity_weights = (
        feature_weights.groupby(["dataset", "feature_set", "feature_group", "entity"], dropna=False)
        .agg(importance=("mean_abs_coef", "sum"), n_features=("feature", "count"), signed_mean=("mean_coef", "mean"))
        .reset_index()
        .sort_values(["dataset", "feature_set", "importance"], ascending=[True, True, False])
    )
    return feature_weights, group_weights, entity_weights


def df_to_md(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return ""
    show = df.head(max_rows).copy() if max_rows is not None else df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def add_dataset_column(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.insert(0, "dataset", dataset)
    return out


def run_feature_suite(
    sample_df: pd.DataFrame,
    dataset: str,
    top_features: int,
    raw_gene_pcs: int,
    selected_gene_n: int,
    include_risk: bool,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    feature_sets = ["latent", "raw_gene_svd", "scalar", "scalar_region", "curated_gene", "ad_informed_static", "ad_informed_cpt"]
    if include_risk:
        feature_sets.append("scalar_region_risk")
    summaries = []
    folds = []
    coefs = []
    univar = []
    selected = []
    for feature_set in feature_sets:
        if feature_set == "raw_gene_svd":
            summary, fold, coef = loo_raw_gene_svd(sample_df, raw_gene_pcs)
        elif feature_set == "ad_informed_cpt":
            summary, fold, coef, selected_df = loo_ad_informed_cpt(sample_df, selected_gene_n)
            if not selected_df.empty:
                selected.append(add_dataset_column(selected_df, dataset))
        else:
            summary, fold, coef = loo_probe(sample_df, feature_set)
        if not summary.empty:
            summaries.append(add_dataset_column(summary, dataset))
        if not fold.empty:
            folds.append(add_dataset_column(fold, dataset))
        if not coef.empty:
            coefs.append(add_dataset_column(coef, dataset))
        if feature_set not in {"raw_gene_svd", "ad_informed_cpt"}:
            u = univariate_feature_stats(sample_df, feature_set, top_features)
            if not u.empty:
                univar.append(add_dataset_column(u, dataset))
    return summaries, folds, coefs, univar, selected


def write_markdown_report(
    out_dir: Path,
    summary_df: pd.DataFrame,
    group_weights: pd.DataFrame,
    feature_weights: pd.DataFrame,
    selected_freq: pd.DataFrame,
    curated_sources: pd.DataFrame,
) -> None:
    lines = [
        "# Multicell Integrated AD Model",
        "",
        "Statistical unit: sample/chip. Each row used for AUROC is a held-out sample, not an individual NVU/cell.",
        "",
        "## AUROC Summary",
        "",
    ]
    if not summary_df.empty:
        show = summary_df.sort_values(["dataset", "auroc"], ascending=[True, False])
        lines.append(df_to_md(show))
    else:
        lines.append("No valid AUROC summary was produced.")
    lines.extend(["", "## Top Feature Groups", ""])
    if not group_weights.empty:
        for dataset in group_weights["dataset"].drop_duplicates().tolist():
            lines.append(f"### {dataset}")
            lines.append(df_to_md(group_weights[group_weights["dataset"] == dataset], max_rows=12))
            lines.append("")
    lines.extend(["", "## Top Features", ""])
    if not feature_weights.empty:
        best_sets = summary_df.sort_values("auroc", ascending=False)["feature_set"].head(3).tolist() if not summary_df.empty else []
        for feature_set in best_sets:
            lines.append(f"### {feature_set}")
            lines.append(df_to_md(feature_weights[feature_weights["feature_set"] == feature_set], max_rows=20))
            lines.append("")
    lines.extend(["", "## Fold-Internal Selected Genes", ""])
    if not selected_freq.empty:
        lines.append(df_to_md(selected_freq, max_rows=40))
    else:
        lines.append("No fold-internal selected genes were recorded.")
    lines.extend(["", "## Curated Gene Sources", ""])
    if not curated_sources.empty:
        lines.append(df_to_md(curated_sources, max_rows=60))
    lines.extend(
        [
            "",
            "## Figure Panel Recommendation",
            "",
            "A. Feature schematic: CPT latent summaries, curated AD/NVU modules, curated genes, and fold-internal selected genes.",
            "",
            "B. Held-out sample-level AUROC: raw gene SVD vs CPT latent vs curated modules vs AD-informed CPT.",
            "",
            "C. Interpretable cell/module heatmap: module-color by cell-type coefficients for microglia/immune, astrocyte, endothelial, pericyte/mural, VLMC/fibroblast, and neuronal axes.",
            "",
            "D. Gene dot plot: curated and fold-selected genes with selection frequency, coefficient sign, and AD-Control direction.",
            "",
            "E. Spatial/NVU biology: region composition and LR pairs, highlighting plaque-associated hippocampal regions and glial-vascular communication.",
            "",
        ]
    )
    (out_dir / "multicell_integrated_ad_model_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.root / "02_Result" / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_df, scalar_names, curated_sources = sample_rows_from_results(
        args.root,
        args.latent_pcs,
        args.max_curated_genes,
    )
    pred_summary = sample_summary_from_predictions(load_existing_predictions(args.root))
    if not pred_summary.empty:
        merge_cols = ["sample_id", "group", "label", "tissue"]
        extra_cols = [c for c in pred_summary.columns if c not in merge_cols]
        sample_df = sample_df.merge(pred_summary[merge_cols + extra_cols], on=merge_cols, how="left")
    sample_df = sample_df.replace([np.inf, -np.inf], np.nan)
    sample_df["risk__available"] = sample_df.get("risk__available", pd.Series(np.nan, index=sample_df.index)).fillna(0.0)
    risk_cols = [c for c in sample_df.columns if c.startswith("risk") and c != "risk__available"]
    qc_df = sample_df[sample_df["risk__available"] == 1.0].copy()
    sample_df = sample_df.fillna(0.0)
    qc_df = qc_df.fillna(0.0)
    sample_df.to_csv(out_dir / "multicell_sample_feature_matrix.csv", index=False)
    qc_df.to_csv(out_dir / "multicell_qc36_feature_matrix.csv", index=False)
    scalar_names.to_csv(out_dir / "multicell_scalar_feature_names.csv", index=False)
    curated_sources.to_csv(out_dir / "multicell_curated_gene_sources.csv", index=False)

    summaries, folds, coefs, univar, selected = run_feature_suite(
        sample_df,
        "all44",
        args.top_features,
        args.raw_gene_pcs,
        args.fold_selected_genes,
        include_risk=False,
    )
    if len(qc_df) >= 4 and len(qc_df["label"].unique()) == 2:
        q_summaries, q_folds, q_coefs, q_univar, q_selected = run_feature_suite(
            qc_df,
            "qc36_risk_available",
            args.top_features,
            args.raw_gene_pcs,
            args.fold_selected_genes,
            include_risk=True,
        )
        summaries += q_summaries
        folds += q_folds
        coefs += q_coefs
        univar += q_univar
        selected += q_selected

    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    fold_df = pd.concat(folds, ignore_index=True) if folds else pd.DataFrame()
    coef_df = pd.concat(coefs, ignore_index=True) if coefs else pd.DataFrame()
    univar_df = pd.concat(univar, ignore_index=True) if univar else pd.DataFrame()
    selected_df = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    feature_weights, group_weights, entity_weights = aggregate_coefficients_by_dataset(coef_df)
    selected_freq = pd.DataFrame()
    if not selected_df.empty:
        selected_freq = (
            selected_df.groupby(["dataset", "gene"], dropna=False)
            .agg(n_folds_selected=("fold", "count"))
            .reset_index()
            .sort_values(["dataset", "n_folds_selected", "gene"], ascending=[True, False, True])
        )

    summary_df.to_csv(out_dir / "multicell_model_auc_summary.csv", index=False)
    fold_df.to_csv(out_dir / "multicell_model_loo_predictions.csv", index=False)
    coef_df.to_csv(out_dir / "multicell_model_fold_coefficients.csv", index=False)
    feature_weights.to_csv(out_dir / "multicell_model_feature_weights.csv", index=False)
    group_weights.to_csv(out_dir / "multicell_model_feature_group_importance.csv", index=False)
    entity_weights.to_csv(out_dir / "multicell_model_entity_importance.csv", index=False)
    univar_df.to_csv(out_dir / "multicell_model_univariate_feature_stats.csv", index=False)
    selected_df.to_csv(out_dir / "multicell_model_fold_selected_genes.csv", index=False)
    selected_freq.to_csv(out_dir / "multicell_model_selected_gene_frequency.csv", index=False)
    write_markdown_report(out_dir, summary_df, group_weights, feature_weights, selected_freq, curated_sources)

    report = {
        "root": str(args.root),
        "output_dir": str(out_dir),
        "n_samples": int(len(sample_df)),
        "n_qc_risk_samples": int(len(qc_df)),
        "group_counts": sample_df["group"].value_counts().to_dict(),
        "tissue_counts": sample_df["tissue"].value_counts().to_dict(),
        "risk_feature_columns": risk_cols,
        "auc_summary": summary_df.to_dict("records") if not summary_df.empty else [],
        "top_group_importance": group_weights.head(20).to_dict("records") if not group_weights.empty else [],
        "top_features": feature_weights.head(30).to_dict("records") if not feature_weights.empty else [],
        "outputs": {
            "sample_feature_matrix": str(out_dir / "multicell_sample_feature_matrix.csv"),
            "qc36_feature_matrix": str(out_dir / "multicell_qc36_feature_matrix.csv"),
            "auc_summary": str(out_dir / "multicell_model_auc_summary.csv"),
            "feature_weights": str(out_dir / "multicell_model_feature_weights.csv"),
            "feature_group_importance": str(out_dir / "multicell_model_feature_group_importance.csv"),
            "entity_importance": str(out_dir / "multicell_model_entity_importance.csv"),
            "univariate_feature_stats": str(out_dir / "multicell_model_univariate_feature_stats.csv"),
            "curated_gene_sources": str(out_dir / "multicell_curated_gene_sources.csv"),
            "fold_selected_genes": str(out_dir / "multicell_model_fold_selected_genes.csv"),
            "selected_gene_frequency": str(out_dir / "multicell_model_selected_gene_frequency.csv"),
            "markdown_report": str(out_dir / "multicell_integrated_ad_model_report.md"),
        },
    }
    (out_dir / "multicell_model_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

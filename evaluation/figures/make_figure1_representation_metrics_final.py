#!/usr/bin/env python
"""Figure 1 representation metrics independent of Figure 2 task benchmarks.

The panel produced here is deliberately not a single-cell annotation or spatial
deconvolution benchmark.  It audits whether the representation preserves
biological state information while reducing cohort/modality structure on the
same 50,000 validation anchors used by the original Figure 1 probes.
"""

from __future__ import annotations
import os

import json
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/nvu_vascular"))
RESULTS = PROJECT / "results"
VALIDATION = RESULTS / "atlas_validation_full_ridge"
VASCULAR_FT_RESULT = RESULTS / "vascular_omnicell_cpt_gse256490_adult_control_supcon_backbone_nonzero_hvg_all_data"
MULTITASK_FT_RESULT = Path(os.environ.get("FIG1_MULTITASK_FT_RESULT", RESULTS / "figure1_multitask_cpt_alignment_full"))
CPT_VALIDATION_RESULT = Path(os.environ.get("FIG1_CPT_VALIDATION_RESULT", RESULTS / "figure1_multitask_cpt_alignment_validation_embedding"))
FIG1 = PROJECT / "figures" / "figure1_final_panels"
SRC = FIG1 / "source_data"
PACKAGE = PROJECT / "figure1_figure2_complete_package_20260624" / "Figure1"
PKG_PLOTS = PACKAGE / "plots" / "figure1_final_panels"
PKG_SRC = PACKAGE / "source_data" / "figure1_final_panels"
PKG_CODE = PACKAGE / "code"

METHODS = {
    "Raw expression SVD": {
        "kind": "raw_svd",
        "path": VALIDATION / "raw_svd_features.npy",
        "color": "#9BA4B5",
    },
    "OmniCell native": {
        "kind": "direct",
        "path": RESULTS / "figure1_validation_native_omnicell" / "embedding.npy",
        "color": "#7B6FA6",
    },
    "OmniCell-CPT": {
        "kind": "direct",
        "path": CPT_VALIDATION_RESULT / "embedding.npy",
        "color": "#A83B3B",
    },
}

PLOT_METRICS = [
    ("Disease-state readout", "AD/control AUROC"),
    ("Disease-state readout", "AD/control balanced accuracy"),
    ("Aging-state readout", "Age Pearson r"),
    ("Cohort alignment", "neighbor entropy"),
    ("Cohort alignment", "normalized iLISI"),
    ("Cohort alignment", "same-label neighbor rate"),
]

PALETTE = {
    "ink": "#1F2933",
    "muted": "#667085",
    "grid": "#D7DEE8",
    "soft": "#F6F8FB",
}

SCORE_MIN = 0.05
SCORE_MAX = 0.95
BASELINE_HIGHER = 0.25
BASELINE_LOWER = 0.75

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 7,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": False,
        "agg.path.chunksize": 20000,
    }
)


def sem(values: list[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(len(arr)))


def save_panel(fig: plt.Figure, stem: Path, dpi: int = 900) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def split_by_group(groups: np.ndarray, y: np.ndarray | None, n_splits: int = 5):
    groups = pd.Series(groups).fillna("unknown").astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 3:
        return GroupKFold(n_splits=min(n_splits, len(unique_groups))).split(np.zeros(len(groups)), y, groups)
    if y is not None and len(np.unique(y)) > 1:
        codes = pd.factorize(y)[0]
        min_class = np.bincount(codes).min()
        return StratifiedKFold(n_splits=max(2, min(n_splits, int(min_class))), shuffle=True, random_state=13).split(
            np.zeros(len(y)), y
        )
    return GroupKFold(n_splits=2).split(np.zeros(len(groups)), y, groups)


def load_method(name: str, spec: dict[str, object], meta: pd.DataFrame) -> np.ndarray:
    path = Path(spec["path"])
    if not path.exists():
        raise FileNotFoundError(f"{name}: missing {path}")
    if spec["kind"] == "raw_svd":
        return np.array(np.load(path), dtype=np.float32, copy=True)
    arr = np.load(path, mmap_mode="r")
    if spec["kind"] == "direct":
        if arr.shape[0] != len(meta):
            raise ValueError(f"{name}: expected {len(meta)} rows, found {arr.shape[0]}")
        return np.array(arr, dtype=np.float32, copy=True)
    rows = meta["_embedding_row"].to_numpy(dtype=np.int64)
    return np.array(arr[rows], dtype=np.float32, copy=True)


def scaled_low_dim(x: np.ndarray, n_components: int = 60) -> np.ndarray:
    x = np.array(x, dtype=np.float32, copy=True)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = StandardScaler().fit_transform(x).astype(np.float32)
    if x.shape[1] > n_components:
        x = PCA(n_components=n_components, svd_solver="randomized", random_state=17).fit_transform(x).astype(np.float32)
        x = StandardScaler().fit_transform(x).astype(np.float32)
    return x


def disease_probe(method: str, x: np.ndarray, meta: pd.DataFrame) -> list[dict[str, object]]:
    labels = meta["condition_inferred"].fillna("Unknown").astype(str)
    keep = labels.isin(["AD", "Control"])
    y = labels[keep].to_numpy()
    x_use = x[keep.to_numpy()]
    groups = meta.loc[keep, "sample_id"].astype(str).to_numpy()
    if len(np.unique(y)) < 2:
        return []

    scores = {"AD/control balanced accuracy": [], "AD/control macro F1": [], "AD/control MCC": [], "AD/control AUROC": []}
    model = make_pipeline(StandardScaler(), RidgeClassifier(class_weight="balanced"))
    for train, test in split_by_group(groups, y, n_splits=5):
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        model.fit(x_use[train], y[train])
        pred = model.predict(x_use[test])
        scores["AD/control balanced accuracy"].append(balanced_accuracy_score(y[test], pred))
        scores["AD/control macro F1"].append(f1_score(y[test], pred, average="macro"))
        scores["AD/control MCC"].append(matthews_corrcoef(y[test], pred))
        dec = model.decision_function(x_use[test])
        classes = model.named_steps["ridgeclassifier"].classes_
        if len(classes) == 2:
            pos = dec if classes[1] == "AD" else -dec
            scores["AD/control AUROC"].append(roc_auc_score((y[test] == "AD").astype(int), pos))

    rows = []
    for metric, values in scores.items():
        values = [v for v in values if np.isfinite(v)]
        if values:
            rows.append(
                {
                    "method": method,
                    "domain": "Disease-state readout",
                    "metric": metric,
                    "value": float(np.mean(values)),
                    "sem": sem(values),
                    "n_replicates": len(values),
                    "n_obs": int(len(y)),
                    "direction": "higher",
                }
            )
    return rows


def age_probe(method: str, x: np.ndarray, meta: pd.DataFrame) -> list[dict[str, object]]:
    age = pd.to_numeric(meta["age_years"], errors="coerce")
    keep = age.notna()
    if keep.sum() < 500:
        return []
    y = age[keep].to_numpy(dtype=np.float32)
    x_use = x[keep.to_numpy()]
    groups = meta.loc[keep, "sample_id"].astype(str).to_numpy()
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    scores = {"Age R2": [], "Age MAE": [], "Age Pearson r": [], "Age Spearman r": []}
    for train, test in split_by_group(groups, y, n_splits=5):
        model.fit(x_use[train], y[train])
        pred = model.predict(x_use[test])
        scores["Age R2"].append(r2_score(y[test], pred))
        scores["Age MAE"].append(mean_absolute_error(y[test], pred))
        if len(np.unique(y[test])) > 2 and np.std(pred) > 0:
            scores["Age Pearson r"].append(float(stats.pearsonr(y[test], pred).statistic))
            scores["Age Spearman r"].append(float(stats.spearmanr(y[test], pred).statistic))

    rows = []
    for metric, values in scores.items():
        values = [v for v in values if np.isfinite(v)]
        if values:
            rows.append(
                {
                    "method": method,
                    "domain": "Aging-state readout",
                    "metric": metric,
                    "value": float(np.mean(values)),
                    "sem": sem(values),
                    "n_replicates": len(values),
                    "n_obs": int(keep.sum()),
                    "direction": "lower" if metric == "Age MAE" else "higher",
                }
            )
    return rows


def neighborhood_metrics(method: str, x: np.ndarray, meta: pd.DataFrame, k: int = 30) -> list[dict[str, object]]:
    x_use = scaled_low_dim(x)
    idx = NearestNeighbors(n_neighbors=min(k + 1, len(x_use)), metric="cosine", n_jobs=-1).fit(x_use).kneighbors(
        x_use, return_distance=False
    )[:, 1:]
    rows: list[dict[str, object]] = []

    def label_array(col: str) -> np.ndarray:
        return meta[col].fillna("Unknown").astype(str).to_numpy()

    def add_mixing(col: str, domain: str) -> None:
        labels = label_array(col)
        classes = np.array(sorted(pd.unique(labels)))
        mapper = {c: i for i, c in enumerate(classes)}
        n_classes = len(classes)
        if n_classes < 2:
            return
        entropy, ilisi, same = [], [], []
        for i, neigh in enumerate(idx):
            counts = np.zeros(n_classes, dtype=float)
            for lab in labels[neigh]:
                counts[mapper[lab]] += 1.0
            probs = counts / max(1.0, counts.sum())
            nz = probs[probs > 0]
            entropy.append(float(-np.sum(nz * np.log(nz)) / np.log(n_classes)))
            ilisi.append(float((1.0 / np.sum(probs**2)) / n_classes))
            same.append(float(np.mean(labels[neigh] == labels[i])))
        sil = silhouette_score(x_use, labels, metric="cosine", sample_size=min(8000, len(labels)), random_state=19)
        rows.extend(
            [
                {
                    "method": method,
                    "domain": domain,
                    "metric": "neighbor entropy",
                    "value": float(np.mean(entropy)),
                    "sem": 0.0,
                    "n_replicates": 1,
                    "n_obs": int(len(labels)),
                    "direction": "higher",
                },
                {
                    "method": method,
                    "domain": domain,
                    "metric": "normalized iLISI",
                    "value": float(np.mean(ilisi)),
                    "sem": 0.0,
                    "n_replicates": 1,
                    "n_obs": int(len(labels)),
                    "direction": "higher",
                },
                {
                    "method": method,
                    "domain": domain,
                    "metric": "same-label neighbor rate",
                    "value": float(np.mean(same)),
                    "sem": 0.0,
                    "n_replicates": 1,
                    "n_obs": int(len(labels)),
                    "direction": "lower",
                },
                {
                    "method": method,
                    "domain": domain,
                    "metric": "silhouette",
                    "value": float(sil),
                    "sem": 0.0,
                    "n_replicates": 1,
                    "n_obs": int(len(labels)),
                    "direction": "lower",
                },
            ]
        )

    def add_biological_conservation(col: str = "vascular_class") -> None:
        labels = label_array(col)
        if len(np.unique(labels)) < 2:
            return
        purity = [float(np.mean(labels[neigh] == labels[i])) for i, neigh in enumerate(idx)]
        sil = silhouette_score(x_use, labels, metric="cosine", sample_size=min(8000, len(labels)), random_state=23)
        rows.extend(
            [
                {
                    "method": method,
                    "domain": "Biological-state conservation",
                    "metric": "vascular-state kNN purity",
                    "value": float(np.mean(purity)),
                    "sem": 0.0,
                    "n_replicates": 1,
                    "n_obs": int(len(labels)),
                    "direction": "higher",
                },
                {
                    "method": method,
                    "domain": "Biological-state conservation",
                    "metric": "vascular-state silhouette",
                    "value": float(sil),
                    "sem": 0.0,
                    "n_replicates": 1,
                    "n_obs": int(len(labels)),
                    "direction": "higher",
                },
            ]
        )

    add_biological_conservation()
    add_mixing("cohort", "Cohort alignment")
    add_mixing("modality", "Modality alignment")
    return rows


def compute_metrics() -> pd.DataFrame:
    meta = pd.read_csv(VALIDATION / "validation_cells.csv", low_memory=False)
    rows: list[dict[str, object]] = []
    for method, spec in METHODS.items():
        print(f"[fig1 metrics] loading {method}", flush=True)
        x = load_method(method, spec, meta)
        print(f"[fig1 metrics] {method}: {x.shape}", flush=True)
        rows.extend(disease_probe(method, x, meta))
        rows.extend(age_probe(method, x, meta))
        rows.extend(neighborhood_metrics(method, x, meta))
        del x
    out = pd.DataFrame(rows)
    SRC.mkdir(parents=True, exist_ok=True)
    out.to_csv(SRC / "fig1c_representation_metrics_final_source.csv", index=False)
    return out


def metric_direction(metrics: pd.DataFrame, domain: str, metric: str) -> str:
    sub = metrics[(metrics["domain"].eq(domain)) & (metrics["metric"].eq(metric))]
    if sub.empty:
        return "higher"
    return str(sub["direction"].iloc[0])


def clean_cpt_method_labels(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    tuned = metrics["method"].astype(str).eq("OmniCell-CPT fine-tuned")
    frozen = metrics["method"].astype(str).eq("OmniCell-CPT")
    metrics.loc[tuned, "method"] = "OmniCell-CPT"
    if tuned.any() and frozen.any():
        metrics = metrics.loc[~(frozen)].copy()
    return metrics.reset_index(drop=True)


def selected_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = clean_cpt_method_labels(metrics)
    selected = metrics.merge(pd.DataFrame(PLOT_METRICS, columns=["domain", "metric"]), on=["domain", "metric"], how="inner")
    selected = selected[selected["method"].isin(METHODS)].copy()
    selected["method"] = pd.Categorical(selected["method"], categories=list(METHODS), ordered=True)
    return selected.sort_values(["domain", "metric", "method"]).reset_index(drop=True)


def add_scaled_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    scaled_values: list[float] = []
    scaled_sem: list[float] = []
    for (domain, metric), sub in out.groupby(["domain", "metric"], sort=False):
        idx = sub.index
        vals = sub["value"].astype(float)
        direction = str(sub["direction"].iloc[0])
        lo = float(vals.min())
        hi = float(vals.max())
        span = hi - lo
        sem_raw = sub["sem"].fillna(0).astype(float)
        if abs(span) < 1e-12:
            scaled = pd.Series(np.full(len(sub), 0.50), index=idx)
            local_sem = pd.Series(np.zeros(len(sub)), index=idx)
        elif direction == "lower":
            scaled = SCORE_MIN + 0.70 * (vals - lo) / span
            scaled = scaled.clip(lower=SCORE_MIN, upper=BASELINE_LOWER)
            local_sem = (0.70 * sem_raw / span).clip(lower=0.0, upper=0.18)
            local_sem.index = idx
        else:
            scaled = BASELINE_HIGHER + 0.70 * (vals - lo) / span
            scaled = scaled.clip(lower=BASELINE_HIGHER, upper=SCORE_MAX)
            local_sem = (0.70 * sem_raw / span).clip(lower=0.0, upper=0.18)
            local_sem.index = idx
        scaled_values.extend(scaled.loc[idx].tolist())
        scaled_sem.extend(local_sem.loc[idx].tolist())
    out["scaled_score"] = scaled_values
    out["scaled_sem"] = scaled_sem
    return out


def draw(metrics: pd.DataFrame) -> None:
    method_order = list(METHODS)
    metrics = add_scaled_scores(selected_metrics(metrics))
    fig = plt.figure(figsize=(7.65, 4.85))
    gs = fig.add_gridspec(2, 3, left=0.15, right=0.985, top=0.78, bottom=0.12, wspace=0.42, hspace=0.68)

    for i, (domain, metric) in enumerate(PLOT_METRICS):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        sub = metrics[(metrics["domain"].eq(domain)) & (metrics["metric"].eq(metric))].copy()
        sub["method"] = pd.Categorical(sub["method"], categories=method_order, ordered=True)
        sub = sub.sort_values("method")
        direction = metric_direction(metrics, domain, metric)
        y = np.arange(len(method_order))
        values = []
        for j, method in enumerate(method_order):
            row = sub[sub["method"].eq(method)]
            if row.empty:
                values.append(np.nan)
                continue
            value = float(row["scaled_score"].iloc[0])
            err = float(row["scaled_sem"].iloc[0])
            raw_value = float(row["value"].iloc[0])
            values.append(value)
            ax.barh(
                j,
                value,
                height=0.58,
                color=METHODS[method]["color"],
                alpha=0.92,
                xerr=err if err > 0 else None,
                error_kw=dict(ecolor=PALETTE["ink"], lw=0.75, capsize=1.8, capthick=0.75),
            )
            ax.text(
                1.012,
                j,
                f"{value:.2f} ({raw_value:.2f})",
                ha="left",
                va="center",
                fontsize=5.15,
                color=PALETTE["ink"],
                clip_on=False,
            )

        ax.set_yticks(y)
        ax.set_yticklabels(method_order if i % 3 == 0 else [])
        ax.invert_yaxis()
        title = (
            metric.replace("AD/control ", "")
            .replace("same-label neighbor rate", "same-cohort neighbor rate")
            .replace("neighbor entropy", "neighbor entropy")
        )
        ax.set_title(f"{domain}\n{title}", loc="left", fontsize=7.0, fontweight="bold", pad=7)
        finite = np.asarray([v for v in values if np.isfinite(v)])
        if len(finite):
            ax.set_xlim(0, 1.08)
            ax.set_xticks([0, 0.25, 0.50, 0.75, 1.00], ["0", "0.25", "0.50", "0.75", "1.0"])
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=5.6, length=2)
        ax.tick_params(axis="y", labelsize=6.1, length=0)
        ax.text(
            0.985,
            1.015,
            "lower better" if direction == "lower" else "higher better",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=4.85,
            color=PALETTE["muted"],
        )

    fig.text(0.025, 0.965, "Figure 1 representation scorecard", fontsize=11.2, fontweight="bold", color=PALETTE["ink"], ha="left", va="top")
    fig.text(
        0.025,
        0.905,
        "Scaled metric values from an independent 50,000-anchor audit; disease and age readouts are higher-is-better, while cohort-neighbor separation remains lower-is-better. Raw values are shown in parentheses.",
        fontsize=6.15,
        color=PALETTE["muted"],
        ha="left",
        va="top",
    )
    save_panel(fig, FIG1 / "fig1c_representation_metrics_final")

    for suffix in [".pdf", ".svg", ".png", ".tiff"]:
        src = FIG1 / f"fig1c_representation_metrics_final{suffix}"
        shutil.copy2(src, FIG1 / f"fig1c_omnicell_finetuned_performance_audit{suffix}")


def sync_outputs() -> None:
    PKG_PLOTS.mkdir(parents=True, exist_ok=True)
    PKG_SRC.mkdir(parents=True, exist_ok=True)
    PKG_CODE.mkdir(parents=True, exist_ok=True)
    stems = ["fig1c_representation_metrics_final", "fig1c_omnicell_finetuned_performance_audit"]
    for stem in stems:
        for suffix in [".pdf", ".svg", ".png", ".tiff"]:
            src = FIG1 / f"{stem}{suffix}"
            if src.exists():
                shutil.copy2(src, PKG_PLOTS / src.name)
    for name in [
        "fig1c_representation_metrics_final_source.csv",
        "fig1c_representation_metrics_final_selected_source.csv",
        "fig1c_representation_metrics_final_contract.json",
        "fig1c_finetune_diagnostic_summary.json",
        "fig1c_finetune_diagnostic_summary.txt",
    ]:
        src = SRC / name
        if src.exists():
            shutil.copy2(src, PKG_SRC / src.name)
    shutil.copy2(Path(__file__), PKG_CODE / Path(__file__).name)


def write_contract(metrics: pd.DataFrame) -> None:
    selected = add_scaled_scores(selected_metrics(metrics))
    selected.to_csv(SRC / "fig1c_representation_metrics_final_selected_source.csv", index=False)
    best = []
    for (domain, metric), sub in selected.groupby(["domain", "metric"], sort=False):
        direction = str(sub["direction"].iloc[0])
        idx = sub["value"].idxmin() if direction == "lower" else sub["value"].idxmax()
        row = sub.loc[idx]
        best.append(
            {
                "domain": domain,
                "metric": metric,
                "direction": direction,
                "best_method": row["method"],
                "best_value": float(row["value"]),
            }
        )
    payload = {
        "core_conclusion": "Figure 1C evaluates representation-level behavior only; Figure 2 task benchmarks are intentionally excluded.",
        "n_validation_anchors": int(pd.read_csv(VALIDATION / "validation_cells.csv", usecols=["sample_id"]).shape[0]),
        "methods": list(METHODS),
        "selected_metrics": best,
        "scaling": "Within each higher-is-better metric, the worst displayed method maps to 0.25 and the best displayed method maps to 0.95. Within each lower-is-better metric, the best displayed method maps to 0.05 and the worst displayed method maps to 0.75, so lower scaled values remain better. No panel uses 1.00 as the cap.",
        "source_data": str(SRC / "fig1c_representation_metrics_final_source.csv"),
        "overwritten_legacy_panel": "fig1c_omnicell_finetuned_performance_audit.* now points to representation metrics, not Figure 2 metrics.",
    }
    (SRC / "fig1c_representation_metrics_final_contract.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def summarize_dict_counts(obj: object) -> dict[str, int]:
    if not isinstance(obj, dict):
        return {}
    out: dict[str, int] = {}
    for k, v in obj.items():
        try:
            out[str(k)] = int(v)
        except Exception:
            continue
    return out


def write_finetune_diagnostics(metrics: pd.DataFrame) -> None:
    selected = add_scaled_scores(selected_metrics(metrics))
    run_config = read_json_if_exists(MULTITASK_FT_RESULT / "run_config.json")
    embed_config = read_json_if_exists(CPT_VALIDATION_RESULT / "run_config.json")
    clean_v3 = read_json_if_exists(VASCULAR_FT_RESULT / "clean_vascular_recluster_v3" / "clean_vascular_recluster_v3_summary.json")
    review_v4 = read_json_if_exists(VASCULAR_FT_RESULT / "clean_vascular_cluster_review_v4" / "clean_vascular_cluster_review_v4_summary.json")
    source_v6 = read_json_if_exists(VASCULAR_FT_RESULT / "vascular_source_diagnostic_v6" / "vascular_source_diagnostic_v6_summary.json")

    validation = pd.read_csv(VALIDATION / "validation_cells.csv", low_memory=False)
    condition_counts = summarize_dict_counts(validation.get("condition", pd.Series(dtype=str)).fillna("NA").value_counts().to_dict())
    modality_counts = summarize_dict_counts(validation.get("modality", pd.Series(dtype=str)).fillna("NA").value_counts().to_dict())

    metric_summary = []
    for (domain, metric), sub in selected.groupby(["domain", "metric"], sort=False):
        rows = []
        for _, row in sub.sort_values("method").iterrows():
            rows.append(
                {
                    "method": str(row["method"]),
                    "raw_value": float(row["value"]),
                    "scaled_score": float(row["scaled_score"]),
                    "direction": str(row["direction"]),
                }
            )
        metric_summary.append({"domain": domain, "metric": metric, "methods": rows})

    single_source = {}
    if isinstance(source_v6.get("single_cell"), dict):
        single_source = source_v6["single_cell"]  # type: ignore[assignment]

    clean_single = {}
    clean_spatial = {}
    if isinstance(review_v4.get("single_cell"), dict):
        clean_single = review_v4["single_cell"]  # type: ignore[assignment]
    if isinstance(review_v4.get("spatial"), dict):
        clean_spatial = review_v4["spatial"]  # type: ignore[assignment]

    diagnostic = {
        "figure_change": {
            "displayed_methods": list(METHODS),
            "cpt_definition": "The displayed OmniCell-CPT row uses the fine-tuned CPT embedding; the separate 'OmniCell-CPT fine-tuned' row is intentionally removed.",
            "score_scaling": "For higher-is-better metrics, the worst displayed method maps to 0.25 and the best displayed method maps to 0.95. For lower-is-better metrics, the best displayed method maps to 0.05 and the worst displayed method maps to 0.75, so lower scaled values remain better. Raw metric values are retained in parentheses and source data.",
        },
        "fine_tune_input": {
            "embedding_path": str(METHODS["OmniCell-CPT"]["path"]),
            "checkpoint": str(MULTITASK_FT_RESULT / "backbone"),
            "objective": run_config.get("objective"),
            "loss_weights": run_config.get("loss_weights"),
            "pooling": embed_config.get("pooling"),
            "hvg_top": embed_config.get("hvg_top"),
            "hvg_tokens_available": embed_config.get("hvg_tokens_available"),
            "n_cells": embed_config.get("n_rows"),
        },
        "validation_set": {
            "n_validation_anchors": int(validation.shape[0]),
            "condition_counts": condition_counts,
            "modality_counts": modality_counts,
        },
        "current_scores": metric_summary,
        "why_finetune_gain_is_limited": [
            "The CPT row now uses the multi-task fine-tuned checkpoint, but the run is intentionally conservative: one epoch over selected cohorts with reconstruction retained as the dominant loss.",
            "Disease, age and cell-class labels are heterogeneous across cohorts; label coverage is therefore lower than the full 50,000-anchor audit and split-to-split uncertainty remains visible.",
            "The validation disease probe is label-limited and imbalanced; AD/control non-null anchors are far fewer than the full 50,000-anchor audit, so gains have large split-to-split uncertainty.",
            "The alignment losses deliberately trade off cohort/modality removal against biological-state preservation, so not every readout is expected to improve monotonically.",
            "Further gains likely require longer training plus balanced source sampling rather than stronger display scaling.",
        ],
        "source_effect_diagnostics": {
            "single_cell_nmi_module_vs_sample_id": single_source.get("nmi_module_vs_sample_id"),
            "single_cell_nmi_module_vs_condition": single_source.get("nmi_module_vs_condition"),
            "single_cell_nmi_module_vs_bio_macro": single_source.get("nmi_module_vs_bio_macro"),
        },
        "clean_vascular_counts": {
            "single_cell_macro_counts": summarize_dict_counts(clean_single.get("macro_counts")) if clean_single else {},
            "spatial_macro_counts": summarize_dict_counts(clean_spatial.get("macro_counts")) if clean_spatial else {},
            "recluster_v3": clean_v3,
        },
        "recommended_next_steps": [
            "For Figure 1, keep this direction-aware scaled representation scorecard and avoid Figure 2 annotation/deconvolution metrics.",
            "For true representation improvement, continue the multi-task fine-tune with broader balanced sampling and then regenerate this same validation embedding.",
            "For vascular biology, keep separate single-cell and spatial subtype workflows, remove contaminants/doublets, and train broad-to-fine supervision before spatial deconvolution supervision.",
            "Use source-balanced splits and explicit batch/source residualization during UMAP visualization, otherwise the manifold can show sample/condition arcs instead of biological subtype axes.",
        ],
    }
    (SRC / "fig1c_finetune_diagnostic_summary.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")

    lines = [
        "Figure 1C fine-tune diagnostic",
        "",
        "Display fix:",
        "- OmniCell-CPT now represents the fine-tuned CPT embedding; the separate CPT fine-tuned row is removed.",
        "- Scaled values are direction-aware min-max scores: higher-is-better panels increase toward 0.95, while lower-is-better panels decrease toward 0.05; raw values remain in parentheses.",
        "",
        "Why the fine-tuning gain is limited:",
    ]
    lines.extend([f"- {x}" for x in diagnostic["why_finetune_gain_is_limited"]])
    lines.extend(["", "Recommended next steps:"])
    lines.extend([f"- {x}" for x in diagnostic["recommended_next_steps"]])
    (SRC / "fig1c_finetune_diagnostic_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIG1.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    cached = SRC / "fig1c_representation_metrics_final_source.csv"
    metrics = compute_metrics()
    metrics = clean_cpt_method_labels(metrics)
    metrics.to_csv(cached, index=False)
    draw(metrics)
    write_contract(metrics)
    write_finetune_diagnostics(metrics)
    sync_outputs()
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

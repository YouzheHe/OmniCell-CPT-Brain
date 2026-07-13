#!/usr/bin/env python
"""Figure 3 vascular subtype marker audit.

This is the manual-review layer requested for Figure 3. It reruns
FindAllMarkers-style marker discovery on the clean v11 vascular clusters,
exports top10 markers per cluster, flags non-vascular/contaminant markers, and
suggests cluster pairs that may need manual merging.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import scanpy as sc

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_figure3_vascular_findallmarkers_v13 as v13  # noqa: E402


DEFAULT_RESULT_DIR = f"{v13.BASE}/figure3_vascular_marker_audit_v19/single_cell"
DEFAULT_FIGURE_DIR = f"{v13.FIG_BASE}/figure3_vascular_marker_audit_v19/single_cell"

VASCULAR_MODULE_TO_CLASS = {
    "Endothelial_core": "Endothelial",
    "BBB_capillary": "Endothelial",
    "Arterial_EC": "Endothelial",
    "Venous_activated_EC": "Endothelial",
    "Pericyte": "Pericyte",
    "SMC": "SMC",
    "Fibroblast_VLMC": "Fibroblast_VLMC",
}

OTHER_MODULES = {
    "Astrocyte": ["GFAP", "AQP4", "ALDH1L1", "SLC1A2", "SLC1A3", "GJA1"],
    "Oligodendrocyte": ["PLP1", "MBP", "MOG", "MOBP", "MAG", "CLDN11"],
    "OPC": ["PDGFRA", "VCAN", "CSPG4", "OLIG1", "OLIG2"],
    "Microglia_immune": ["P2RY12", "CX3CR1", "AIF1", "C1QA", "C1QB", "TYROBP", "PTPRC"],
    "Neuron": ["RBFOX3", "SNAP25", "SYT1", "SLC17A7", "GAD1", "GAD2"],
    "RBC_blood": ["HBB", "HBA1", "HBA2", "ALAS2"],
    "Ependymal_choroid": ["FOXJ1", "TTR", "AQP1", "KRT18", "KRT8"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad", default=v13.DEFAULT_H5AD)
    p.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    p.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    p.add_argument("--matrix-key", default="expanded_marker_log1p")
    p.add_argument("--gene-key", default="expanded_marker_genes")
    p.add_argument("--cluster-key", default="v11_clean_cluster")
    p.add_argument("--class-key", default="v11_marker_class")
    p.add_argument("--method", default="wilcoxon", choices=["wilcoxon", "t-test_overestim_var", "t-test"])
    p.add_argument("--top-n", type=int, default=200)
    p.add_argument("--audit-top-n", type=int, default=10)
    p.add_argument("--dotplot-top-per-cluster", type=int, default=10)
    p.add_argument("--min-pct-in", type=float, default=0.08)
    p.add_argument("--min-pct-delta", type=float, default=0.03)
    p.add_argument("--merge-jaccard-threshold", type=float, default=0.35)
    p.add_argument("--contaminant-min-hits", type=int, default=2)
    p.add_argument("--min-cluster-cells", type=int, default=80)
    p.add_argument("--dpi", type=int, default=800)
    return p.parse_args()


def build_gene_module_index() -> dict[str, list[str]]:
    modules: dict[str, list[str]] = {}
    for module, genes in {**v13.MODULES, **OTHER_MODULES}.items():
        for gene in genes:
            modules.setdefault(gene.upper(), []).append(module)
    return modules


def top_per_group(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    primary = df[df["passes_filter"] & ~df["nuisance_gene"]].copy()
    fallback = df[~df["nuisance_gene"]].copy()
    out = []
    for group in sorted(df["group"].astype(str).unique(), key=lambda x: (len(x), x)):
        sub = primary[primary["group"].astype(str).eq(group)].sort_values("rank")
        if len(sub) < n:
            seen = set(sub["gene"].astype(str))
            extra = fallback[fallback["group"].astype(str).eq(group) & ~fallback["gene"].astype(str).isin(seen)].sort_values("rank")
            sub = pd.concat([sub, extra], ignore_index=True)
        out.append(sub.head(n))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def module_hits(genes: list[str], gene_to_modules: dict[str, list[str]]) -> tuple[Counter, Counter]:
    vascular: Counter = Counter()
    other: Counter = Counter()
    for gene in genes:
        for module in gene_to_modules.get(gene.upper(), []):
            if module in VASCULAR_MODULE_TO_CLASS:
                vascular[module] += 1
            elif module in OTHER_MODULES or module == "Contaminant_glia_neuron":
                other[module] += 1
    return vascular, other


def format_counts(counts: Counter) -> str:
    return "; ".join(f"{key}:{value}" for key, value in counts.most_common())


def dominant_class_from_modules(counts: Counter) -> tuple[str, int, str]:
    class_counts: Counter = Counter()
    for module, count in counts.items():
        klass = VASCULAR_MODULE_TO_CLASS.get(module)
        if klass:
            class_counts[klass] += count
    if not class_counts:
        return "", 0, ""
    klass, score = class_counts.most_common(1)[0]
    best_modules = [m for m, c in counts.items() if VASCULAR_MODULE_TO_CLASS.get(m) == klass and c > 0]
    return klass, int(score), ",".join(best_modules)


def cluster_mode(series: pd.Series) -> str:
    value = series.astype(str).mode()
    return value.iloc[0] if len(value) else ""


def pairwise_marker_overlap(audit: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    rows = []
    best: dict[str, dict[str, object]] = {}
    marker_sets = {
        row["cluster"]: set(str(row["top10_markers"]).split(", ")) - {""}
        for _, row in audit.iterrows()
    }
    for i, row_a in audit.iterrows():
        for j, row_b in audit.iterrows():
            if j <= i:
                continue
            a = row_a["cluster"]
            b = row_b["cluster"]
            set_a = marker_sets.get(a, set())
            set_b = marker_sets.get(b, set())
            if not set_a or not set_b:
                continue
            jaccard = len(set_a & set_b) / max(len(set_a | set_b), 1)
            same_existing = row_a["existing_broad_class"] == row_b["existing_broad_class"]
            same_dominant = row_a["dominant_marker_class"] and row_a["dominant_marker_class"] == row_b["dominant_marker_class"]
            rec = {
                "cluster_a": a,
                "cluster_b": b,
                "marker_jaccard": float(jaccard),
                "shared_markers": ", ".join(sorted(set_a & set_b)),
                "same_existing_broad_class": bool(same_existing),
                "same_dominant_marker_class": bool(same_dominant),
                "suggest_merge": bool(jaccard >= threshold and (same_existing or same_dominant)),
            }
            rows.append(rec)
            if rec["suggest_merge"]:
                for cluster, other in [(a, b), (b, a)]:
                    if cluster not in best or jaccard > float(best[cluster]["marker_jaccard"]):
                        best[cluster] = {
                            "merge_candidate": other,
                            "marker_jaccard": float(jaccard),
                            "shared_markers": rec["shared_markers"],
                        }
    return pd.DataFrame(rows).sort_values("marker_jaccard", ascending=False), best


def make_audit_table(marker: sc.AnnData, cluster_markers: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_to_modules = build_gene_module_index()
    top10 = top_per_group(cluster_markers, args.audit_top_n)
    obs = marker.obs.copy()
    rows = []
    for group, sub in obs.groupby(args.cluster_key, observed=False):
        group = str(group)
        top = top10[top10["group"].astype(str).eq(group)].sort_values("rank")
        genes = top["gene"].astype(str).head(args.audit_top_n).tolist()
        vascular_hits, other_hits = module_hits(genes, gene_to_modules)
        dominant_class, dominant_score, dominant_modules = dominant_class_from_modules(vascular_hits)
        existing_class = cluster_mode(sub[args.class_key]) if args.class_key in sub else ""
        contaminant_genes = [
            gene for gene in genes
            if any(module in OTHER_MODULES or module == "Contaminant_glia_neuron" for module in gene_to_modules.get(gene.upper(), []))
        ]
        if len(sub) < args.min_cluster_cells:
            recommendation = "manual_review_low_cell_count"
        elif len(contaminant_genes) >= args.contaminant_min_hits and sum(other_hits.values()) >= dominant_score:
            recommendation = "mark_as_other_or_contaminant"
        elif dominant_class and existing_class and dominant_class != existing_class:
            recommendation = "manual_review_broad_class_mismatch"
        elif not dominant_class:
            recommendation = "manual_review_weak_marker_specificity"
        else:
            recommendation = "keep_pending_manual_review"
        rows.append(
            {
                "cluster": group,
                "existing_broad_class": existing_class,
                "dominant_marker_class": dominant_class,
                "dominant_marker_score": dominant_score,
                "dominant_marker_modules": dominant_modules,
                "n_cells": int(len(sub)),
                "top10_markers": ", ".join(genes),
                "vascular_module_hits": format_counts(vascular_hits),
                "other_contaminant_hits": format_counts(other_hits),
                "other_contaminant_genes": ", ".join(contaminant_genes),
                "sample_top3": "; ".join(sub["sample_id"].astype(str).value_counts().head(3).index.tolist()) if "sample_id" in sub else "",
                "condition_composition": "; ".join(f"{k}:{v}" for k, v in sub.get("condition_inferred", pd.Series(["Unknown"] * len(sub), index=sub.index)).astype(str).value_counts().head(4).items()),
                "initial_recommendation": recommendation,
            }
        )
    audit = pd.DataFrame(rows).sort_values(["existing_broad_class", "cluster"])
    pairwise, best = pairwise_marker_overlap(audit, args.merge_jaccard_threshold)
    audit["merge_candidate"] = audit["cluster"].map(lambda x: best.get(x, {}).get("merge_candidate", ""))
    audit["merge_marker_jaccard"] = audit["cluster"].map(lambda x: best.get(x, {}).get("marker_jaccard", np.nan))
    audit["merge_shared_markers"] = audit["cluster"].map(lambda x: best.get(x, {}).get("shared_markers", ""))
    audit["final_manual_action_hint"] = audit.apply(action_hint, axis=1)
    return audit, pairwise


def action_hint(row: pd.Series) -> str:
    if row["initial_recommendation"] == "mark_as_other_or_contaminant":
        return "check_other_markers_then_exclude_or_label_Other"
    if isinstance(row.get("merge_candidate"), str) and row["merge_candidate"]:
        return f"inspect_merge_with_{row['merge_candidate']}"
    return str(row["initial_recommendation"])


def run_markers(marker: sc.AnnData, groupby: str, args: argparse.Namespace, result_dir: Path, figure_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = v13.rank_to_table(marker, groupby, args)
    table.to_csv(result_dir / f"figure3_vascular_marker_audit_v19_findallmarkers_{groupby}_top{args.top_n}.csv", index=False)
    top10 = top_per_group(table, args.audit_top_n)
    top10.to_csv(result_dir / f"figure3_vascular_marker_audit_v19_findallmarkers_{groupby}_top{args.audit_top_n}.csv", index=False)
    genes = top10.sort_values(["group", "rank"])["gene"].astype(str).tolist()
    dot_src = v13.dotplot_source(marker, groupby, genes)
    dot_src.to_csv(result_dir / f"figure3_vascular_marker_audit_v19_findallmarkers_{groupby}_top{args.audit_top_n}_dotplot_source.csv", index=False)
    v13.plot_dotplot(
        dot_src,
        f"Figure 3 marker audit top{args.audit_top_n}: {groupby}",
        figure_dir / f"figure3_vascular_marker_audit_v19_{groupby}_top{args.audit_top_n}_dotplot",
        args.dpi,
    )
    v13.module_dotplot(marker, groupby, result_dir, figure_dir, args.dpi)
    return table, top10


def write_report(result_dir: Path, summary: dict[str, object], audit: pd.DataFrame) -> None:
    counts = audit["final_manual_action_hint"].value_counts().to_dict()
    lines = [
        "# Figure 3 Vascular Marker Audit v19",
        "",
        "This output is intended for manual cluster review before final subtype naming or spatial deconvolution.",
        "",
        "## Summary",
        "",
        f"- Cells: {summary['n_cells']}",
        f"- Genes: {summary['n_genes']}",
        f"- Cluster key: {summary['cluster_key']}",
        f"- Class key: {summary['class_key']}",
        f"- FindAllMarkers method: {summary['method']}",
        f"- Top markers for manual audit: {summary['audit_top_n']}",
        f"- Workbook: {summary['workbook']}",
        "",
        "## Manual Action Counts",
        "",
    ]
    for key, value in counts.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## High-priority Review Rows", ""])
    priority = audit[audit["final_manual_action_hint"].str.contains("other|merge|mismatch|weak", case=False, na=False)].head(30)
    if priority.empty:
        lines.append("No high-priority rows were flagged.")
    else:
        cols = ["cluster", "existing_broad_class", "dominant_marker_class", "top10_markers", "other_contaminant_genes", "merge_candidate", "final_manual_action_hint"]
        lines.append("|" + "|".join(cols) + "|")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, row in priority.iterrows():
            lines.append("|" + "|".join(str(row.get(col, "")) for col in cols) + "|")
    (result_dir / "Figure3_vascular_marker_audit_v19_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    figure_dir = Path(args.figure_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.h5ad)
    required = [args.cluster_key, args.class_key, "sample_id", "condition_inferred"]
    marker = v13.make_expression_adata(adata, args.matrix_key, args.gene_key, required)
    marker.obs[args.cluster_key] = marker.obs[args.cluster_key].astype(str).astype("category")
    marker.obs[args.class_key] = marker.obs[args.class_key].astype(str).astype("category")

    cluster_markers, cluster_top10 = run_markers(marker, args.cluster_key, args, result_dir, figure_dir)
    class_markers, class_top10 = run_markers(marker, args.class_key, args, result_dir, figure_dir)
    audit, pairwise = make_audit_table(marker, cluster_markers, args)

    audit.to_csv(result_dir / "figure3_vascular_marker_audit_v19_cluster_top10_manual_review.csv", index=False)
    pairwise.to_csv(result_dir / "figure3_vascular_marker_audit_v19_pairwise_marker_overlap.csv", index=False)

    xlsx = result_dir / "figure3_vascular_marker_audit_v19_manual_review_workbook.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        audit.to_excel(writer, sheet_name="manual_review", index=False)
        cluster_top10.to_excel(writer, sheet_name="cluster_top10_long", index=False)
        class_top10.to_excel(writer, sheet_name="class_top10_long", index=False)
        pairwise.to_excel(writer, sheet_name="pairwise_overlap", index=False)
        cluster_markers.to_excel(writer, sheet_name="cluster_all_markers", index=False)
        class_markers.to_excel(writer, sheet_name="class_all_markers", index=False)

    summary = {
        "input_h5ad": str(args.h5ad),
        "n_cells": int(marker.n_obs),
        "n_genes": int(marker.n_vars),
        "cluster_key": args.cluster_key,
        "class_key": args.class_key,
        "method": args.method,
        "top_n": args.top_n,
        "audit_top_n": args.audit_top_n,
        "cluster_marker_rows": int(len(cluster_markers)),
        "class_marker_rows": int(len(class_markers)),
        "manual_review_rows": int(len(audit)),
        "result_dir": str(result_dir),
        "figure_dir": str(figure_dir),
        "workbook": str(xlsx),
    }
    (result_dir / "figure3_vascular_marker_audit_v19_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(result_dir, summary, audit)
    print(json.dumps(summary, indent=2), flush=True)
    print("manual_review_preview", flush=True)
    print(audit.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

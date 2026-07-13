#!/usr/bin/env python
"""Supplementary statistics and source tables for Figure 4.

All outputs are generated on the server from existing BI result files.
No values are hand-edited or boosted.
"""

from __future__ import annotations
import os

import argparse
import importlib.util
import math
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DEFAULT_BI = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/BI"))
PY_ENV = "${PYTHON}"

REFERENCE_MAP = {
    "Microglia/immune": "Mathys et al., Nature 2019; human AD microglial-state literature; canonical immune/complement markers; internal selected-gene overlap.",
    "Astrocyte": "Mathys et al., Nature 2019; reactive astrocyte marker literature; canonical astrocyte markers; internal selected-gene overlap.",
    "Endothelial": "Vanlandewijck et al., Nature 2018 brain vascular atlas; canonical endothelial markers; internal selected-gene overlap.",
    "Pericyte/mural": "Vanlandewijck et al., Nature 2018 brain vascular atlas; canonical mural/pericyte markers; internal selected-gene overlap.",
    "VLMC/fibroblast": "Vanlandewijck et al., Nature 2018 vessel-associated/VLMC markers; extracellular-matrix/fibroblast marker literature; internal selected-gene overlap.",
    "Neuron": "Mathys et al., Nature 2019; canonical neuronal and AD-associated neuronal genes; internal selected-gene overlap.",
    "Immune/complement": "MSigDB Hallmark Complement and GO immune/complement categories; AD immune/microglia literature; internal selected-gene overlap.",
    "Iron/oxidative stress": "MSigDB Hallmark Reactive Oxygen Species Pathway; GO response to oxidative stress and iron-ion homeostasis; internal selected-gene overlap.",
    "Angiogenesis/vascular": "MSigDB Hallmark Angiogenesis and GO blood-vessel development categories; brain vascular marker literature; internal selected-gene overlap.",
    "Neuronal apoptosis": "MSigDB Hallmark Apoptosis and GO neuron apoptotic process categories; AD neuronal-stress literature; internal selected-gene overlap.",
    "Lipid/APOE stress": "AD APOE/lipid-processing literature and GO lipid transport/cholesterol categories; internal selected-gene overlap.",
    "ECM/VLMC remodeling": "GO extracellular matrix organization and vessel-associated fibroblast/VLMC marker literature; internal selected-gene overlap.",
}

REFERENCE_URLS = {
    "Mathys et al., Nature 2019": "https://pubmed.ncbi.nlm.nih.gov/31042697/",
    "Vanlandewijck et al., Nature 2018": "https://pubmed.ncbi.nlm.nih.gov/29443965/",
    "MSigDB Hallmark gene sets": "https://www.gsea-msigdb.org/gsea/msigdb/",
    "HALLMARK_COMPLEMENT": "https://www.gsea-msigdb.org/gsea/msigdb/cards/HALLMARK_COMPLEMENT",
    "HALLMARK_ANGIOGENESIS": "https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_ANGIOGENESIS.html",
    "HALLMARK_APOPTOSIS": "https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_APOPTOSIS.html",
    "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY": "https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY.html",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bi-root", type=Path, default=DEFAULT_BI)
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    parser.add_argument("--n-permutation", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260709)
    return parser.parse_args()


def load_figure4_module(root: Path):
    p = root / "scripts" / "make_figure4_nature_style.py"
    spec = importlib.util.spec_from_file_location("figure4_strict", p)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 160,
    })


def save_fig(fig, stem: Path, png_dpi: int = 450) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    q = np.empty(m, dtype=float)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        true_rank = m - rank + 1
        val = pvals[idx] * m / true_rank
        prev = min(prev, val)
        q[idx] = min(prev, 1.0)
    return q


def format_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def paired_auc_tests(root: Path, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = root / "figures" / "multicell_integrated_ad_model" / "source_cpt_vs_raw_predictions.csv"
    pred = pd.read_csv(pred_path)
    cpt = pred[(pred.dataset == "all44") & (pred.feature_set == "latent")][["heldout_sample_id", "heldout_label", "pred_prob"]].rename(columns={"pred_prob": "score_cpt"})
    raw = pred[(pred.dataset == "all44") & (pred.feature_set == "raw_gene_svd")][["heldout_sample_id", "heldout_label", "pred_prob"]].rename(columns={"pred_prob": "score_raw"})
    df = cpt.merge(raw, on=["heldout_sample_id", "heldout_label"], how="inner").sort_values("heldout_sample_id")
    y = df.heldout_label.to_numpy(int)
    cpt_score = df.score_cpt.to_numpy(float)
    raw_score = df.score_raw.to_numpy(float)
    obs_cpt = roc_auc_score(y, cpt_score)
    obs_raw = roc_auc_score(y, raw_score)
    obs_delta = obs_cpt - obs_raw
    rng = np.random.default_rng(seed)
    boot_delta = []
    perm_delta = []
    n = len(df)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boot_delta.append(roc_auc_score(y[idx], cpt_score[idx]) - roc_auc_score(y[idx], raw_score[idx]))
    boot_delta = np.asarray(boot_delta, dtype=float)
    rng_perm = np.random.default_rng(seed + 1000)
    for _ in range(n_boot):
        swap = rng_perm.random(n) < 0.5
        s1 = cpt_score.copy()
        s2 = raw_score.copy()
        s1[swap], s2[swap] = raw_score[swap], cpt_score[swap]
        perm_delta.append(roc_auc_score(y, s1) - roc_auc_score(y, s2))
    perm_delta = np.asarray(perm_delta, dtype=float)
    p_perm = (np.sum(np.abs(perm_delta) >= abs(obs_delta)) + 1) / (len(perm_delta) + 1)
    p_boot_one_sided = (np.sum(boot_delta <= 0) + 1) / (len(boot_delta) + 1)
    out = pd.DataFrame([
        {
            "comparison": "CPT latent vs raw gene SVD",
            "source_file": str(pred_path),
            "n_samples": int(n),
            "n_ad": int(y.sum()),
            "n_control": int((1 - y).sum()),
            "auroc_cpt_latent": float(obs_cpt),
            "auroc_raw_gene_svd": float(obs_raw),
            "delta_auroc_cpt_minus_raw": float(obs_delta),
            "paired_bootstrap_se": float(boot_delta.std(ddof=1)),
            "paired_bootstrap_ci_low": float(np.percentile(boot_delta, 2.5)),
            "paired_bootstrap_ci_high": float(np.percentile(boot_delta, 97.5)),
            "paired_permutation_p_two_sided": float(p_perm),
            "paired_bootstrap_p_delta_le_0": float(p_boot_one_sided),
            "n_resamples": int(n_boot),
            "test_definition": "Samples were paired by heldout_sample_id. Bootstrap resampled samples; permutation randomly swapped CPT/raw scores within each sample.",
        }
    ])
    dist = pd.DataFrame({"bootstrap_delta_auroc": boot_delta})
    return out, dist


def selected_frequency_and_universe(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    freq_path = root / "results" / "multicell_integrated_ad_model" / "multicell_model_selected_gene_frequency.csv"
    mat_path = root / "results" / "multicell_integrated_ad_model" / "multicell_qc36_feature_matrix.csv"
    freq = pd.read_csv(freq_path)
    freq = freq[freq.dataset == "all44"].copy()
    freq["gene"] = freq["gene"].astype(str).str.upper()
    mat = pd.read_csv(mat_path, nrows=1)
    universe_genes = []
    for col in mat.columns:
        if col.startswith("gene__") and col.endswith("__mean_cell"):
            universe_genes.append(col.split("__")[1].upper())
    universe = pd.DataFrame({"gene": sorted(set(universe_genes) | set(freq.gene))})
    universe = universe.merge(freq[["gene", "n_folds_selected"]], on="gene", how="left")
    universe["n_folds_selected"] = universe["n_folds_selected"].fillna(0).astype(int)
    universe["candidate_universe_source"] = str(mat_path)
    universe["selected_frequency_source"] = str(freq_path)
    return freq, universe


def permutation_enrichment(root: Path, gene_sets: dict[str, list[str]], set_type: str, n_perm: int, seed: int) -> pd.DataFrame:
    _, universe = selected_frequency_and_universe(root)
    genes = universe.gene.to_numpy(str)
    weights = universe.n_folds_selected.to_numpy(float)
    gene_to_weight = dict(zip(genes, weights))
    rng = np.random.default_rng(seed)
    rows = []
    for name, raw_genes in gene_sets.items():
        set_genes = sorted({g.upper() for g in raw_genes})
        overlap = [g for g in set_genes if g in gene_to_weight]
        selected_overlap = [g for g in overlap if gene_to_weight[g] > 0]
        k = len(overlap)
        obs = float(sum(gene_to_weight[g] for g in overlap))
        if k == 0:
            null = np.array([0.0])
            p = 1.0
            z = np.nan
        else:
            null = np.empty(n_perm, dtype=float)
            for i in range(n_perm):
                idx = rng.choice(len(genes), size=k, replace=False)
                null[i] = weights[idx].sum()
            p = (np.sum(null >= obs) + 1) / (n_perm + 1)
            z = (obs - null.mean()) / (null.std(ddof=1) + 1e-12)
        rows.append({
            "set_type": set_type,
            "gene_set": name,
            "observed_selected_fold_sum": obs,
            "set_size_defined": len(set_genes),
            "set_size_in_universe": k,
            "n_selected_genes_overlap": len(selected_overlap),
            "selected_genes_overlap": ";".join(sorted(selected_overlap, key=lambda g: (-gene_to_weight[g], g))),
            "null_mean_selected_fold_sum": float(null.mean()),
            "null_sd_selected_fold_sum": float(null.std(ddof=1)) if len(null) > 1 else 0.0,
            "enrichment_z": float(z) if np.isfinite(z) else np.nan,
            "empirical_p_greater": float(p),
            "n_permutations": int(n_perm),
            "universe_n_genes": int(len(genes)),
            "test_definition": "Random gene sets of identical size were sampled without replacement from all gene__*_mean_cell candidate genes; statistic is sum n_folds_selected.",
        })
    out = pd.DataFrame(rows)
    out["q_value_bh_within_set_type"] = bh_fdr(out.empirical_p_greater.to_numpy(float))
    return out.sort_values("observed_selected_fold_sum", ascending=False)


def make_gene_set_source_tables(fig4, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for set_type, panel, sets in [
        ("cell_type_marker", "Figure4D", fig4.CELL_TYPE_GENE_SETS),
        ("functional_gene_set", "Figure4E/F", fig4.FUNCTION_GENE_SETS),
    ]:
        for name, genes in sets.items():
            for gene in sorted({g.upper() for g in genes}):
                rows.append({
                    "panel": panel,
                    "set_type": set_type,
                    "gene_set_name": name,
                    "gene": gene,
                    "source_basis": REFERENCE_MAP.get(name, "Curated marker/function gene list; internal selected-gene overlap."),
                    "source_policy": "Gene set is used only for enrichment/interpretability of fold-internal model-selected genes; it is not used to create CPT latent features in A/B.",
                })
    source_table = pd.DataFrame(rows)
    refs = pd.DataFrame([
        {"reference_short": k, "url": v, "usage": "Gene-set provenance / manuscript citation support"}
        for k, v in REFERENCE_URLS.items()
    ])
    return source_table, refs


def external_validation_summary(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    gse138_probe = root / "external_validation" / "GSE138852" / "results" / "GSE138852_external_probe_summary.csv"
    if gse138_probe.exists():
        df = pd.read_csv(gse138_probe)
        for _, r in df.iterrows():
            rows.append({
                "dataset": "GSE138852",
                "external_analysis_type": "AD-informed module score transfer",
                "comparison_scope": r["stratum"],
                "method_label": "Internal AD/NVU module score",
                "n": int(r["n_cells"]),
                "n_ad": int(r["n_ad"]),
                "n_control": int(r["n_control"]),
                "metric": "AUROC_oriented",
                "value": float(r["auroc_oriented"]),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "source_file": str(gse138_probe),
                "interpretation_boundary": "External validation of AD-informed gene/module transfer; not native CPT latent validation.",
            })
    gse263_nested = root / "external_validation" / "GSE263468_CZI" / "results" / "source_GSE263468_CZI_cpt_foldselected_nested_vs_raw_auc_bootstrap.csv"
    if gse263_nested.exists():
        df = pd.read_csv(gse263_nested)
        for _, r in df.iterrows():
            rows.append({
                "dataset": "GSE263468_CZI",
                "external_analysis_type": "CPT-informed fold-selected feature transfer",
                "comparison_scope": r["feature_scope"],
                "method_label": r["method_label"],
                "n": int(r["n_donors"]),
                "n_ad": int(r["n_ad"]),
                "n_control": int(r["n_control"]),
                "metric": "AUROC",
                "value": float(r["auroc"]),
                "ci_low": float(r["auroc_ci_low"]),
                "ci_high": float(r["auroc_ci_high"]),
                "source_file": str(gse263_nested),
                "interpretation_boundary": "External comparison uses transferred AD-informed selected features; not native CPT latent validation.",
            })
    out = pd.DataFrame(rows)
    return out, out.copy()


def make_external_validation_figure(external_df: pd.DataFrame, out_dir: Path) -> None:
    if external_df.empty:
        return
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.55})
    # GSE138852 stratum AUROC
    ax = axes[0]
    g = external_df[(external_df.dataset == "GSE138852") & (external_df.metric == "AUROC_oriented")].copy()
    keep = ["all_cells", "astrocyte", "endothelial", "neuron", "oligodendrocyte", "OPC", "microglia"]
    g = g[g.comparison_scope.isin(keep)].copy()
    g["comparison_scope"] = pd.Categorical(g.comparison_scope, categories=keep, ordered=True)
    g = g.sort_values("comparison_scope")
    y = np.arange(len(g))[::-1]
    ax.barh(y, g.value, color="#7EA6B5", edgecolor="#333333", linewidth=0.45, height=0.62)
    for yy, val in zip(y, g.value):
        ax.text(min(val + 0.025, 0.98), yy, f"{val:.2f}", va="center", fontsize=6.2)
    ax.set_yticks(y)
    ax.set_yticklabels(g.comparison_scope.astype(str))
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("External AUROC")
    ax.grid(axis="x", color="#DCE2EC", lw=0.8)
    ax.set_axisbelow(True)
    ax.add_patch(mpl.patches.Rectangle((0, 1.025), 1, 0.12, transform=ax.transAxes, color="#E8E8EA", clip_on=False, zorder=-1))
    ax.text(0.5, 1.085, "GSE138852 module transfer", transform=ax.transAxes, ha="center", va="center", fontsize=7.2, fontweight="bold")
    ax.text(-0.14, 1.08, "a", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")

    # GSE263468 nested AUROC, cell-type-aware and all-cell
    ax = axes[1]
    h = external_df[(external_df.dataset == "GSE263468_CZI") & (external_df.metric == "AUROC")].copy()
    h = h[h.comparison_scope.isin(["all_cell", "cell_type_aware"])].copy()
    h["label"] = h["comparison_scope"].map({"all_cell": "All-cell", "cell_type_aware": "Cell-type aware"}) + "\n" + h["method_label"].replace({"CPT-informed fold-selected": "AD-informed selected", "Raw gene SVD": "Raw SVD"})
    order = [
        "All-cell\nRaw SVD", "All-cell\nAD-informed selected",
        "Cell-type aware\nRaw SVD", "Cell-type aware\nAD-informed selected",
    ]
    h["label"] = pd.Categorical(h["label"], categories=order, ordered=True)
    h = h.sort_values("label")
    x = np.arange(len(h))
    colors = ["#8C83B8" if "Raw" in str(lbl) else "#B64342" for lbl in h.label]
    yerr = np.vstack([h.value - h.ci_low, h.ci_high - h.value])
    ax.bar(x, h.value, yerr=yerr, color=colors, edgecolor="#333333", linewidth=0.45, capsize=2.0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v).replace("\n", "\n") for v in h.label], rotation=35, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("External AUROC")
    ax.grid(axis="y", color="#DCE2EC", lw=0.8)
    ax.set_axisbelow(True)
    ax.add_patch(mpl.patches.Rectangle((0, 1.025), 1, 0.12, transform=ax.transAxes, color="#E8E8EA", clip_on=False, zorder=-1))
    ax.text(0.5, 1.085, "GSE263468_CZI feature transfer", transform=ax.transAxes, ha="center", va="center", fontsize=7.2, fontweight="bold")
    ax.text(-0.14, 1.08, "b", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    save_fig(fig, out_dir / "ExtendedData_Figure4_external_validation")
    plt.close(fig)


def write_summary_readme(out_dir: Path, auc_test: pd.DataFrame, d_perm: pd.DataFrame, e_perm: pd.DataFrame) -> None:
    auc = auc_test.iloc[0]
    d_top = d_perm.sort_values("observed_selected_fold_sum", ascending=False).head(3)
    e_top = e_perm.sort_values("observed_selected_fold_sum", ascending=False).head(3)
    lines = [
        "# Figure 4 supplementary statistics",
        "",
        f"Output root: `{out_dir}`",
        "",
        "## A/B AUROC paired test",
        f"CPT latent AUROC = {auc.auroc_cpt_latent:.3f}; raw gene SVD AUROC = {auc.auroc_raw_gene_svd:.3f}; delta = {auc.delta_auroc_cpt_minus_raw:.3f}.",
        f"Paired bootstrap 95% CI for delta = [{auc.paired_bootstrap_ci_low:.3f}, {auc.paired_bootstrap_ci_high:.3f}]; paired permutation P = {format_p(auc.paired_permutation_p_two_sided)}.",
        "",
        "## D cell-type enrichment",
        "Permutation test samples random gene sets of identical size from all candidate gene features and compares sum(n_folds_selected).",
        d_top[["gene_set", "observed_selected_fold_sum", "empirical_p_greater", "q_value_bh_within_set_type"]].to_string(index=False),
        "",
        "## E functional enrichment",
        e_top[["gene_set", "observed_selected_fold_sum", "empirical_p_greater", "q_value_bh_within_set_type"]].to_string(index=False),
        "",
        "## External validation boundary",
        "External panels are stored separately under `extended_data_external_validation` and should be described as AD-informed selected gene/module transfer, not native CPT latent validation.",
        "",
        f"Runtime: `{PY_ENV}`",
    ]
    (out_dir / "Figure4_SUPPLEMENTARY_STATISTICS_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.bi_root
    fig4 = load_figure4_module(root)
    fig_dir = root / "figures" / "figure4_nature_style"
    stats_dir = fig_dir / "statistics"
    supp_dir = fig_dir / "supplementary_tables"
    ext_dir = fig_dir / "extended_data_external_validation"
    for d in [stats_dir, supp_dir, ext_dir]:
        d.mkdir(parents=True, exist_ok=True)

    auc_test, auc_boot = paired_auc_tests(root, args.n_bootstrap, args.seed)
    d_perm = permutation_enrichment(root, fig4.CELL_TYPE_GENE_SETS, "cell_type_marker", args.n_permutation, args.seed + 10)
    e_perm = permutation_enrichment(root, fig4.FUNCTION_GENE_SETS, "functional_gene_set", args.n_permutation, args.seed + 20)
    all_perm = pd.concat([d_perm, e_perm], ignore_index=True)
    all_perm["q_value_bh_all_tests"] = bh_fdr(all_perm.empirical_p_greater.to_numpy(float))

    gene_sets, refs = make_gene_set_source_tables(fig4, root)
    ext_summary, ext_source = external_validation_summary(root)

    auc_test.to_csv(stats_dir / "Figure4_AUROC_paired_test.csv", index=False)
    auc_boot.to_csv(stats_dir / "Figure4_AUROC_delta_bootstrap_distribution.csv", index=False)
    d_perm.to_csv(stats_dir / "Figure4D_cell_type_marker_permutation_enrichment.csv", index=False)
    e_perm.to_csv(stats_dir / "Figure4E_functional_gene_set_permutation_enrichment.csv", index=False)
    all_perm.to_csv(stats_dir / "Figure4DE_all_permutation_enrichment_with_global_fdr.csv", index=False)

    gene_sets.to_csv(supp_dir / "Supplementary_Table_gene_sets.csv", index=False)
    refs.to_csv(supp_dir / "Supplementary_Table_gene_set_reference_sources.csv", index=False)
    ext_source.to_csv(ext_dir / "ExtendedData_Figure4_external_validation_source.csv", index=False)

    # Also write a single workbook when openpyxl is available.
    try:
        with pd.ExcelWriter(supp_dir / "Figure4_supplementary_statistics_and_gene_sets.xlsx") as writer:
            auc_test.to_excel(writer, sheet_name="AUC_paired_test", index=False)
            d_perm.to_excel(writer, sheet_name="D_cell_enrichment", index=False)
            e_perm.to_excel(writer, sheet_name="E_function_enrichment", index=False)
            all_perm.to_excel(writer, sheet_name="DE_global_FDR", index=False)
            gene_sets.to_excel(writer, sheet_name="Gene_sets", index=False)
            refs.to_excel(writer, sheet_name="References", index=False)
            ext_source.to_excel(writer, sheet_name="External_validation", index=False)
    except Exception as exc:
        (supp_dir / "Figure4_xlsx_write_failed.txt").write_text(str(exc), encoding="utf-8")

    make_external_validation_figure(ext_summary, ext_dir)
    write_summary_readme(fig_dir, auc_test, d_perm, e_perm)
    print(fig_dir)
    print(stats_dir)
    print(supp_dir)
    print(ext_dir)


if __name__ == "__main__":
    main()

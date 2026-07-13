#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(r"${LOCAL_USER_HOME}\Documents\链接武超-NVU AI\standard_omnicell_vascular_manifold_preview")
OUT = BASE / "standard_omnicell_marker_annotation_cleanup"

ARTIFACT_EXACT = {
    "MALAT1",
    "FTX",
    "XIST",
    "NEAT1",
    "TMSB4X",
    "B2M",
    "ACTB",
    "EEF1A1",
    "H3-3B",
    "MT-RNR1",
    "MT-RNR2",
}
ARTIFACT_PREFIXES = ("MT-", "RPL", "RPS")

REFERENCE = {
    "Endothelial": [
        "CLDN5",
        "PECAM1",
        "VWF",
        "CDH5",
        "FLT1",
        "KDR",
        "TIE1",
        "ESAM",
        "ERG",
        "CD34",
        "ICAM2",
        "NOS3",
        "THBD",
        "SELE",
        "TJP1",
        "EPAS1",
        "SPARCL1",
        "RAMP2",
        "IFI27",
        "PLVAP",
        "ACKR1",
        "GJA5",
        "SLCO4A1",
        "EGFL7",
        "APOLD1",
        "PODXL",
        "ABCG2",
        "HEG1",
    ],
    "Pericyte": [
        "PDGFRB",
        "RGS5",
        "KCNJ8",
        "ABCC9",
        "NOTCH3",
        "DES",
        "ANPEP",
        "CD248",
        "CSPG4",
        "MCAM",
        "NDUFA4L2",
        "HIGD1B",
        "COX4I2",
    ],
    "SMC": [
        "ACTA2",
        "MYH11",
        "TAGLN",
        "CNN1",
        "MYLK",
        "MYOCD",
        "SMTN",
        "CALD1",
        "TPM2",
        "MYL9",
        "LMOD1",
        "ACTG2",
    ],
    "Fibroblast_VLMC": [
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "DCN",
        "LUM",
        "PDGFRA",
        "ABCA8",
        "COL6A1",
        "COL6A2",
        "COL6A3",
        "FBLN1",
        "FBLN2",
        "FBLN5",
        "LTBP2",
        "LTBP4",
        "BGN",
        "DPT",
        "FBN1",
        "MFAP4",
        "CFD",
        "C7",
        "PI16",
        "CXCL14",
        "CCDC80",
        "COL14A1",
        "COL15A1",
        "ADAMTS2",
        "ADAMTS9",
        "IGFBP7",
        "APOD",
        "MGP",
        "CLEC3B",
        "SFRP2",
        "MFAP5",
    ],
    "Inflammatory_or_activation": [
        "HLA-A",
        "HLA-B",
        "HLA-C",
        "HLA-E",
        "CFB",
        "CST3",
        "SPP1",
        "CHI3L2",
        "IFITM3",
        "ISG15",
        "IFI6",
        "CXCL12",
        "CCL2",
        "IL6",
    ],
    "Matrix_remodeling": [
        "ADAMTS1",
        "ADAMTS4",
        "MMP2",
        "MMP9",
        "TIMP1",
        "TIMP3",
        "SERPINE1",
        "THBS1",
    ],
}

WEIGHTED_REFERENCE = {
    "Endothelial": {
        "CLDN5": 10,
        "PECAM1": 10,
        "VWF": 8,
        "CDH5": 8,
        "FLT1": 7,
        "KDR": 7,
        "TIE1": 6,
        "ESAM": 6,
        "ERG": 5,
        "CD34": 5,
        "ICAM2": 5,
        "NOS3": 4,
        "THBD": 4,
        "SELE": 3,
        "TJP1": 3,
        "EPAS1": 3,
        "RAMP2": 3,
        "GJA5": 2,
        "ACKR1": 2,
        "PLVAP": 2,
    },
    "Pericyte": {
        "PDGFRB": 10,
        "RGS5": 9,
        "KCNJ8": 8,
        "ABCC9": 7,
        "NOTCH3": 7,
        "DES": 6,
        "ANPEP": 5,
        "CD248": 5,
        "CSPG4": 4,
        "MCAM": 3,
        "NDUFA4L2": 3,
        "HIGD1B": 3,
        "COX4I2": 3,
    },
    "SMC": {
        "ACTA2": 10,
        "MYH11": 10,
        "TAGLN": 9,
        "CNN1": 8,
        "MYLK": 7,
        "MYOCD": 6,
        "SMTN": 5,
        "CALD1": 5,
        "TPM2": 4,
        "MYL9": 4,
    },
    "Fibroblast_VLMC": {
        "COL1A1": 10,
        "COL1A2": 9,
        "COL3A1": 9,
        "DCN": 8,
        "LUM": 8,
        "PDGFRA": 7,
        "ABCA8": 7,
        "COL6A1": 6,
        "COL6A2": 6,
        "COL6A3": 6,
        "FBLN1": 6,
        "FBLN2": 5,
        "FBLN5": 5,
        "LTBP2": 5,
        "LTBP4": 5,
        "BGN": 5,
        "DPT": 5,
        "FBN1": 5,
        "MFAP4": 4,
        "CFD": 4,
        "C7": 4,
        "PI16": 4,
        "CXCL14": 4,
        "COL14A1": 4,
        "COL15A1": 4,
        "ADAMTS2": 3,
        "ADAMTS9": 3,
        "IGFBP7": 3,
        "APOD": 3,
        "MGP": 3,
    },
}

REFERENCE_ORDER = ["Endothelial", "Pericyte", "SMC", "Fibroblast_VLMC"]
PALETTE = {
    "Endothelial": "#4C78A8",
    "Pericyte": "#E39D3E",
    "SMC": "#D35F5F",
    "Fibroblast_VLMC": "#4E9B68",
    "Inflammatory_or_activation": "#8C6BB1",
    "Matrix_remodeling": "#8C6D31",
}


def is_artifact(gene: str) -> bool:
    gene = str(gene)
    return gene in ARTIFACT_EXACT or gene.startswith(ARTIFACT_PREFIXES)


def gene_category(gene: str) -> str:
    gene = str(gene)
    if is_artifact(gene):
        return "artifact"
    hits = [cat for cat, genes in REFERENCE.items() if gene in genes]
    return hits[0] if hits else "other"


def read_modality(modality: str) -> dict[str, pd.DataFrame]:
    return {
        "markers": pd.read_csv(BASE / f"standard_omnicell_findallmarkers_{modality}_standard_selected_cluster_top200.csv"),
        "clusters": pd.read_csv(BASE / f"standard_omnicell_cluster_summary_{modality}.csv"),
        "dot": pd.read_csv(BASE / f"standard_omnicell_reference_dotplot_source_{modality}.csv"),
    }


def add_marker_flags(markers: pd.DataFrame) -> pd.DataFrame:
    out = markers.copy()
    out["gene_symbol"] = out["gene_symbol"].astype(str)
    out["marker_category"] = out["gene_symbol"].map(gene_category)
    out["artifact_or_housekeeping"] = out["marker_category"].eq("artifact")
    out["vascular_reference_marker"] = out["marker_category"].isin(REFERENCE_ORDER)
    return out


def category_scores(dot: pd.DataFrame) -> pd.DataFrame:
    d = dot.copy()
    d["marker_category"] = d["gene_symbol"].map(gene_category)
    d = d[d["marker_category"].isin(REFERENCE_ORDER)]
    rows = []
    for (group, cat), sub in d.groupby(["group", "marker_category"], observed=True):
        rows.append(
            {
                "group": group,
                "marker_category": cat,
                "mean_z": sub["z_log1p_mean_expression"].mean(),
                "mean_pct_expressing": sub["pct_expressing"].mean(),
                "n_reference_genes": sub["gene_symbol"].nunique(),
                "top_reference_genes_by_z": ", ".join(
                    sub.sort_values("z_log1p_mean_expression", ascending=False)["gene_symbol"].head(8)
                ),
            }
        )
    return pd.DataFrame(rows)


def weighted_category_scores(dot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, sub in dot.groupby("group", observed=True):
        for cat, weights in WEIGHTED_REFERENCE.items():
            x = sub[sub["gene_symbol"].isin(weights)].copy()
            if x.empty:
                rows.append(
                    {
                        "group": group,
                        "marker_category": cat,
                        "weighted_z": np.nan,
                        "weighted_pct_expressing": np.nan,
                        "n_weighted_genes_detected": 0,
                        "top_weighted_genes_by_z": "",
                    }
                )
                continue
            w = x["gene_symbol"].map(weights).astype(float).to_numpy()
            rows.append(
                {
                    "group": group,
                    "marker_category": cat,
                    "weighted_z": np.average(x["z_log1p_mean_expression"].to_numpy(), weights=w),
                    "weighted_pct_expressing": np.average(x["pct_expressing"].to_numpy(), weights=w),
                    "n_weighted_genes_detected": x["gene_symbol"].nunique(),
                    "top_weighted_genes_by_z": ", ".join(
                        x.sort_values("z_log1p_mean_expression", ascending=False)["gene_symbol"].head(8)
                    ),
                }
            )
    return pd.DataFrame(rows)


def cluster_work_table(modality: str, data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    markers = add_marker_flags(data["markers"])
    selected = data["clusters"][data["clusters"]["cluster_key"].eq("louvain_r0p2")].copy()
    scores = category_scores(data["dot"])
    weighted = weighted_category_scores(data["dot"])
    score_wide = weighted.pivot(index="group", columns="marker_category", values="weighted_z").reset_index()
    score_wide.columns.name = None
    rows = []
    for cluster, sub in markers.groupby("group", sort=True):
        clean = sub[~sub["artifact_or_housekeeping"]].sort_values("rank")
        vascular = clean[clean["vascular_reference_marker"]]
        supportive = clean[clean["marker_category"].isin(REFERENCE.keys())]
        row = {
            "modality": modality,
            "cluster": cluster,
            "top_clean_markers": ", ".join(clean["gene_symbol"].head(18)),
            "top_vascular_reference_markers": ", ".join(vascular["gene_symbol"].head(18)),
            "top_supportive_markers": ", ".join(supportive["gene_symbol"].head(18)),
            "artifact_markers_in_top200": int(sub["artifact_or_housekeeping"].sum()),
            "vascular_reference_markers_in_top200": int(sub["vascular_reference_marker"].sum()),
        }
        rows.append(row)
    work = pd.DataFrame(rows)
    work = work.merge(selected, left_on="cluster", right_on="cluster", how="left")
    work = work.merge(score_wide, left_on="cluster", right_on="group", how="left").drop(columns=["group"], errors="ignore")
    for cat in REFERENCE_ORDER:
        if cat not in work.columns:
            work[cat] = np.nan
    work["max_weighted_score"] = work[REFERENCE_ORDER].max(axis=1)
    sorted_scores = np.sort(work[REFERENCE_ORDER].to_numpy(dtype=float), axis=1)
    work["weighted_score_margin"] = sorted_scores[:, -1] - sorted_scores[:, -2]
    work["suggested_annotation"] = work[REFERENCE_ORDER].idxmax(axis=1)
    work.loc[work["max_weighted_score"].lt(0.50), "suggested_annotation"] = "low_confidence_mixed"
    work.loc[work["n_points"].lt(50), "suggested_annotation"] = "exclude_low_n"
    work["suggested_annotation_note"] = work.apply(annotation_note, axis=1)
    return markers, scores, weighted, work


def annotation_note(row: pd.Series) -> str:
    if row.get("suggested_annotation") == "exclude_low_n":
        return "Very small cluster; exclude from biological naming unless manually validated."
    if row.get("suggested_annotation") == "low_confidence_mixed":
        return "No weighted vascular module exceeds 0.5; treat as mixed/low-confidence spatial or transitional state."
    vals = {cat: row.get(cat, np.nan) for cat in REFERENCE_ORDER}
    vals = {k: v for k, v in vals.items() if pd.notna(v)}
    if not vals:
        return "No reference-marker score available; inspect top markers."
    ranked = sorted(vals.items(), key=lambda x: x[1], reverse=True)
    top, second = ranked[0], ranked[1] if len(ranked) > 1 else (None, np.nan)
    gap = top[1] - second[1] if second[0] is not None else np.nan
    if pd.notna(gap) and gap < 0.15:
        return f"Mixed or transitional; top module {top[0]} only weakly exceeds {second[0]}."
    return f"{top[0]}-enriched by reference markers; verify with clean DE genes."


def plot_reference_dotplot(dot: pd.DataFrame, modality: str, selected_genes: list[str]) -> None:
    d = dot[dot["gene_symbol"].isin(selected_genes)].copy()
    d["gene_symbol"] = pd.Categorical(d["gene_symbol"], categories=selected_genes[::-1], ordered=True)
    groups = sorted(d["group"].unique())
    d["group"] = pd.Categorical(d["group"], categories=groups, ordered=True)
    fig_h = max(5.0, 0.22 * len(selected_genes) + 1.4)
    fig_w = max(4.8, 0.42 * len(groups) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    size = 12 + 140 * d["pct_expressing"].clip(0, 1)
    sc = ax.scatter(
        d["group"].cat.codes,
        d["gene_symbol"].cat.codes,
        s=size,
        c=d["z_log1p_mean_expression"].clip(-2, 2),
        cmap="RdBu_r",
        vmin=-2,
        vmax=2,
        edgecolor="#7c8793",
        linewidth=0.25,
    )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_yticks(range(len(selected_genes)))
    ax.set_yticklabels(selected_genes[::-1])
    ax.set_xlabel("graph cluster")
    ax.set_ylabel("")
    ax.set_title(f"Clean vascular reference markers: {modality}", loc="left", fontweight="bold")
    ax.grid(axis="both", color="#D9DEE7", linewidth=0.5, alpha=0.75)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("z-score of log1p mean")
    for pct, label in [(0.25, "25%"), (0.5, "50%"), (0.75, "75%")]:
        ax.scatter([], [], s=12 + 140 * pct, c="white", edgecolor="#7c8793", linewidth=0.25, label=label)
    ax.legend(title="pct expressing", bbox_to_anchor=(1.03, 0.25), loc="center left", borderaxespad=0)
    fig.tight_layout()
    stem = OUT / f"clean_reference_dotplot_{modality}"
    for ext in ["png", "pdf", "svg", "tiff"]:
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=700 if ext in {"png", "tiff"} else None, bbox_inches="tight")
    plt.close(fig)


def plot_weighted_score_heatmap(weighted: pd.DataFrame, modality: str) -> None:
    wide = weighted.pivot(index="marker_category", columns="group", values="weighted_z").reindex(REFERENCE_ORDER)
    fig_w = max(3.6, 0.42 * wide.shape[1] + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, 2.4))
    im = ax.imshow(wide.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(wide.shape[1]))
    ax.set_xticklabels(wide.columns, rotation=45, ha="right")
    ax.set_yticks(range(wide.shape[0]))
    ax.set_yticklabels(wide.index)
    ax.set_title(f"Weighted vascular marker modules: {modality}", loc="left", fontweight="bold")
    for i in range(wide.shape[0]):
        for j in range(wide.shape[1]):
            val = wide.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6, color="#111827")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("weighted marker z-score")
    fig.tight_layout()
    stem = OUT / f"weighted_reference_module_heatmap_{modality}"
    for ext in ["png", "pdf", "svg", "tiff"]:
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=700 if ext in {"png", "tiff"} else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
        }
    )
    selected_genes = [
        "CLDN5",
        "PECAM1",
        "VWF",
        "CDH5",
        "FLT1",
        "KDR",
        "TIE1",
        "ESAM",
        "ERG",
        "TJP1",
        "EPAS1",
        "RAMP2",
        "IFI27",
        "PLVAP",
        "ACKR1",
        "GJA5",
        "PDGFRB",
        "RGS5",
        "KCNJ8",
        "ABCC9",
        "NOTCH3",
        "ANPEP",
        "CD248",
        "CSPG4",
        "MCAM",
        "NDUFA4L2",
        "HIGD1B",
        "COX4I2",
        "ACTA2",
        "MYH11",
        "TAGLN",
        "CNN1",
        "MYLK",
        "MYOCD",
        "SMTN",
        "CALD1",
        "TPM2",
        "MYL9",
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "DCN",
        "LUM",
        "PDGFRA",
        "ABCA8",
        "COL6A1",
        "COL6A2",
        "FBLN1",
        "LTBP2",
        "BGN",
        "FBN1",
        "MFAP4",
        "C7",
        "PI16",
        "CXCL14",
        "COL14A1",
        "COL15A1",
        "ADAMTS9",
        "IGFBP7",
        "APOD",
        "MGP",
    ]
    outputs = {}
    for modality in ["single_cell", "spatial"]:
        data = read_modality(modality)
        markers, scores, weighted, work = cluster_work_table(modality, data)
        markers.to_csv(OUT / f"{modality}_top200_markers_with_flags.csv", index=False)
        markers[~markers["artifact_or_housekeeping"]].groupby("group", group_keys=False).head(50).to_csv(
            OUT / f"{modality}_clean_top50_markers.csv", index=False
        )
        scores.to_csv(OUT / f"{modality}_reference_module_scores.csv", index=False)
        weighted.to_csv(OUT / f"{modality}_weighted_reference_module_scores.csv", index=False)
        work.to_csv(OUT / f"{modality}_cluster_annotation_worktable.csv", index=False)
        plot_reference_dotplot(data["dot"], modality, selected_genes)
        plot_weighted_score_heatmap(weighted, modality)
        outputs[modality] = (markers, scores, weighted, work)
    excel = OUT / "standard_omnicell_clean_marker_annotation_workbook.xlsx"
    with pd.ExcelWriter(excel, engine="openpyxl") as writer:
        for modality, (markers, scores, weighted, work) in outputs.items():
            work.to_excel(writer, sheet_name=f"{modality}_worktable"[:31], index=False)
            markers[~markers["artifact_or_housekeeping"]].groupby("group", group_keys=False).head(50).to_excel(
                writer, sheet_name=f"{modality}_clean_top50"[:31], index=False
            )
            scores.to_excel(writer, sheet_name=f"{modality}_module_scores"[:31], index=False)
            weighted.to_excel(writer, sheet_name=f"{modality}_weighted_scores"[:31], index=False)
            markers.to_excel(writer, sheet_name=f"{modality}_all_flags"[:31], index=False)
    print(excel)


if __name__ == "__main__":
    main()

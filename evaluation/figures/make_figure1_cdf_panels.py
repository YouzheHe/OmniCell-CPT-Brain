#!/usr/bin/env python
"""Draw final Figure 1C/D/F panels for the NVU OmniCell study.

Panels:
- 1C: annotated all-cell composition across modalities.
- 1D: study-axis coverage across modalities.
- 1F: polished all-cell OmniCell latent atlas.

The script uses completed Figure 1 source-data tables and does not recompute
OmniCell embeddings.
"""

from __future__ import annotations
import os

import json
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


WORK_ROOT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}"))
PROJECT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/nvu_vascular"))
FIG1 = PROJECT / "figures" / "figure1_final_panels"
SRC = FIG1 / "source_data"
PACKAGE = PROJECT / "figure1_figure2_complete_package_20260624" / "Figure1"
DATASET_ROOT = WORK_ROOT / "NVU_hyz"
SPLIT_37774677_ROOT = WORK_ROOT / "data" / "37774677"
SPLIT_37774677_AUDIT = SRC / "fig1c_37774677_split_annotation_audit.csv"
UCSC_37774677_META = SPLIT_37774677_ROOT / "ucsc_ad_aging_brain_meta.tsv"
EXCLUDE_FULL_COMPOSITION_SAMPLE_IDS = {"37824649"}

SPLIT_37774677_DEFAULT_LABELS = {
    "Astrocytes.h5ad": "Ast",
    "Excitatory_neurons_set1.h5ad": "Exc L2-3 CBLN2 LINC02306",
    "Excitatory_neurons_set2.h5ad": "Exc",
    "Excitatory_neurons_set3.h5ad": "Exc",
    "Immune_cells.h5ad": "Mic",
    "Inhibitory_neurons.h5ad": "Inh",
    "Oligodendrocytes.h5ad": "Oli",
    "OPCs.h5ad": "OPC",
    "Vasculature_cells.h5ad": "End",
}

ALLCELL_SOURCE = SRC / "fig1_allcell_omnicell_embedding_source.csv"
FULL_INPUT_SOURCE = FIG1 / "fig1_input_region_counts_source_data.csv"
MANIFEST_SOURCE = DATASET_ROOT / "dataset_manifest.json"
FULL_CELL_LABEL_PRIORITY = [
    "CellType_m",
    "subclass.v4",
    "CellType",
    "celltype_unit",
    "subcelltype",
    "ground_truth_celltype",
    "ground_truth_label",
    "cell_type",
    "celltype",
    "major_cell_type",
    "Bayes24_anno",
    "bayes.anno",
]


PALETTE = {
    "ink": "#1F2933",
    "muted": "#667085",
    "grid": "#D7DEE8",
    "soft": "#F6F8FB",
}

BROAD_ORDER = [
    "Excitatory neuron",
    "Inhibitory neuron",
    "Neuron",
    "Astrocyte",
    "Oligodendrocyte",
    "OPC",
    "Microglia/immune",
    "Vascular",
    "Ependymal/choroid",
    "Other",
]

BROAD_COLORS = {
    "Excitatory neuron": "#4F7EA8",
    "Inhibitory neuron": "#8E72A7",
    "Neuron": "#6D8DC8",
    "Astrocyte": "#58A87A",
    "Oligodendrocyte": "#B8A35A",
    "OPC": "#E0A458",
    "Microglia/immune": "#8A7A64",
    "Vascular": "#00A6A6",
    "Ependymal/choroid": "#C66A5B",
    "Other": "#C7C9CC",
}

BROAD_SHORT = {
    "Excitatory neuron": "Excitatory",
    "Inhibitory neuron": "Inhibitory",
    "Neuron": "Neuron",
    "Astrocyte": "Astrocyte",
    "Oligodendrocyte": "Oligo.",
    "OPC": "OPC",
    "Microglia/immune": "Microglia",
    "Vascular": "Vascular",
    "Ependymal/choroid": "Ependymal",
    "Other": "Other",
}

MODALITY_ORDER = ["single-cell / snRNA", "spatial transcriptomics"]
MODALITY_LABELS = {
    "single-cell / snRNA": "scRNA/snRNA-seq",
    "spatial transcriptomics": "Spatial transcriptomics",
}

STUDY_AXIS_ORDER = [
    "AD/control cohorts",
    "AD pathology/resilience",
    "Aging cortex",
    "Reference cortex",
    "Other public brain cohorts",
]

REGMOD_ORDER = [
    "PFC sc/snRNA",
    "Cortex sc/snRNA",
    "Cortex spatial",
    "Hippocampus sc/snRNA",
    "Hippocampus spatial",
]

REGMOD_COLORS = {
    "PFC sc/snRNA": "#7B6FA6",
    "Cortex sc/snRNA": "#4E79A7",
    "Cortex spatial": "#76B7B2",
    "Hippocampus sc/snRNA": "#E28E52",
    "Hippocampus spatial": "#C56E5B",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 7,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "legend.frameon": False,
        "agg.path.chunksize": 20000,
    }
)


def save_panel(fig: plt.Figure, stem: Path, dpi: int = 900) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def fmt_count(n: float) -> str:
    n = float(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return f"{int(n):,}"


def normalize_modality(series: pd.Series) -> pd.Series:
    out = series.astype(str).replace({"unknown": "single-cell / snRNA", "single_cell": "single-cell / snRNA", "spatial": "spatial transcriptomics"})
    out = out.replace({"single-cell / snRNA data": "single-cell / snRNA"})
    return out


def broad_cell_class(label: object) -> str:
    if pd.isna(label):
        return "Other"
    s = str(label).strip()
    low = s.lower()
    clean = low.replace("-", "_").replace("/", "_").replace(" ", "_")
    if low in {"", "0", "na", "nan", "none", "unknown", "unclassified", "other", "mixed", "ambiguous"}:
        return "Other"
    if re.match(r"^(exc|ex)\b", low):
        return "Excitatory neuron"
    if re.match(r"^(inh|in)\b", low):
        return "Inhibitory neuron"
    if clean in {"ast", "astro", "astrocyte", "astrocytes"} or re.match(r"^(ast|astro)\b", low):
        return "Astrocyte"
    if clean in {"oli", "odc", "oligo", "oligodendrocyte", "oligodendrocytes"} or re.match(r"^(oli|oligo)\b", low):
        return "Oligodendrocyte"
    if clean in {"opc", "opcs", "oligodendrocyte_precursor_cell"} or re.match(r"^opc\b", low):
        return "OPC"
    if clean in {"mic", "mg", "micro", "microglia", "cams", "cam", "t_cells", "t_cell"} or re.match(r"^(mic|micro)\b", low):
        return "Microglia/immune"
    if (
        clean in {"end", "endo", "endothelial", "per", "pericyte", "smc", "fib", "fibroblast", "vasculature_cells", "vsmc", "asmc"}
        or clean.startswith(("end_", "per", "fib", "smc", "vsmc", "asmc"))
    ):
        return "Vascular"
    if clean in {"inh", "in", "gaba", "gabaergic"} or any(
        k in low
        for k in [
            "inhibitory",
            "interneuron",
            "gaba",
            "pvalb",
            "sst",
            "vip",
            "lamp5",
            "reln",
            "chandelier",
            "sncg",
            "ndnf",
        ]
    ):
        return "Inhibitory neuron"
    if clean in {"ex", "exc", "ex_sub", "glut", "glutamatergic"} or any(
        k in low
        for k in [
            "excit",
            "glutamatergic",
            "glutamate",
            "pyramidal",
            "granule",
            "mitral",
            "tufted",
            "l2",
            "l3",
            "l4",
            "l5",
            "l6",
            " it ",
            " et ",
            " ct ",
            " np ",
            "it neuron",
            "et neuron",
            "ct neuron",
            "np neuron",
            "ca1",
            "ca2",
            "ca3",
            "dg",
            "dentate",
            "subiculum",
            "sub_",
        ]
    ):
        return "Excitatory neuron"
    if low in {"neuron", "neurons", "neuronal", "neuronal cell"} or low.endswith(" neuron") or low.endswith(" neurons"):
        return "Neuron"
    if clean in {"asc", "astro"} or "astro" in low or "bergmann glial" in low:
        return "Astrocyte"
    if clean in {"opc", "oligodendrocyte_precursor_cell"} or "oligodendrocyte precursor" in low or low == "opc" or " opcs" in low:
        return "OPC"
    if clean in {"odc", "oligo", "oligodendrocyte"} or "oligodendrocyte" in low or low.startswith("oligo"):
        return "Oligodendrocyte"
    if clean in {"mg", "micro"} or any(
        k in low
        for k in [
            "microglia",
            "microglial",
            "macrophage",
            "immune",
            "monocyte",
            "leukocyte",
            "lymphocyte",
            "myeloid",
            " t cell",
            " b cell",
            "mast cell",
            "blood cell",
        ]
    ):
        return "Microglia/immune"
    if any(
        k in low
        for k in [
            "endo",
            "endothelial",
            "endotheli",
            "pericyte",
            "mural",
            "vascular",
            "vasc",
            "vlmc",
            "fibroblast",
            "smooth muscle",
            " smc",
            "arter",
            "venous",
            "vein",
            "capillary",
            "perivascular",
        ]
    ):
        return "Vascular"
    if any(k in low for k in ["ependy", "choroid"]):
        return "Ependymal/choroid"
    return "Other"


def present_order(values: pd.Series, preferred: list[str]) -> list[str]:
    present = set(values.dropna().astype(str))
    return [x for x in preferred if x in present] + sorted(present.difference(preferred))


def load_allcell() -> pd.DataFrame:
    df = pd.read_csv(ALLCELL_SOURCE)
    df["modality_display"] = normalize_modality(df["modality_display"])
    df["cell_class"] = df["cell_class"].fillna("Other").astype(str)
    df.loc[~df["cell_class"].isin(BROAD_ORDER), "cell_class"] = "Other"
    df["region_modality"] = df.apply(region_modality_label, axis=1)
    return df


def region_modality_label(row: pd.Series) -> str:
    detail = str(row.get("brain_region_detail", ""))
    region = str(row.get("brain_region", ""))
    modality = str(row.get("modality_display", ""))
    if "Prefrontal" in detail:
        return "PFC sc/snRNA"
    if region == "Cortex":
        return "Cortex spatial" if "spatial" in modality else "Cortex sc/snRNA"
    if region == "Hippocampus":
        return "Hippocampus spatial" if "spatial" in modality else "Hippocampus sc/snRNA"
    return f"{region} {modality}".strip()


def load_manifest() -> list[dict]:
    payload = json.loads(MANIFEST_SOURCE.read_text(encoding="utf-8"))
    return payload.get("samples", [])


def manifest_modality(sample: dict) -> str:
    text = " ".join(str(sample.get(c, "")) for c in ["sample_id", "source_h5ad", "region_source_note"]).lower()
    coord_dim = sample.get("coord_dim", "")
    if isinstance(coord_dim, str):
        coord_text = coord_dim.lower()
        if coord_text in {"2", "2d", "xy", "spatial"}:
            return "spatial transcriptomics"
    elif coord_dim == 2:
        return "spatial transcriptomics"
    if "spatial" in text or "saptial" in text:
        return "spatial transcriptomics"
    return "single-cell / snRNA"


def load_37774677_split_counts(sample_id: str, modality: str, manifest_n: int) -> list[dict]:
    rows: list[dict] = []
    audit_rows: list[dict] = []
    if UCSC_37774677_META.exists():
        usecols = ["cellName", "Major_Cell_Type", "Cell_Type"]
        meta = pd.read_csv(UCSC_37774677_META, sep="\t", usecols=usecols)
        counts = meta["Cell_Type"].value_counts(dropna=False)
        observed_n = int(counts.sum())
        for label, n in counts.items():
            label_text = "NA" if pd.isna(label) else str(label)
            rows.append(
                {
                    "sample_id": sample_id,
                    "modality_display": modality,
                    "label_col": "UCSC_Cell_Type",
                    "cell_label": label_text,
                    "cell_class": broad_cell_class(label_text),
                    "n": int(n),
                    "manifest_n": manifest_n,
                    "obs_path": str(UCSC_37774677_META),
                }
            )
        audit_rows.append(
            {
                "file": str(UCSC_37774677_META),
                "status": "read",
                "n_cells": observed_n,
                "label_col": "Cell_Type",
                "note": "Full UCSC Cell Browser metadata used for 37774677 annotation counts.",
            }
        )
        if manifest_n > observed_n:
            audit_rows.append(
                {
                    "file": str(UCSC_37774677_META),
                    "status": "manifest_gap_not_plotted",
                    "n_cells": int(manifest_n - observed_n),
                    "label_col": "",
                    "note": "37774677 cells without UCSC metadata are not plotted as Other.",
                }
            )
        pd.DataFrame(audit_rows).to_csv(SPLIT_37774677_AUDIT, index=False)
        return rows

    if not SPLIT_37774677_ROOT.exists():
        audit_rows.append(
            {
                "file": str(SPLIT_37774677_ROOT),
                "status": "missing_split_annotation_dir",
                "n_cells": 0,
                "label_col": "",
                "note": "37774677 split annotation directory was not found.",
            }
        )
        pd.DataFrame(audit_rows).to_csv(SPLIT_37774677_AUDIT, index=False)
        rows.append(
            {
                "sample_id": sample_id,
                "modality_display": modality,
                "label_col": "missing_37774677_split_annotations",
                "cell_label": "Unclassified",
                "cell_class": "Other",
                "n": manifest_n,
                "manifest_n": manifest_n,
                "obs_path": str(SPLIT_37774677_ROOT),
            }
        )
        return rows

    try:
        import anndata as ad
    except Exception as exc:
        audit_rows.append(
            {
                "file": str(SPLIT_37774677_ROOT),
                "status": "missing_anndata",
                "n_cells": 0,
                "label_col": "",
                "note": repr(exc),
            }
        )
        pd.DataFrame(audit_rows).to_csv(SPLIT_37774677_AUDIT, index=False)
        rows.append(
            {
                "sample_id": sample_id,
                "modality_display": modality,
                "label_col": "unreadable_37774677_split_annotations",
                "cell_label": "Unclassified",
                "cell_class": "Other",
                "n": manifest_n,
                "manifest_n": manifest_n,
                "obs_path": str(SPLIT_37774677_ROOT),
            }
        )
        return rows

    files = sorted(SPLIT_37774677_ROOT.glob("*.h5ad"))
    observed_n = 0
    for h5ad_path in files:
        default_label = SPLIT_37774677_DEFAULT_LABELS.get(h5ad_path.name, h5ad_path.stem)
        try:
            adata = ad.read_h5ad(h5ad_path, backed="r")
            obs = adata.obs
            label_col = "cell_type_high_resolution" if "cell_type_high_resolution" in obs.columns else None
            if label_col is None:
                counts = pd.Series({default_label: int(adata.n_obs)})
                label_col_text = "split_file_default_label"
            else:
                counts = obs[label_col].value_counts(dropna=False)
                label_col_text = label_col
            observed_n += int(counts.sum())
            for label, n in counts.items():
                label_text = "NA" if pd.isna(label) else str(label)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "modality_display": modality,
                        "label_col": label_col_text,
                        "cell_label": label_text,
                        "cell_class": broad_cell_class(label_text),
                        "n": int(n),
                        "manifest_n": manifest_n,
                        "obs_path": str(h5ad_path),
                    }
                )
            audit_rows.append(
                {
                    "file": str(h5ad_path),
                    "status": "read",
                    "n_cells": int(counts.sum()),
                    "label_col": label_col_text,
                    "note": "",
                }
            )
            try:
                adata.file.close()
            except Exception:
                pass
        except Exception as exc:
            audit_rows.append(
                {
                    "file": str(h5ad_path),
                    "status": "unreadable",
                    "n_cells": 0,
                    "label_col": "",
                    "note": str(exc),
                }
            )

    if not rows:
        rows.append(
            {
                "sample_id": sample_id,
                "modality_display": modality,
                "label_col": "unreadable_37774677_split_annotations",
                "cell_label": "Unclassified",
                "cell_class": "Other",
                "n": manifest_n,
                "manifest_n": manifest_n,
                "obs_path": str(SPLIT_37774677_ROOT),
            }
        )
    elif manifest_n > observed_n:
        audit_rows.append(
            {
                "file": str(SPLIT_37774677_ROOT),
                "status": "manifest_gap_not_plotted",
                "n_cells": int(manifest_n - observed_n),
                "label_col": "",
                "note": "Some split h5ad annotation files are missing or still unreadable; rerun once files are complete.",
            }
        )
    pd.DataFrame(audit_rows).to_csv(SPLIT_37774677_AUDIT, index=False)
    return rows


def sample_obs_path(sample_id: str) -> Path:
    direct = DATASET_ROOT / sample_id / "obs.parquet"
    if direct.exists():
        return direct
    if sample_id == "AD_sc":
        return DATASET_ROOT / "AD_Hip_sc" / "obs.parquet"
    return direct


def sample_n_cells(sample: dict, obs_path=None) -> int:
    for key in ["n_cells", "n_spots", "n_obs"]:
        if sample.get(key) is not None:
            return int(sample[key])
    if obs_path is not None and obs_path.exists():
        return int(pq.ParquetFile(obs_path).metadata.num_rows)
    return 0


def cell_label_column(obs_path: Path):
    names = set(pq.ParquetFile(obs_path).schema.names)
    for col in FULL_CELL_LABEL_PRIORITY:
        if col in names:
            return col
    return None


def load_full_cell_composition() -> pd.DataFrame:
    rows = []
    for sample in load_manifest():
        sample_id = str(sample.get("sample_id", ""))
        if sample_id in EXCLUDE_FULL_COMPOSITION_SAMPLE_IDS:
            continue
        obs_path = sample_obs_path(sample_id)
        modality = manifest_modality(sample)
        manifest_n = sample_n_cells(sample, obs_path)

        if sample_id == "37774677":
            rows.extend(load_37774677_split_counts(sample_id, modality, manifest_n))
            continue

        if not obs_path.exists():
            rows.append(
                {
                    "sample_id": sample_id,
                    "modality_display": modality,
                    "label_col": "missing_obs",
                    "cell_label": "Unclassified",
                    "cell_class": "Other",
                    "n": manifest_n,
                    "manifest_n": manifest_n,
                    "obs_path": str(obs_path),
                }
            )
            continue

        label_col = cell_label_column(obs_path)
        if label_col is None:
            rows.append(
                {
                    "sample_id": sample_id,
                    "modality_display": modality,
                    "label_col": "no_celltype_column",
                    "cell_label": "Unclassified",
                    "cell_class": "Other",
                    "n": manifest_n,
                    "manifest_n": manifest_n,
                    "obs_path": str(obs_path),
                }
            )
            continue

        labels = pd.read_parquet(obs_path, columns=[label_col])[label_col]
        counts = labels.value_counts(dropna=False)
        observed_n = int(counts.sum())
        for label, n in counts.items():
            label_text = "NA" if pd.isna(label) else str(label)
            rows.append(
                {
                    "sample_id": sample_id,
                    "modality_display": modality,
                    "label_col": label_col,
                    "cell_label": label_text,
                    "cell_class": broad_cell_class(label),
                    "n": int(n),
                    "manifest_n": manifest_n,
                    "obs_path": str(obs_path),
                }
            )
        if manifest_n > observed_n:
            rows.append(
                {
                    "sample_id": sample_id,
                    "modality_display": modality,
                    "label_col": label_col,
                    "cell_label": "Unclassified manifest remainder",
                    "cell_class": "Other",
                    "n": manifest_n - observed_n,
                    "manifest_n": manifest_n,
                    "obs_path": str(obs_path),
                }
            )

    full = pd.DataFrame(rows)
    full.loc[~full["cell_class"].isin(BROAD_ORDER), "cell_class"] = "Other"
    full.to_csv(SRC / "fig1c_allcell_cell_class_composition_dataset_source.csv", index=False)
    other_audit = (
        full[full["cell_class"].eq("Other")]
        .groupby(["modality_display", "label_col", "cell_label"], observed=True)["n"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    other_audit.to_csv(SRC / "fig1c_allcell_cell_class_other_audit.csv", index=False)

    summary = full.groupby(["modality_display", "cell_class"], observed=True)["n"].sum().reset_index()
    totals = summary.groupby("modality_display", observed=True)["n"].sum().rename("total")
    summary = summary.merge(totals, on="modality_display")
    summary["fraction"] = summary["n"] / summary["total"]
    summary["composition_scope"] = "dataset_manifest obs labels, with 37774677 split annotations merged and 37824649 excluded"
    summary["cell_class"] = pd.Categorical(summary["cell_class"], BROAD_ORDER, ordered=True)
    summary = summary.sort_values(["modality_display", "cell_class"]).reset_index(drop=True)
    summary.to_csv(SRC / "fig1c_allcell_cell_class_composition_source.csv", index=False)
    return summary


def draw_panel_c(summary: pd.DataFrame) -> None:
    modalities = [m for m in MODALITY_ORDER if m in set(summary["modality_display"])]
    grand_total = int(summary["n"].sum())
    fig, ax = plt.subplots(figsize=(8.2, 2.55))
    fig.subplots_adjust(left=0.17, right=0.76, top=0.70, bottom=0.24)

    y_pos = np.arange(len(modalities))[::-1]
    for y, modality in zip(y_pos, modalities):
        left = 0.0
        sub = summary[summary["modality_display"].eq(modality)].set_index("cell_class")
        for cls in BROAD_ORDER:
            if cls not in sub.index:
                continue
            frac = float(sub.loc[cls, "fraction"])
            n = int(sub.loc[cls, "n"])
            if frac <= 0:
                continue
            ax.barh(y, frac, left=left, height=0.52, color=BROAD_COLORS[cls], edgecolor="white", linewidth=0.7)
            if frac >= 0.085:
                ax.text(left + frac / 2, y, BROAD_SHORT[cls], ha="center", va="center", fontsize=6.2, color="white", fontweight="bold")
            left += frac
        ax.text(1.018, y, f"n = {fmt_count(sub['n'].sum())}", ha="left", va="center", fontsize=6.4, color=PALETTE["muted"])

    ax.set_yticks(y_pos, [MODALITY_LABELS.get(m, m) for m in modalities])
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.00], ["0", "25", "50", "75", "100%"])
    ax.set_xlabel("fraction of all input cells/spots")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    ax.set_axisbelow(True)
    fig.text(0.17, 0.955, "Input cell-class composition across modalities", ha="left", va="top", fontsize=9.4, fontweight="bold")
    fig.text(
        0.17,
        0.885,
        f"{fmt_count(grand_total)} annotated input cells/spots; 37774677 split annotations are merged and 37824649 is excluded.",
        fontsize=6.0,
        color=PALETTE["muted"],
        ha="left",
        va="top",
    )

    handles = [
        mpl.lines.Line2D([0], [0], marker="s", lw=0, markersize=5.2, markerfacecolor=BROAD_COLORS[c], markeredgewidth=0, label=BROAD_SHORT[c])
        for c in BROAD_ORDER
        if c in set(summary["cell_class"])
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.035, 0.50), fontsize=5.8, title="Broad class", title_fontsize=6.4, labelspacing=0.38)
    save_panel(fig, FIG1 / "fig1c_allcell_cell_class_composition")


def infer_study_axis(row: pd.Series) -> str:
    text = " ".join(str(row.get(c, "")) for c in ["sample_id", "source_h5ad", "region_source_note"]).lower()
    if "39402379" in text:
        return "Aging cortex"
    if "37774677" in text:
        return "AD pathology/resilience"
    if "ad_hip" in text or "ad_cortex" in text:
        return "AD/control cohorts"
    if "cortex_spatial" in text or "cortex_sc" in text:
        return "Reference cortex"
    return "Other public brain cohorts"


def draw_panel_d() -> None:
    full = pd.read_csv(FULL_INPUT_SOURCE)
    full["modality"] = normalize_modality(full["modality"])
    full["study_axis"] = full.apply(infer_study_axis, axis=1)
    grouped = (
        full.groupby(["study_axis", "modality"], observed=True)
        .agg(n_cells=("n_cells", "sum"), n_samples=("sample_id", "nunique"))
        .reset_index()
    )
    grouped["log10_cells"] = np.log10(grouped["n_cells"].astype(float) + 1.0)
    grouped.to_csv(SRC / "fig1d_study_axis_modality_coverage_source.csv", index=False)

    rows = [x for x in STUDY_AXIS_ORDER if x in set(grouped["study_axis"])]
    cols = [x for x in MODALITY_ORDER if x in set(grouped["modality"])]
    modality_colors = {
        "single-cell / snRNA": "#4E79A7",
        "spatial transcriptomics": "#76B7B2",
    }

    fig, ax = plt.subplots(figsize=(7.5, 3.25))
    fig.subplots_adjust(left=0.33, right=0.72, top=0.78, bottom=0.18)

    max_log = max(1.0, float(grouped["log10_cells"].max()))
    for i, axis in enumerate(rows):
        for j, modality in enumerate(cols):
            hit = grouped[grouped["study_axis"].eq(axis) & grouped["modality"].eq(modality)]
            if hit.empty:
                ax.scatter(j, i, s=95, facecolor="#F5F7FA", edgecolor="#D7DEE8", linewidth=0.7, zorder=2)
                ax.text(j, i, "0", ha="center", va="center", fontsize=6.0, color=PALETTE["muted"])
                continue
            n_cells = int(hit["n_cells"].iloc[0])
            n_samples = int(hit["n_samples"].iloc[0])
            log_cells = float(hit["log10_cells"].iloc[0])
            size = 130 + (log_cells / max_log) ** 2 * 920
            ax.scatter(j, i, s=size, color=modality_colors.get(modality, "#8ABBD0"), alpha=0.88, edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(j, i + 0.02, fmt_count(n_cells), ha="center", va="center", fontsize=6.2, color="white", fontweight="bold", zorder=4)
            ax.text(j, i + 0.30, f"{n_samples} samples", ha="center", va="center", fontsize=5.4, color=PALETTE["ink"], zorder=4)

    ax.set_xlim(-0.55, len(cols) - 0.45)
    ax.set_ylim(len(rows) - 0.55, -0.55)
    ax.set_xticks(np.arange(len(cols)), ["sc/snRNA-seq", "Spatial\ntranscriptomics"], fontsize=6.5)
    ax.set_yticks(np.arange(len(rows)), rows, fontsize=6.4)
    ax.tick_params(length=0)
    ax.grid(axis="both", color=PALETTE["grid"], linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_sizes = [1e5, 1e6, 5e6]
    handles = [
        ax.scatter([], [], s=130 + (np.log10(v + 1) / max_log) ** 2 * 920, color="#A7C7D8", alpha=0.88, edgecolor="white", linewidth=0.8)
        for v in legend_sizes
    ]
    labels = [fmt_count(v) for v in legend_sizes]
    modality_handles = [
        mpl.lines.Line2D([0], [0], marker="o", lw=0, markersize=5, markerfacecolor=modality_colors[c], markeredgewidth=0, label=("sc/snRNA-seq" if c == "single-cell / snRNA" else "Spatial"))
        for c in cols
    ]
    fig.legend(handles, labels, title="cells/spots", loc="center left", bbox_to_anchor=(0.77, 0.57), fontsize=5.8, title_fontsize=6.4, labelspacing=1.0)
    fig.legend(handles=modality_handles, loc="center left", bbox_to_anchor=(0.77, 0.24), fontsize=5.8, title="Modality", title_fontsize=6.4, labelspacing=0.55)

    fig.text(0.33, 0.955, "Study-axis coverage across modalities", ha="left", va="top", fontsize=9.4, fontweight="bold")
    fig.text(0.33, 0.885, "Counts are computed from the full Figure 1 input manifest.", fontsize=6.0, color=PALETTE["muted"], ha="left", va="top")
    save_panel(fig, FIG1 / "fig1d_study_axis_modality_coverage")


def add_umap_arrows(ax: plt.Axes) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = x1 - x0
    dy = y1 - y0
    base = (x0 + 0.045 * dx, y0 + 0.055 * dy)
    ax.annotate("", xy=(base[0] + 0.11 * dx, base[1]), xytext=base, arrowprops=dict(arrowstyle="-|>", lw=0.6, color=PALETTE["ink"]))
    ax.annotate("", xy=(base[0], base[1] + 0.11 * dy), xytext=base, arrowprops=dict(arrowstyle="-|>", lw=0.6, color=PALETTE["ink"]))
    ax.text(base[0] + 0.12 * dx, base[1] - 0.01 * dy, "UMAP1", ha="left", va="top", fontsize=5.4)
    ax.text(base[0] - 0.01 * dx, base[1] + 0.12 * dy, "UMAP2", ha="right", va="bottom", fontsize=5.4, rotation=90)


def umap_limits(df: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    x0, x1 = np.nanpercentile(df["omnicell_umap_1"], [0.2, 99.8])
    y0, y1 = np.nanpercentile(df["omnicell_umap_2"], [0.2, 99.8])
    dx = x1 - x0
    dy = y1 - y0
    return (x0 - 0.05 * dx, x1 + 0.05 * dx), (y0 - 0.05 * dy, y1 + 0.05 * dy)


def style_umap_axis(ax: plt.Axes, limits: tuple[tuple[float, float], tuple[float, float]], arrows: bool = False) -> None:
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if arrows:
        add_umap_arrows(ax)


def draw_umap(
    ax: plt.Axes,
    df: pd.DataFrame,
    color_col: str,
    colors: dict[str, str],
    order: list[str],
    title: str,
    label_names: bool = False,
    limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> None:
    cats = present_order(df[color_col], order)
    plot_order = [c for c in cats if c in {"Other", "Ependymal/choroid"}] + [c for c in cats if c not in {"Other", "Ependymal/choroid", "Vascular"}]
    if "Vascular" in cats:
        plot_order.append("Vascular")
    rng = np.random.default_rng(20260625)
    for cat in plot_order:
        sub = df[df[color_col].astype(str).eq(cat)]
        if sub.empty:
            continue
        if len(sub) > 1:
            sub = sub.iloc[rng.permutation(len(sub))]
        ax.scatter(
            sub["omnicell_umap_1"],
            sub["omnicell_umap_2"],
            s=0.22 if color_col == "cell_class" else 0.20,
            color=colors.get(cat, "#C7C9CC"),
            alpha=0.66 if color_col == "cell_class" else 0.58,
            linewidths=0,
            rasterized=True,
        )
    ax.set_title(title, loc="left", fontsize=8.0, fontweight="bold", pad=3)
    if limits is None:
        limits = umap_limits(df)
    style_umap_axis(ax, limits, arrows=True)
    if label_names:
        for cat in cats:
            sub = df[df[color_col].astype(str).eq(cat)]
            if len(sub) < 30:
                continue
            x = float(sub["omnicell_umap_1"].median())
            y = float(sub["omnicell_umap_2"].median())
            label = BROAD_SHORT.get(cat, cat)
            ax.text(
                x,
                y,
                label,
                fontsize=5.8,
                fontweight="bold",
                color=PALETTE["ink"],
                ha="center",
                va="center",
                path_effects=[pe.withStroke(linewidth=2.4, foreground="white", alpha=0.88)],
                zorder=5,
            )


def draw_panel_f(df: pd.DataFrame) -> None:
    out_df = df.copy()
    out_df.to_csv(SRC / "fig1f_allcell_omnicell_latent_atlas_source.csv", index=False)

    limits = umap_limits(out_df)
    fig = plt.figure(figsize=(9.2, 6.3))
    gs = fig.add_gridspec(
        4,
        5,
        left=0.035,
        right=0.80,
        top=0.82,
        bottom=0.055,
        wspace=0.06,
        hspace=0.23,
        height_ratios=[1.12, 1.12, 0.78, 0.78],
    )
    ax_class = fig.add_subplot(gs[0:2, 0:2])
    ax_region = fig.add_subplot(gs[0:2, 2:5])

    draw_umap(
        ax_class,
        out_df,
        "cell_class",
        BROAD_COLORS,
        BROAD_ORDER,
        "all cells by broad cell class",
        label_names=False,
        limits=limits,
    )
    draw_umap(
        ax_region,
        out_df,
        "region_modality",
        REGMOD_COLORS,
        REGMOD_ORDER,
        "all cells by brain region and modality",
        label_names=False,
        limits=limits,
    )

    mini_classes = [c for c in BROAD_ORDER if c in set(out_df["cell_class"])]
    mini_axes = []
    for i, cls in enumerate(mini_classes[:9]):
        r = 2 + i // 5
        c = i % 5
        ax = fig.add_subplot(gs[r, c])
        mini_axes.append(ax)
        ax.scatter(
            out_df["omnicell_umap_1"],
            out_df["omnicell_umap_2"],
            s=0.08,
            color="#D5DAE1",
            alpha=0.18,
            linewidths=0,
            rasterized=True,
        )
        sub = out_df[out_df["cell_class"].eq(cls)]
        ax.scatter(
            sub["omnicell_umap_1"],
            sub["omnicell_umap_2"],
            s=0.16,
            color=BROAD_COLORS.get(cls, "#C7C9CC"),
            alpha=0.78,
            linewidths=0,
            rasterized=True,
        )
        style_umap_axis(ax, limits, arrows=(i == 0))
        ax.set_title(f"{BROAD_SHORT.get(cls, cls)}  n={fmt_count(len(sub))}", fontsize=5.8, color=BROAD_COLORS.get(cls, PALETTE["ink"]), pad=1.5)

    fig.text(0.035, 0.955, "All-cell OmniCell latent atlas", ha="left", va="top", fontsize=10.5, fontweight="bold")
    fig.text(
        0.035,
        0.905,
        f"Representative cells/spots from cortex, prefrontal cortex and hippocampus; n = {len(out_df):,}; latest CPT checkpoint. Small multiples highlight each class over the shared atlas.",
        ha="left",
        va="top",
        fontsize=6.1,
        color=PALETTE["muted"],
    )

    broad_handles = [
        mpl.lines.Line2D([0], [0], marker="o", lw=0, markerfacecolor=BROAD_COLORS[c], markeredgewidth=0, markersize=4.2, label=BROAD_SHORT[c])
        for c in BROAD_ORDER
        if c in set(out_df["cell_class"])
    ]
    reg_handles = [
        mpl.lines.Line2D([0], [0], marker="o", lw=0, markerfacecolor=REGMOD_COLORS[c], markeredgewidth=0, markersize=4.2, label=c)
        for c in REGMOD_ORDER
        if c in set(out_df["region_modality"])
    ]
    leg1 = fig.legend(broad_handles, [h.get_label() for h in broad_handles], title="Cell class", loc="upper left", bbox_to_anchor=(0.805, 0.77), fontsize=5.8, title_fontsize=6.5, labelspacing=0.45, handletextpad=0.35)
    fig.add_artist(leg1)
    fig.legend(reg_handles, [h.get_label() for h in reg_handles], title="Region / modality", loc="upper left", bbox_to_anchor=(0.805, 0.36), fontsize=5.8, title_fontsize=6.5, labelspacing=0.45, handletextpad=0.35)

    save_panel(fig, FIG1 / "fig1f_allcell_omnicell_latent_atlas")


def write_contract() -> None:
    payload = {
        "core_conclusion": "Figure 1C/D/F show that the full 17.17M-cell NVU input space has broad cellular and study-axis coverage and that the latest OmniCell CPT checkpoint provides an interpretable all-cell latent atlas.",
        "panels": {
            "Figure 1C": "Full input-manifest broad cell-class composition across single-cell/snRNA and spatial transcriptomics.",
            "Figure 1D": "Full input-manifest study-axis coverage across modalities.",
            "Figure 1F": "Representative all-cell OmniCell embedding colored by cell class and region/modality.",
        },
        "source_tables": {
            "allcell_embedding": str(ALLCELL_SOURCE),
            "full_cell_composition": str(SRC / "fig1c_allcell_cell_class_composition_source.csv"),
            "full_cell_composition_by_dataset": str(SRC / "fig1c_allcell_cell_class_composition_dataset_source.csv"),
            "full_cell_composition_other_audit": str(SRC / "fig1c_allcell_cell_class_other_audit.csv"),
            "full_input_manifest": str(FULL_INPUT_SOURCE),
            "dataset_manifest": str(MANIFEST_SOURCE),
        },
    }
    (SRC / "fig1_cdf_panel_contract.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    FIG1.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    df = load_allcell()
    composition = load_full_cell_composition()
    draw_panel_c(composition)
    draw_panel_d()
    draw_panel_f(df)
    write_contract()


if __name__ == "__main__":
    main()

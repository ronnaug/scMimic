import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns


def plot_mimicry_results(
    adata: sc.AnnData,
    save_dir: str = ".",
    project_name: str = "Tumor_Cohort",
    celltype_col: str = "Celltype",
) -> None:
    """
    Generate publication-grade summary figures for MimicDetector outputs:
      1. Composite two-tier figure (Dual UMAPs + Donut chart of tumor plasticity).
      2. Confidence score distribution boxplot across decoded tumor subpopulations.

    Saves both raster (PNG, 300 DPI) and vector (PDF) files to `save_dir`.
    """
    print(f"[INFO] Generating plasticity landscape figures for {project_name}...")
    if save_dir != ".":
        os.makedirs(save_dir, exist_ok=True)

    # Ensure UMAP coordinates exist
    if "X_umap" not in adata.obsm:
        print("[INFO] 'X_umap' not found in adata.obsm. Computing neighborhood graph and UMAP...")
        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata)
        sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
        sc.tl.umap(adata, random_state=42)

    sc.set_figure_params(dpi=130, facecolor="white", vector_friendly=True)
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["text.color"] = "#2C3E50"
    plt.rcParams["axes.labelcolor"] = "#2C3E50"

    unique_verdicts = adata.obs["gsea_mimicry_verdict"].astype(str).unique()

    def _clean_label(verdict: str) -> str:
        clean = verdict.replace("Canonical: ", "").replace("De Novo Myeloid: ", "De Novo: ")
        for suffix in (" (in Epithelial)", " (masked as Macrophage)", " (masked as Fibroblast)",
                       " (masked as Endothelial)", " (masked as Dendritic cell)",
                       " (masked as Plasma cell)", " (masked as Mast)", " (Validated via CNV)"):
            clean = clean.replace(suffix, "")
        return clean

    translated_labels: Dict[str, str] = {v: _clean_label(v) for v in unique_verdicts}

    # Assign semantic color palette matched to updated pipeline verdicts
    palette: Dict[str, str] = {}
    for verdict in unique_verdicts:
        v_low = verdict.lower()
        if "hybrid mimicry" in v_low or "full phenotype switch" in v_low or "true mimicry" in v_low:
            if "endothelial" in v_low:
                palette[verdict] = "#CD6155"  # Muted Terracotta (Vasculogenic)
            elif "neuronal" in v_low or "neuroendocrine" in v_low:
                palette[verdict] = "#52BE80"  # Sage Emerald (Neuronal)
            elif "stromal" in v_low or "fibroblast" in v_low:
                palette[verdict] = "#E59866"  # Soft Apricot (EMT / Stromal)
            elif "myeloid" in v_low or "macrophage" in v_low or "dendritic" in v_low:
                palette[verdict] = "#D4AC0D"  # Warm Ochre (Myeloid)
            elif "lymphoid" in v_low or "t-cell" in v_low:
                palette[verdict] = "#5499C7"  # Steel Blue (Immune Camouflage)
            elif "erythroid" in v_low:
                palette[verdict] = "#EC7063"  # Soft Coral (Erythroid)
            else:
                palette[verdict] = "#F5CBA7"  # Sandy Gold (De Novo Lineage)
        elif "de-differentiation" in v_low or "stem" in v_low:
            palette[verdict] = "#AF7AC5"      # Soft Lavender (Cancer Stem Cells)
        elif "low regulon specificity" in v_low:
            palette[verdict] = "#AEB6BF"      # Neutral Slate Gray (Unclassified Tumor)
        elif "malignant" in v_low:
            palette[verdict] = "#5DADE2"      # Horizon Blue (Canonical Tumor)
        else:
            palette[verdict] = "#EAEDED"      # Subtle Light Gray (Diploid Microenvironment)

    adata.obs["_Plot_Verdict"] = adata.obs["gsea_mimicry_verdict"].astype(str).map(translated_labels)
    plot_palette = {translated_labels[k]: v for k, v in palette.items()}

    # ==========================================================================
    # FIGURE A: COMPOSITE TWO-TIER LANDSCAPE (UMAPS + DONUT CHART)
    # ==========================================================================
    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.28, wspace=0.25)

    ax_umap1 = fig.add_subplot(gs[0, 0])
    sc.pl.embedding(
        adata, basis="umap", color=celltype_col, ax=ax_umap1, show=False, frameon=False, title=""
    )
    ax_umap1.set_title(f"{project_name}: Baseline Cell Type Annotation", fontsize=13, weight="bold", pad=12)

    ax_umap2 = fig.add_subplot(gs[0, 1])
    sc.pl.embedding(
        adata,
        basis="umap",
        color="_Plot_Verdict",
        ax=ax_umap2,
        show=False,
        palette=plot_palette,
        frameon=False,
        legend_loc="right margin",
        legend_fontsize=9,
        title="",
    )
    ax_umap2.set_title(f"{project_name}: Decoded Plasticity Landscape", fontsize=13, weight="bold", pad=12)

    for ax in (ax_umap1, ax_umap2):
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)

    ax_pie = fig.add_subplot(gs[1, :])
    tumor_mask = ~adata.obs["gsea_mimicry_verdict"].astype(str).str.startswith("Healthy")
    tumor_counts = adata.obs.loc[tumor_mask, "_Plot_Verdict"].value_counts()

    if not tumor_counts.empty:
        # Group very small slices (< 1.5%) into "Other Minor Programs" to keep labels readable
        total_tumor = tumor_counts.sum()
        major_mask = (tumor_counts / total_tumor) >= 0.015
        major_counts = tumor_counts[major_mask]
        minor_sum = tumor_counts[~major_mask].sum()
        if minor_sum > 0:
            major_counts = pd.concat([major_counts, pd.Series({"Other Minor Programs": minor_sum})])
            plot_palette["Other Minor Programs"] = "#D5D8DC"

        pie_colors = [plot_palette.get(name, "#D5D8DC") for name in major_counts.index]

        wedges, _, autotexts = ax_pie.pie(
            major_counts,
            colors=pie_colors,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.80,
            wedgeprops=dict(width=0.32, edgecolor="white", linewidth=2.0),
            textprops={"fontsize": 9.5, "weight": "bold", "color": "#2C3E50"},
        )
        for autotext in autotexts:
            autotext.set_fontsize(8.5)

        ax_pie.legend(
            wedges,
            major_counts.index,
            title="Malignant Phenotypic States",
            title_fontproperties={"weight": "bold", "size": 11},
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=9.5,
        )
        ax_pie.set_title(
            f"Fractional Composition of Intratumoral Plasticity ({project_name})",
            fontsize=13,
            weight="bold",
            pad=18,
        )
    else:
        ax_pie.text(0.5, 0.5, "No malignant populations detected", ha="center", va="center", color="#7F8C8D")

    for spine in ax_pie.spines.values():
        spine.set_visible(False)

    main_path = os.path.join(save_dir, f"{project_name}_plasticity_landscape.png")
    plt.savefig(main_path, bbox_inches="tight", pad_inches=0.35, dpi=300)
    plt.savefig(main_path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.35, format="pdf")
    plt.show()
    plt.close(fig)

    # ==========================================================================
    # FIGURE B: ALGORITHMIC CONFIDENCE DISTRIBUTION BOXPLOT
    # ==========================================================================
    tumor_related = adata.obs.loc[tumor_mask].copy()
    if not tumor_related.empty:
        fig_box, ax_box = plt.subplots(figsize=(10, 5.2))
        order = tumor_related["_Plot_Verdict"].value_counts().index.tolist()
        sns.boxplot(
            data=tumor_related,
            x="_Plot_Verdict",
            y="mimicry_confidence",
            order=order,
            palette=plot_palette,
            hue="_Plot_Verdict",
            legend=False,
            width=0.45,
            ax=ax_box,
            boxprops=dict(alpha=0.85, edgecolor="#2C3E50", linewidth=1.0),
        )
        ax_box.set_xticks(range(len(order)))
        ax_box.set_xticklabels(order, rotation=30, ha="right", fontsize=9.5)
        ax_box.set_title(
            f"Algorithmic Confidence Distribution Across Decoded Clones ({project_name})",
            fontsize=12,
            weight="bold",
            pad=14,
        )
        ax_box.set_xlabel("")
        ax_box.set_ylabel("Calibrated Confidence Score (0.0 - 1.0)", fontsize=10.5)
        sns.despine(ax=ax_box)

        box_path = os.path.join(save_dir, f"{project_name}_confidence_boxplot.png")
        plt.savefig(box_path, bbox_inches="tight", pad_inches=0.3, dpi=300)
        plt.savefig(box_path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.3, format="pdf")
        plt.show()
        plt.close(fig_box)

    # Clean up temporary plotting column
    if "_Plot_Verdict" in adata.obs.columns:
        del adata.obs["_Plot_Verdict"]

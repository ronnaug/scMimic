import os
import re
from typing import List, Optional

import infercnvpy as icnv
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import xgboost as xgb
from joblib import Parallel, delayed


def run_mimic_detector_pipeline(
    adata: sc.AnnData,
    gtf_path: str,
    tf_list_path: str,
    celltype_col: str = "Celltype",
    n_jobs: int = 8,
    window_size: int = 30,
    reference_cell_types: Optional[List[str]] = None,
    disable_cnv_filter: bool = False,
    malignant_cell_types: Optional[List[str]] = None,
    verbose: bool = True,
) -> sc.AnnData:
    """
    Detect tumor lineage plasticity and phenotypic mimicry from single-cell RNA-seq data.

    Parameters
    ----------
    adata : AnnData
        Log-normalized single-cell expression object containing cell type annotations.
    gtf_path : str
        Path to a local GENCODE GTF annotation file for mapping genomic coordinates.
    tf_list_path : str
        Path to a newline-delimited text file of transcription factor gene symbols.
    celltype_col : str, default="Celltype"
        Column in `adata.obs` specifying cell type labels.
    n_jobs : int, default=8
        Number of CPU cores for parallel XGBoost regulon inference.
    window_size : int, default=30
        Sliding genomic window size (number of genes) for `infercnvpy`.
    reference_cell_types : list of str, optional
        Diploid control cell types used as baseline for CNV inference.
    disable_cnv_filter : bool, default=False
        If True, bypasses CNV thresholding (recommended for hematologic malignancies).
    malignant_cell_types : list of str, optional
        Expected baseline tumor lineage labels (e.g., `["Epithelial"]`).
    verbose : bool, default=True
        Whether to print progress logs.

    Returns
    -------
    AnnData
        Copy of input `adata` with inferred CNV scores, regulon activities, and mimicry verdicts.
    """
    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    if reference_cell_types is None:
        reference_cell_types = [
            "T cell", "B cell", "NK cell", "pDC",
            "Endothelial", "Fibroblast", "Macrophage", "Monocyte", "Plasma cell"
        ]

    if malignant_cell_types is None:
        malignant_cell_types = ["Epithelial"]

    _log("[INFO] Starting MimicDetector pipeline...")
    adata_res = adata.copy()

    if not adata_res.obs_names.is_unique:
        adata_res.obs_names_make_unique()

    if celltype_col not in adata_res.obs.columns:
        raise KeyError(f"Required column '{celltype_col}' not found in adata.obs.")

    # ==========================================================================
    # STEP 1: GENOMIC INSTABILITY & CNV PROFILING
    # ==========================================================================
    _log(f"[INFO] Parsing GENCODE gene coordinates from: {gtf_path}")
    gtf_cols = ["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
    chunks = []
    for chunk in pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        header=None,
        names=gtf_cols,
        dtype={"seqname": str, "feature": str, "start": int, "end": int, "attribute": str},
        chunksize=100000,
    ):
        chunks.append(chunk[chunk["feature"] == "gene"])
    df_gencode = pd.concat(chunks, axis=0)

    gene_name_pattern = re.compile(r'gene_name "([^"]+)"')

    def _extract_gene_name(attr_str: str) -> Optional[str]:
        match = gene_name_pattern.search(attr_str)
        return match.group(1) if match else None

    df_gencode["gene_name"] = df_gencode["attribute"].apply(_extract_gene_name)
    df_gencode["chromosome"] = "chr" + df_gencode["seqname"].str.replace("chr", "", regex=False).astype(str)
    df_gencode = df_gencode.dropna(subset=["gene_name"]).drop_duplicates(subset=["gene_name"])
    ref_coords = df_gencode.set_index("gene_name")[["chromosome", "start", "end"]]

    _log("[INFO] Mapping genes to chromosomal coordinates...")
    common_genes = [g for g in adata_res.var_names if g in ref_coords.index]
    if not common_genes:
        raise ValueError("Zero gene overlap between adata.var_names and GTF. Check gene symbol casing (e.g., Human vs Mouse).")

    # Slice 1: Initial coordinate mapping
    adata_res = adata_res[:, common_genes].copy()
    adata_res.var["chromosome"] = ref_coords.loc[adata_res.var_names, "chromosome"].astype(str)
    adata_res.var["start"] = ref_coords.loc[adata_res.var_names, "start"].astype(int)
    adata_res.var["end"] = ref_coords.loc[adata_res.var_names, "end"].astype(int)

    # Slice 2: Filter standard autosomes and chrX
    valid_chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    adata_res = adata_res[:, adata_res.var["chromosome"].isin(valid_chroms)].copy()

    # Sort genes by genomic position
    sort_df = adata_res.var.copy()
    sort_df["chrom_num"] = sort_df["chromosome"].str.replace("chr", "", regex=False).replace({"X": 23}).astype(int)
    sorted_genes = sort_df.sort_values(["chrom_num", "start"]).index

    # Slice 3: Final coordinate lock
    adata_res = adata_res[:, sorted_genes].copy()
    adata_res.var["chromosome"] = sort_df.loc[sorted_genes, "chromosome"].astype(str)
    adata_res.var["start"] = sort_df.loc[sorted_genes, "start"].astype(int)
    adata_res.var["end"] = sort_df.loc[sorted_genes, "end"].astype(int)

    # Validate reference populations
    available_refs = [r for r in reference_cell_types if r in adata_res.obs[celltype_col].unique()]
    if not available_refs:
        raise ValueError(f"None of the specified reference cell types {reference_cell_types} were found in '{celltype_col}'.")
    _log(f"[INFO] Using diploid reference populations: {available_refs}")

    # n_jobs=1 is kept intentionally in infercnvpy to avoid duplicating large matrices in memory
    _log("[INFO] Running chromosomal CNV inference (infercnvpy)...")
    icnv.tl.infercnv(
        adata_res,
        reference_key=celltype_col,
        reference_cat=available_refs,
        window_size=window_size,
        n_jobs=1,
        inplace=True,
    )

    if "X_cnv" not in adata_res.obsm:
        raise KeyError("Matrix 'X_cnv' missing from adata.obsm after infercnv execution.")

    raw_cnv_data = adata_res.obsm["X_cnv"]
    if hasattr(raw_cnv_data, "toarray"):
        cnv_matrix_full = raw_cnv_data.toarray()
    elif scipy.sparse.issparse(raw_cnv_data):
        cnv_matrix_full = raw_cnv_data.A
    else:
        cnv_matrix_full = np.asarray(raw_cnv_data)

    _log(f"[INFO] Extracted CNV matrix with shape: {cnv_matrix_full.shape}")
    _log("[INFO] Computing per-cell genomic instability index...")

    adata_res.obsm["X_cnv"] = cnv_matrix_full
    ref_mask = adata_res.obs[celltype_col].isin(available_refs).values

    # Calibrate baseline noise on the genomically stable lymphoid core if present
    strict_lymphoid = [r for r in ["T cell", "B cell", "NK cell", "pDC"] if r in available_refs]
    cal_mask = adata_res.obs[celltype_col].isin(strict_lymphoid).values if strict_lymphoid else ref_mask

    ref_baseline_profile = np.median(cnv_matrix_full[cal_mask], axis=0)
    individual_cnv_scores = np.sqrt(np.mean((cnv_matrix_full - ref_baseline_profile) ** 2, axis=1))
    adata_res.obs["cnv_score"] = individual_cnv_scores

    # Adaptive malignancy threshold (median + 1.5 * MAD)
    cal_scores = individual_cnv_scores[cal_mask]
    baseline_noise = float(np.median(cal_scores))
    mad_noise = float(np.median(np.abs(cal_scores - baseline_noise))) * 1.4826
    cnv_threshold = float(np.clip(baseline_noise + 1.5 * mad_noise, 0.0250, 0.0850))

    if np.isnan(cnv_threshold) or cnv_threshold < 0.005:
        cnv_threshold = 0.0250

    _log(
        f"[INFO] CNV baseline median: {baseline_noise:.4f} | MAD: {mad_noise:.4f} | "
        f"Malignancy threshold: {cnv_threshold:.4f}"
    )

    # Identify aneuploid cells across all non-reference populations to capture masked phenotypes
    is_malignant_by_default = adata_res.obs[celltype_col].isin(malignant_cell_types).values
    is_cancer_chaos = (~ref_mask) & (individual_cnv_scores >= cnv_threshold)

    is_dataset_normal = is_cancer_chaos.sum() == 0
    if "Cancer_Type" in adata_res.obs.columns:
        if adata_res.obs["Cancer_Type"].astype(str).str.contains("Normal|normal", case=False, na=False).mean() > 0.7:
            is_dataset_normal = True

    if disable_cnv_filter:
        is_dataset_normal = False
        is_cancer_chaos = is_malignant_by_default.copy()
        _log("[INFO] CNV filter manually disabled (hematologic malignancy mode).")

    if is_dataset_normal:
        _log("[INFO] Dataset classified as diploid normal tissue. Terminating pipeline early.")
        adata_res.obs["gsea_mimicry_verdict"] = "Healthy Tissue (Validated via CNV)"
        adata_res.obs["mimicry_best_signature"] = "None"
        adata_res.obs["mimicry_z_score"] = 0.0
        adata_res.obs["mimicry_confidence"] = 1.0
        return adata_res
    # ==========================================================================
    # STEP 2: DE NOVO REGULON RECONSTRUCTION (XGBOOST)
    # ==========================================================================
    _log("[INFO] Formatting expression matrix for de novo regulon inference...")
    X_dense = adata_res.X.toarray() if scipy.sparse.issparse(adata_res.X) else np.asarray(adata_res.X)
    X_dense = X_dense.astype(np.float32)
    ex_matrix_df = pd.DataFrame(X_dense, index=adata_res.obs_names, columns=adata_res.var_names)

    with open(tf_list_path, "r", encoding="utf-8") as f:
        db_tfs = [line.strip().upper() for line in f if line.strip()]

    gene_mapping = {g.upper(): g for g in ex_matrix_df.columns}
    active_tfs_upper = [tf for tf in db_tfs if tf in gene_mapping]
    tf_list = list(dict.fromkeys([gene_mapping[tf] for tf in active_tfs_upper]))

    if not tf_list:
        raise ValueError(f"No transcription factors from '{tf_list_path}' matched adata.var_names.")
    _log(f"[INFO] Active transcription factors matched in expression matrix: {len(tf_list)}")

    # Select informative target genes (top HVGs + active TFs) to accelerate GRN inference
    adata_hvg = adata_res.copy()
    n_top = min(3000, adata_hvg.n_vars)
    sc.pp.highly_variable_genes(adata_hvg, n_top_genes=n_top, flavor="seurat")
    hvg_genes = adata_hvg.var_names[adata_hvg.var["highly_variable"]].tolist()
    target_genes = list(dict.fromkeys(hvg_genes + tf_list))

    # Pre-slice TF predictor matrix to avoid DataFrame copying across parallel workers
    tf_indices = [ex_matrix_df.columns.get_loc(tf) for tf in tf_list]
    X_tf_all = X_dense[:, tf_indices]

    def _xgb_node(target_idx: int, target_name: str):
        y = X_dense[:, target_idx]
        if np.std(y) < 1e-6:
            return None

        # Exclude self-regulation when the target gene is itself a transcription factor
        if target_name in tf_list:
            self_pos = tf_list.index(target_name)
            mask = np.ones(len(tf_list), dtype=bool)
            mask[self_pos] = False
            X_pred = X_tf_all[:, mask]
            pred_names = [tf for i, tf in enumerate(tf_list) if i != self_pos]
        else:
            X_pred = X_tf_all
            pred_names = tf_list

        if not pred_names:
            return None

        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
            tree_method="hist",
        )
        model.fit(X_pred, y)

        results = []
        for tf_name, score in zip(pred_names, model.feature_importances_):
            if score > 0.001:
                results.append({"TF": tf_name, "target": target_name, "importance": float(score)})
        return results

    _log(f"[INFO] Inferring de novo regulons across {len(target_genes)} target genes on {n_jobs} cores...")
    target_tasks = [(ex_matrix_df.columns.get_loc(g), g) for g in target_genes]
    parallel_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_xgb_node)(t_idx, t_name) for t_idx, t_name in target_tasks
    )
    flattened_results = [item for sublist in parallel_results if sublist for item in sublist]

    if not flattened_results:
        raise RuntimeError("XGBoost failed to infer regulatory edges. Verify input normalization.")

    adjacencies = pd.DataFrame(flattened_results)
    _log(f"[INFO] De novo regulatory network built. Total edges: {len(adjacencies)}")

    # ==========================================================================
    # STEP 3: HYBRID SIGNATURE ASSEMBLY & WEIGHTED ACTIVITY SCORING
    # ==========================================================================
    tf_activity_dict = {}

    # Filter valid target genes and sort regulatory edges by feature importance
    valid_adj = adjacencies[adjacencies["target"].isin(ex_matrix_df.columns)].sort_values(
        "importance", ascending=False
    )

    for tf, group in valid_adj.groupby("TF", sort=False):
        top_edges = group.head(50)
        if len(top_edges) >= 5:
            weights = top_edges["importance"].values
            weights = weights / weights.sum()
            tf_activity_dict[tf] = ex_matrix_df[top_edges["target"].tolist()].values @ weights

    tf_activity_matrix = pd.DataFrame(tf_activity_dict, index=adata_res.obs_names)

    _log("[INFO] Extracting lineage-specific de novo TF signatures...")
    tf_by_celltype = tf_activity_matrix.groupby(adata_res.obs[celltype_col], observed=True).mean()
    stress_genes = {"JUN", "JUNB", "FOS", "FOSB", "EGR1", "ATF3"}

    baseline_lymphoid_refs = [r for r in ["T cell", "NK cell"] if r in tf_by_celltype.index]
    if not baseline_lymphoid_refs:
        baseline_lymphoid_refs = tf_by_celltype.index.tolist()
    lymphoid_baseline_profile = tf_by_celltype.loc[baseline_lymphoid_refs].mean()

    myeloid_keywords = ("macrophage", "monocyte", "dendritic", "myeloid", "microglia")

    hybrid_signatures = {}
    for cell_type in tf_by_celltype.index:
        if cell_type in malignant_cell_types or cell_type in baseline_lymphoid_refs:
            continue
        specificity = tf_by_celltype.loc[cell_type] - lymphoid_baseline_profile
        top_de_novo = (
            specificity.dropna()
            .drop(labels=list(stress_genes), errors="ignore")
            .sort_values(ascending=False)
            .head(4)
            .index.tolist()
        )
        if top_de_novo:
            ct_lower = str(cell_type).lower()
            if any(k in ct_lower for k in myeloid_keywords):
                hybrid_signatures[f"De Novo Myeloid: {cell_type}"] = top_de_novo
            else:
                hybrid_signatures[f"De Novo: {cell_type}"] = top_de_novo

    # ==========================================================================
    # CANONICAL CROSS-LINEAGE MIMICRY SIGNATURES
    # ==========================================================================
    canonical_signatures = {
        "Canonical: Myeloid (Macrophage/Monocyte)": [
            "SPI1", "SPIB", "SPIC", "IRF8", "CEBPA", "CEBPB", "MAF", "MAFB",
            "RUNX1", "MYO1F", "CSF1R", "CD68", "LYZ", "ITGAM", "AIF1"
        ],
        "Canonical: Stromal (Fibroblast/EMT)": [
            "TWIST1", "TWIST2", "SNAI1", "SNAI2", "ZEB1", "ZEB2", "PRRX1", "PRRX2",
            "GLI1", "GLI2", "SMAD2", "SMAD3", "SMAD4", "FAP", "ACTA2", "COL1A1", "VIM"
        ],
        "Canonical: Endothelial (Vasculogenic)": [
            "ERG", "FLI1", "ETS1", "ETS2", "ELK1", "SOX17", "SOX18", "TAL1",
            "KDR", "PECAM1", "VWF", "CD34", "ENG", "NOS3"
        ],
        "Canonical: Lymphoid (T-cell Camouflage)": [
            "TBX21", "GATA3", "RORC", "FOXP3", "STAT1", "STAT4", "STAT6", "RUNX3",
            "CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "IL2RA"
        ],
        "Canonical: Stemness (Cancer Stem Cells)": [
            "POU5F1", "NANOG", "SOX2", "MYC", "KLF4", "LIN28A", "SALL4", "PROM1",
            "ALDH1A1", "NES", "BMI1", "KIT", "CD44"
        ],
        "Canonical: Neuronal/Neuroendocrine": [
            "ASCL1", "NEUROD1", "NEUROD2", "SOX10", "POU3F2", "POU3F3", "INSM1",
            "SYP", "CHGA", "NCAM1", "TUBB3", "MAP2", "OLIG2"
        ],
        "Canonical: Erythroid/Heme": [
            "GATA1", "GATA2", "KLF1", "TAL1",
            "ALAS2", "FECH", "SLC25A37", "TFRC", "GYPA", "GYPB"
        ],
    }
    hybrid_signatures.update(canonical_signatures)

    # ==========================================================================
    # HYBRID ACTIVITY SCORING (TF REGULON ACTIVITY + MARKER EXPRESSION)
    # ==========================================================================
    _log("[INFO] Computing hybrid regulon and surface marker activity scores...")
    gsea_scores_dict = {}
    tf_cols_set = set(tf_activity_matrix.columns)
    ex_cols_set = set(ex_matrix_df.columns)

    for sig_name, genes in hybrid_signatures.items():
        arrays = []
        for g in genes:
            if g in tf_cols_set:
                arrays.append(tf_activity_matrix[g].values)
            elif g in ex_cols_set:
                arrays.append(ex_matrix_df[g].values)

        if arrays:
            gsea_scores_dict[sig_name] = np.mean(np.column_stack(arrays), axis=1)
        else:
            gsea_scores_dict[sig_name] = np.zeros(adata_res.n_obs, dtype=np.float32)

    gsea_scores_df = pd.DataFrame(gsea_scores_dict, index=adata_res.obs_names)

    # ==========================================================================
    # STEP 4: CROSS-PHENOTYPE CLASSIFICATION & CONFIDENCE CALIBRATION
    # ==========================================================================
    _log("[INFO] Screening cell populations for cross-lineage mimicry...")

    naive_types = adata_res.obs[celltype_col].astype(str).values
    verdicts = naive_types.copy()
    best_signatures_arr = np.full(adata_res.n_obs, "None", dtype=object)
    best_z_scores_arr = np.zeros(adata_res.n_obs, dtype=float)
    confidences = np.zeros(adata_res.n_obs, dtype=float)

    # Calibrate thresholds strictly on the baseline malignant tumor pool
    # to prevent abundant stromal/immune populations from inflating cutoffs
    malignant_pool_mask = is_cancer_chaos & adata_res.obs[celltype_col].isin(malignant_cell_types).values
    target_pool = (
        gsea_scores_df.loc[malignant_pool_mask]
        if malignant_pool_mask.sum() >= 10
        else gsea_scores_df.loc[is_cancer_chaos]
    )


    signature_thresholds = {
        sig: float(np.percentile(target_pool[sig].values, 92)) for sig in hybrid_signatures.keys()
    }

    pool_means = target_pool.mean()
    adaptive_eps = max(float(target_pool.std().median()) * 0.1, 1e-3)
    pool_stds = target_pool.std() + adaptive_eps

    def _find_matching_signature(cell_type: str, signatures) -> Optional[str]:
        ct_low = cell_type.lower()
        for sig in signatures:
            sig_low = sig.lower()
            if any(k in sig_low for k in ("myeloid", "macrophage", "monocyte", "dendritic")) and any(
                k in ct_low for k in ("macrophage", "monocyte", "myeloid", "dendritic")
            ):
                return sig
            if any(k in sig_low for k in ("stromal", "fibroblast")) and any(
                k in ct_low for k in ("fibroblast", "stromal", "mesenchymal")
            ):
                return sig
            if "endothelial" in sig_low and "endothelial" in ct_low:
                return sig
            if any(k in sig_low for k in ("lymphoid", "t-cell")) and any(
                k in ct_low for k in ("t cell", "t-cell", "nk")
            ):
                return sig
            if "b cell" in ct_low and any(k in sig_low for k in ("b-cell", "b cell", "plasma")):
                return sig
            if ct_low in sig_low:
                return sig
        return None

    sig_names = list(hybrid_signatures.keys())
    scores_mat = gsea_scores_df[sig_names].values
    thresh_vec = np.array([signature_thresholds[s] for s in sig_names])
    z_mat = (scores_mat - pool_means[sig_names].values) / pool_stds[sig_names].values
    cnv_vals = adata_res.obs["cnv_score"].values
    non_stem_indices = [i for i, s in enumerate(sig_names) if "Stemness" not in s]

    for idx in range(adata_res.n_obs):
        naive_type = naive_types[idx]
        has_cnv_chaos = is_cancer_chaos[idx]
        cnv_val = cnv_vals[idx]

        # Scenario 1: Diploid genome -> Normal microenvironment or non-malignant cell
        if not has_cnv_chaos:
            verdicts[idx] = f"Healthy {naive_type} (Validated via CNV)"
            confidences[idx] = 1.0 - (cnv_val / (cnv_threshold + 1e-6))
            continue

        # Scenario 2: Aneuploid tumor cell -> Evaluate active transcriptional programs
        cell_scores = scores_mat[idx]
        cell_z = z_mat[idx]
        #active_indices = np.flatnonzero((cell_scores >= thresh_vec) & (cell_scores > 0))
        active_indices = np.flatnonzero((cell_scores >= thresh_vec) & (cell_z >= 1.5) & (cell_scores > 0))


        if len(active_indices) > 0:
            tissue_active = [i for i in active_indices if i in non_stem_indices]
            candidates = tissue_active if len(tissue_active) > 0 else active_indices
            best_idx = candidates[np.argmax(cell_z[candidates])]
            is_sig_stat_active = True
        else:
            best_idx = int(np.argmax(cell_z))
            is_sig_stat_active = False

        best_signature = sig_names[best_idx]
        best_z = float(cell_z[best_idx])
        best_signatures_arr[idx] = best_signature
        best_z_scores_arr[idx] = best_z

        native_sig = _find_matching_signature(naive_type, sig_names)
        is_annotated_malignant = naive_type in malignant_cell_types
        is_cross_phenotype = (best_signature != native_sig) if is_annotated_malignant else True

        sig_prob = float(1.0 / (1.0 + np.exp(-best_z)))

        if is_sig_stat_active and is_cross_phenotype:
            if "Stemness" in best_signature:
                verdicts[idx] = "Tumor De-differentiation (Cancer Stem Cells)"
            elif not is_annotated_malignant:
                verdicts[idx] = f"Full Phenotype Switch: {best_signature} (masked as {naive_type})"
            else:
                verdicts[idx] = f"Hybrid Mimicry: {best_signature} (in {naive_type})"
            confidences[idx] = sig_prob
        elif is_sig_stat_active and not is_cross_phenotype:
            verdicts[idx] = f"Malignant {naive_type} (Standard Tumor)"
            confidences[idx] = float(cnv_val / (cnv_val + cnv_threshold))
        else:
            verdicts[idx] = f"Malignant {naive_type} (Low Regulon Specificity)"
            confidences[idx] = 0.5

    # Store results in AnnData container
    adata_res.obs["gsea_mimicry_verdict"] = verdicts
    adata_res.obs["mimicry_best_signature"] = best_signatures_arr
    adata_res.obs["mimicry_z_score"] = best_z_scores_arr
    adata_res.obs["mimicry_confidence"] = np.clip(confidences, 0.0, 1.0)
    adata_res.obsm["gsea_scores"] = gsea_scores_df

    if verbose:
        print("\n[SUMMARY] Population verdict distribution:")
        print(adata_res.obs["gsea_mimicry_verdict"].value_counts())
        print("\n[INFO] MimicDetector pipeline completed successfully.")

    return adata_res



# scMimic

**scMimic** is a single-cell RNA-seq (scRNA-seq) analysis pipeline designed to detect tumor lineage plasticity, transdifferentiation, and cross-lineage phenotypic mimicry.

By integrating chromosomal copy-number variation (CNV) inference with *de novo* transcription factor (TF) regulon reconstruction and hybrid signature scoring, scMimic separates diploid microenvironment cells from aneuploid tumor cells adopting non-native transcriptional programs (e.g., myeloid, stromal/EMT, vasculogenic, lymphoid, neuronal, or cancer stem-like states).

---

## Key Features

- **Per-Cell Genomic Instability Profiling:** Infers chromosomal CNV profiles (`infercnvpy`) relative to diploid reference populations and computes a robust per-cell root-mean-square deviation (RMSD) score with adaptive median + MAD thresholding.
- **De Novo Regulon Reconstruction:** Uses parallelized gradient-boosted decision trees (`XGBoost`) to infer dataset-specific TF-target regulatory networks and weights top target genes by feature importance.
- **Hybrid Lineage Scoring:** Combines *de novo* data-driven TF regulons with curated canonical lineage drivers and surface markers to overcome scRNA-seq dropout noise.
- **Cross-Phenotype Classification:** Distinguishes between standard malignant cells, **Hybrid Mimicry** (tumor cells retaining baseline lineage markers while co-activating a foreign program), and **Full Phenotype Switch** (aneuploid cells masked as normal stromal or immune cell types).
- **Publication-Grade Visualization:** Generates dual-panel UMAP embeddings, fractional plasticity donut charts, and calibrated confidence distributions.

---

## Reference Data Setup

scMimic does **not** require multi-gigabyte `cisTarget` ranking databases (`.feather`) or motif tables (`.tbl`). Only two lightweight reference files are needed:

1. **GENCODE Human Gene Annotation (`gencode.v50.basic.annotation.gtf`):** Used to map gene symbols to chromosomal coordinates for `infercnvpy`.
2. **Human Transcription Factor List (`hs_hgnc_tfs.txt`):** Plain-text list of human TF gene symbols (from Aerts Lab / pySCENIC resources).

Download and unpack both files via terminal:

```bash
mkdir -p references && cd references

# 1. Download and extract GENCODE release 50 basic gene annotation (GRCh38)
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/gencode.v50.basic.annotation.gtf.gz
gunzip gencode.v50.basic.annotation.gtf.gz

# 2. Download raw human transcription factor list
wget -O hs_hgnc_tfs.txt https://github.com/aertslab/pySCENIC/blob/master/resources/hs_hgnc_tfs.txt
```


## Installation



Clone the repository and install the required dependencies:

```bash
git clone https://github.com
cd scMimic
pip install -r requirements.txt
```

---

## Input Requirements

1. **Single-cell dataset (`AnnData` / `.h5ad`):**
   - `adata.X`: Log-normalized gene expression matrix (`sc.pp.normalize_total` followed by `sc.pp.log1p`).
   - `adata.obs['Celltype']`: Baseline cell type annotations (configurable via `celltype_col`).
   - `adata.var_names`: Standard HGNC gene symbols (matching the GTF annotation).
2. **Genomic annotation (`.gtf`):**
   - A local GENCODE basic annotation GTF file (e.g., `gencode.v50.basic.annotation.gtf`) for mapping genes to chromosomal coordinates.
3. **Transcription factor list (`.txt`):**
   - A newline-delimited text file containing human transcription factor gene symbols (e.g., `hs_hgnc_tfs.txt` from pySCENIC resources).

---

## Quick Start

```python
import scanpy as sc
from scMimic import run_mimic_detector_pipeline
from visualization import plot_mimicry_results

# 1. Load preprocessed (log1p-normalized) single-cell data
adata = sc.read_h5ad("example_tumor_cohort.h5ad")

# 2. Run the scMimic pipeline
adata_results = run_mimic_detector_pipeline(
    adata=adata,
    gtf_path="gencode.v50.basic.annotation.gtf",
    tf_list_path="hs_hgnc_tfs.txt",
    celltype_col="Celltype",
    malignant_cell_types=["Epithelial"],
    n_jobs=8,
)

# 3. Inspect decoded plasticity verdicts
print(adata_results.obs["gsea_mimicry_verdict"].value_counts())

# 4. Generate publication-ready figures (PNG & PDF)
plot_mimicry_results(
    adata=adata_results,
    save_dir="./results",
    project_name="Cohort_Analysis",
    celltype_col="Celltype",
)
```

### Hematologic Malignancies Mode

For liquid tumors (e.g., leukemias, lymphomas) with minimal large-scale chromosomal CNVs, disable the CNV threshold filter and specify the malignant compartment explicitly:

```python
adata_results = run_mimic_detector_pipeline(
    adata=adata,
    gtf_path="gencode.v50.basic.annotation.gtf",
    tf_list_path="hs_hgnc_tfs.txt",
    disable_cnv_filter=True,
    malignant_cell_types=["B cell"],
    n_jobs=8,
)
```

---

## Output Annotations

The pipeline returns an `AnnData` object with the following added fields:

- `adata.obs['cnv_score']`: Per-cell RMSD from the diploid baseline CNV profile.
- `adata.obs['gsea_mimicry_verdict']`: Final phenotypic classification label.
- `adata.obs['mimicry_best_signature']`: Highest-scoring cross-lineage transcriptional program.
- `adata.obs['mimicry_z_score']`: Z-score of the winning program relative to the malignant pool.
- `adata.obs['mimicry_confidence']`: Sigmoid-calibrated confidence score (`0.0` to `1.0`).
- `adata.obsm['X_cnv']`: Chromosomal CNV matrix inferred by `infercnvpy`.
- `adata.obsm['gsea_scores']`: Activity matrix across all *de novo* and canonical signatures.

---

## License

This project is released under the MIT License.

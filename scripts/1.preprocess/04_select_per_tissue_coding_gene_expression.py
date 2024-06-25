import os

from load_and_process_metadata_gtex import load_and_process_metadata_gtex
from process_rna_data import load_and_process_coding_gene_expression_data

metadata = load_and_process_metadata("../data/metadata/gtex_v8_metadata_full.tab")
for tissue in ["Lung", "Ovary", "Prostate", "Colon - Transverse"]:
    if tissue == "Ovary":
        remove_chrX = False
        remove_chrY = True
    elif tissue == "Prostate":
        remove_chrX = False
    else:
        remove_chrX = True
        remove_chrY = False

    metadata_tissue = metadata[metadata["tissue"] == tissue]
    tissue_ids = list(metadata_tissue.index)
    load_and_process_coding_gene_expression_data(gexp_file="../data/gene_expression/GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_gene_tpm.gct.gz",
                                        gtf_file="../data/gene_expression/gencode.v26.GRCh38.genes.gtf",
                                        select_sample_ids=tissue_ids,
                                        outfile_parquet=os.path.join("../data/gene_expression/tissue_gene_expression", f"X_coding_{tissue.lower()}_log2.parquet"),
                                        outfile_csv=os.path.join("../data/gene_expression/tissue_gene_expression", f"X_coding_{tissue.lower()}_log2.csv"),
                                        remove_chrX=remove_chrX)
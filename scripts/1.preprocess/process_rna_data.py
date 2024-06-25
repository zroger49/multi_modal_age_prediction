import gc

import numpy as np
import pandas as pd

from gtfparse import read_gtf

def load_data(gexp_path: str) -> pd.DataFrame:
    gexp_df = pd.read_csv(gexp_path, sep='\t', skiprows=2, header=0, index_col=0)
    return gexp_df


def get_coding_gene_features(gtf_path: str, remove_chrX: bool=True, remove_chrY: bool=True) -> pd.Series:
    gene_features = read_gtf(gtf_path, usecols=["seqname", "gene_id", "gene_type"], result_type="pandas")
    coding_gene_features = gene_features[(gene_features['gene_type'] == 'protein_coding') |
                                         # lincRNA = Long intergenic noncoding RNAs
                                         (gene_features['gene_type'] == 'lincRNA')]

    # Remove sexual chromosomes and mitochondria -- in some cases, we may wish to keep ChrX (e.g. ovary)
    # and ChrY (e.g. prostate)
    coding_gene_features_filtered = coding_gene_features[~(coding_gene_features['seqname'] == "chrM")]["gene_id"]
    if remove_chrX:
        coding_gene_features_filtered = coding_gene_features[~((coding_gene_features['seqname'] == "chrX"))]["gene_id"]
    if remove_chrY:
        coding_gene_features_filtered = coding_gene_features[~(coding_gene_features['seqname'] == "chrY")]["gene_id"]

    return coding_gene_features_filtered


def filter_by_sample_expression(gexp_df: pd.DataFrame, tpm_min=1, tpm_min_prop=0.2):
    sample_expr_mask = gexp_df.apply(lambda gene_tpm: sum(gene_tpm >= tpm_min) >= tpm_min_prop*gene_tpm.shape[0],
                                     axis=1)
    gexp_df_filtered = gexp_df.loc[sample_expr_mask, :]
    return gexp_df_filtered


def load_and_process_coding_gene_expression_data(gexp_file:str, gtf_file:str, select_sample_ids:list=None,
                                                 outfile_parquet: str="X_coding.parquet",
                                                 outfile_csv: str="X_coding.csv",
                                                 remove_chrX: bool=True,
                                                 remove_chrY: bool=True) -> pd.DataFrame:
    print("Load data")
    X = load_data(gexp_file)
    print(X.shape)
    gc.collect()
    print("drop")
    X = X.drop(["transcript_id(s)"], axis=1)
    gc.collect()
    print(X.shape)

    print("select coding genes")
    coding_gene_features = get_coding_gene_features(gtf_file, remove_chrX=remove_chrX, remove_chrY=remove_chrY)
    gc.collect()
    X_coding = X.loc[X.index.isin(coding_gene_features), :]  # it has to be like this or we get dupe indices
    del(coding_gene_features, X)
    gc.collect()
    print(X_coding.shape)

    print("Converting to sample ids")
    # convert to sample ids
    X_coding = X_coding.rename({col: "-".join(col.split("-")[0:3]) for col in X_coding.columns}, axis=1)
    print(X_coding.shape)

    # select ids, if any (usually tissues)
    if select_sample_ids:
        print("Selecting ids...")
        X_coding = X_coding.loc[:, select_sample_ids]
    print(X_coding.shape)

    print("Filter by minimum TPM of 1 in at least 20% of samples")
    X_coding_filtered = filter_by_sample_expression(X_coding)
    del(X_coding)
    gc.collect()
    print(X_coding_filtered.shape)

    print("Transpose")
    X_coding_transp = X_coding_filtered.transpose()
    del(X_coding_filtered)
    X_coding_transp = X_coding_transp.rename_axis("tissue_sample_id")
    gc.collect()
    print(X_coding_transp.shape)

    print("Log2(TPM+1)...")
    X_coding_log = X_coding_transp.map(lambda x: np.log2(x+1))
    del(X_coding_transp)
    gc.collect()
    print(X_coding_log.shape)

    print("write to file")
    if outfile_csv:
        print("csv")
        X_coding_log.to_csv(outfile_csv)
    if outfile_parquet:
        print("parquet")
        X_coding_log.to_parquet(outfile_parquet)

    return X_coding_log
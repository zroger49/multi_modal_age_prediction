import pandas as pd
import numpy as np

from joblib import Memory
memory = Memory('cache')


@memory.cache
def load_lung_data() -> pd.DataFrame:
    return pd.read_csv("data/methylation_lung.csv")

@memory.cache
def load_colon_data() -> pd.DataFrame:
    return pd.read_csv("data/methylation_colon.csv")

@memory.cache
def load_ovary_data() -> pd.DataFrame:
    return pd.read_csv("data/methylation_ovary.csv")

@memory.cache
def load_prostate_data() -> pd.DataFrame:
    return pd.read_csv("data/methylation_prostate.csv")


def split_in_train_test(meth, age_data, test_set):
    """Split the dataset into train and test"""
    #Metadata
    metadata_test = age_data.loc[age_data["tissue_sample_id"].isin(test_set["sample_id"])]
    metadata_train = age_data.loc[~age_data["tissue_sample_id"].isin(test_set["sample_id"])]
    #Age data
    y_test = list(metadata_test["AGE"])
    y_train = list(metadata_train["AGE"])
    #Methylation data
    meth_t = meth.transpose()
    X_train = meth_t.loc[metadata_train.Sample_ID]
    X_test = meth_t.loc[metadata_test.Sample_ID]

    return(X_train, X_test, y_train, y_test)

def load_folds(tissue, num_folds = 5): 
    num_folds = 5
    recreated_folds = []

    for i in range(num_folds):
        # Load train and test data from CSV files
        train_data = pd.read_csv(f'results/3.feature_selection/{tissue}/fold_{i}_train.csv', index_col=0)
        test_data = pd.read_csv(f'results/3.feature_selection/{tissue}/fold_{i}_test.csv', index_col=0)

        # Append the train and test index to the folds list
        recreated_folds.append([train_data.index, test_data.index])
    return recreated_folds

def filter_M_and_XY_probes(meth_data, probe_info, tissue): 
    probe_info = probe_info[probe_info["CHR_hg38"].notna()] #Remove NA probes in the CHR_hg38 genome
    if tissue == "lung" or tissue == "colon_transverse": 
        # Filter out probes in the X, Y 
        chrxy_probes = probe_info[(probe_info['CHR_hg38'] == 'chrX') | (probe_info['CHR_hg38'] == 'chrY')]['Name'].tolist()
        meth_data = meth_data[~meth_data.index.isin(chrxy_probes)]
        
    # Filter ou probes in the M chromossome
    chrM_probes = probe_info[(probe_info['CHR_hg38'] == 'chrM')]['Name'].tolist()
    meth_data = meth_data[~meth_data.index.isin(chrM_probes)]

    return meth_data

def convert_beta_to_m(meth):
    """Convert beta values to M values"""
    # Assuming all columns in the DataFrame are beta values
    beta_columns = meth.columns

    # Create a new DataFrame for M values
    m_dataframe = pd.DataFrame()


    for beta_column in beta_columns:
        if beta_column == "probe": 
            m_dataframe["probe"] = meth[beta_column]
        else: 
            m_dataframe[beta_column] = np.log2(meth[beta_column] / (1 - meth[beta_column]))

    return m_dataframe.copy()
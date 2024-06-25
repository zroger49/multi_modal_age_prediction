"""Run Elastic Net Regression on each tissue on all features. Save features selected in each fold"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import ElasticNet

from mlflow import log_metric

from common_functions import (load_lung_data, 
                              load_colon_data, 
                              load_ovary_data, 
                              load_prostate_data, 
                              split_in_train_test,
                              convert_beta_to_m,
                              load_folds,
                              filter_M_and_XY_probes)


def load_folds_multi_modal(tissue, num_folds = 5): 
    num_folds = 5
    recreated_folds = []

    for i in range(num_folds):
        # Load train and test data from CSV files
        train_data = pd.read_csv(f'results/3.feature_selection_multimodal/{tissue}/fold_{i}_train.csv', index_col=0)
        test_data = pd.read_csv(f'results/3.feature_selection_multimodal/{tissue}/fold_{i}_test.csv', index_col=0)

        # Append the train and test index to the folds list
        recreated_folds.append([train_data.index, test_data.index])
    return recreated_folds

SEED = 42

tissue_param = {"lung": {
                    "name": "Lung", 
                    "name_folder": "Lung",
                    "age_data": r"metadata/lung_annotation_meth.csv",
                    "test_data": "metadata/lung_test_metadata.csv"
                }, 
                "ovary": {
                    "name": "Ovary", 
                    "name_folder": "Ovary",
                    "age_data": r"metadata/ovary_annotation_meth.csv",
                    "test_data": "metadata/ovary_test_metadata.csv"
                },
            }


tissues_to_run = ["lung", "ovary"]

# Load general metadata
metadata = pd.read_csv(r"metadata/eGTExDNA_Pierce_Jan18.09-11-2021.tsv", sep = "\t")
#Load probe info
probe_info = pd.read_csv(r"metadata/methylation_epic_v1.0b5.csv")
probe_info = probe_info[probe_info["CHR_hg38"].notna()] #Remove NA probes in the CHR_hg38 genome


for tissue in tissues_to_run:
    print("Running Analysis for ", tissue) 

    if tissue == "lung":
        meth_data = load_lung_data()
    elif tissue == "ovary": 
        meth_data = load_ovary_data()
    else: 
        print("Error.. No tissue with name %s", tissue)
        next

    #Transform into M values 
    meth_data = convert_beta_to_m(meth_data)
    
    ## Metadata processing
    #Subset metadata
    tissue_name = tissue_param[tissue]["name"]
    metadata_tissue = metadata[metadata["Tissue Site Detail"] == tissue_name]

    #Merge with age data
    age_data = tissue_param[tissue]["age_data"]
    age_data = pd.read_csv(age_data)
    
    # Filter for subject with age data
    samples_with_age = age_data.Sample_ID
    colums = list(samples_with_age)
    colums.append("probe") #Add probe data
    meth_data = meth_data[colums]
    meth_data = meth_data.set_index("probe")

    # Filter out M and XY probes
    meth_data = filter_M_and_XY_probes(meth_data, probe_info, tissue)

    # Load test data
    test_data = tissue_param[tissue]["test_data"]
    test_set = pd.read_csv(test_data)
    
    #Filter for complete samples (with all data types)
    multi_modal_table = "metadata/sample_ids_multiomics_updated_tl_data.csv"
    multi_modal_table = pd.read_csv(multi_modal_table)

    filtered_multi_modal_table = multi_modal_table[multi_modal_table['tissue'] == tissue_name]
    complete_samples = filtered_multi_modal_table[(filtered_multi_modal_table["metadata"]) == 1 & (filtered_multi_modal_table["gene_expression"]) & (filtered_multi_modal_table["metilation"]) &  (filtered_multi_modal_table["telemore"])]
    age_data_multi_modal = age_data.loc[age_data["tissue_sample_id"].isin(complete_samples["sample_id"])]

    #Split in Train and test data
    print("Spliting in train and test data..")
    X_train, X_test, y_train, y_test = split_in_train_test(meth_data, age_data_multi_modal, test_set)
    
    #Load folds for cross validation
    n_folds = 5
    folds = load_folds_multi_modal(tissue, num_folds=n_folds)

    ## Model Training ##
    pipe = Pipeline([
        ("QuantileTransformer", QuantileTransformer(n_quantiles = 100, random_state=SEED, output_distribution = "normal")),
        ("elasticNet", ElasticNet())
    ])

    print("Computing cross-validation scores...")

    cv_scores = {}
    idx = 0

    # Load methylation exclusive samples 
    exclusive_methylation_samples = pd.read_csv(f'results/3.feature_selection_multimodal/{tissue}/exclusive_methylation_train.csv', index_col=0)
    meth_t = meth_data.transpose()
    X_exclusive = meth_t.loc[exclusive_methylation_samples.Sample_ID]
    y_exclusive = exclusive_methylation_samples.AGE.tolist()

    for cv_train_idx, cv_valid_idx in folds: 
        print(f"Fitting model fold {idx}")
        X_fold_train, X_fold_valid = X_train.iloc[cv_train_idx], X_train.iloc[cv_valid_idx]
        y_fold_train, y_fold_valid = np.array(y_train)[cv_train_idx], np.array(y_train)[cv_valid_idx]
        
        X_fold_train = pd.concat([X_fold_train, X_exclusive], axis = 0)
        y_fold_train = np.concatenate([y_fold_train, np.array(y_exclusive)])
                
        pipe = Pipeline([
            ("QuantileTransformer", QuantileTransformer(n_quantiles = 100, random_state=SEED, output_distribution = "normal")),
            ("elasticNet", ElasticNet())
        ])

        pipe.fit( 
            X_fold_train,
            y_fold_train,
        )

        #Save coefficients
        coef = pipe[1].coef_
        probes = meth_data.index
        model_coef =  pd.DataFrame({"probe":probes, "coef" : coef})

        model_coef.to_csv(f"results/3.feature_selection_multimodal/results/{tissue_name}/fold_{idx}/linear_model_coef.csv")
        idx += 1

                    
    print("Fiting whole model")
    
    X_train = pd.concat([X_train, X_exclusive], axis = 0)
    y_train = np.concatenate([y_train, np.array(y_exclusive)])
        
    pipe.fit(X_train, y_train)

    #Save coefficients
    coef = pipe[1].coef_
    probes = meth_data.index
    model_coef =  pd.DataFrame({"probe":probes, "coef" : coef})

    model_coef.to_csv(f"results/3.feature_selection_multimodal/results/{tissue_name}/train/linear_model_coef.csv")

    print("Finished %s", tissue)
    
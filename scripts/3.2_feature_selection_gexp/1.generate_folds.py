import os

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

SEED = 42

tissue_param = {"lung": {
                "name": "Lung", 
                "gene_exp_data": "data/X_coding_lung_log2.csv",
                "age_data": r"metadata/gene_expression_metadata/metadata_lung.tsv",
                "test_data": "metadata/lung_test_metadata.csv"
            }, 
            "ovary": {
                "name": "Ovary", 
                "gene_exp_data": "data/X_coding_ovary_log2.csv",
                "age_data": r"metadata/gene_expression_metadata/metadata_ovary.tsv",
                "test_data": "metadata/ovary_test_metadata.csv"
            }
        }

from common_functions import (load_gexp_data,
                              split_in_train_test
                              )


tissues_to_run = ["lung","ovary"]


for tissue in tissues_to_run:
    res_dir = f"results/3.feature_selection_gene_expresion/{tissue}"
    if not os.path.exists(res_dir): 
        os.mkdir(res_dir)

    gexp_data = load_gexp_data(tissue_param[tissue]["gene_exp_data"])
    gexp_data.index = gexp_data.tissue_sample_id
    gexp_data.drop(columns="tissue_sample_id", axis=1,  inplace=True)
    # Load age data
    age_data = tissue_param[tissue]["age_data"]
    age_data = pd.read_csv(age_data, sep = "\t")

    # Load test set
    test_data = tissue_param[tissue]["test_data"]
    test_set = pd.read_csv(test_data)

    #Divide in train and test set
    X_train, X_test, y_train, y_test = split_in_train_test(gexp_data, age_data, test_set)
    
    print(X_train.shape)
    print(X_test.shape)
    
    #Define folds for CV
    n_folds = 5

    age_bins = np.arange(18, 76, 5)
    y_bins = pd.cut(y_train, age_bins, labels=False)
    skf_gen = StratifiedKFold(n_folds, shuffle = True, random_state = SEED).split(X_train, y_bins)

    folds = [[t[0], t[1]] for t in skf_gen]

    # Save folds to a text file
    for i in range(len(folds)): 
        fold = folds[i]
        train_data = fold[0]
        test_data = fold[1] 

        train_samples = X_train.index[train_data]
        test_samples = X_train.index[test_data]
        
        train_data = pd.DataFrame({"index": train_data, "sample": train_samples})
        test_data = pd.DataFrame({"index": test_data, "sample": test_samples})
        
        # Save train and test data to separate CSV files
        train_data.to_csv(f'results/3.feature_selection_gene_expresion/{tissue}/fold_{i}_train.csv', index=False)
        test_data.to_csv(f'results/3.feature_selection_gene_expresion/{tissue}/fold_{i}_test.csv', index=False)
            


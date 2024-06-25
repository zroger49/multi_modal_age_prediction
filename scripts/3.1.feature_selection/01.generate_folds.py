import os

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

SEED = 42

tissue_param = {"lung": {
                "name": "Lung", 
                "age_data": r"metadata/lung_annotation_meth.csv",
                "test_data": "metadata/lung_test_metadata.csv"
            }, 
            "ovary": {
                "name": "Ovary", 
                "age_data": r"metadata/ovary_annotation_meth.csv",
                "test_data": "metadata/ovary_test_metadata.csv"
            }
        }

from common_functions import (load_lung_data, 
                              load_ovary_data, 
                              split_in_train_test
                              )


tissues_to_run = ["lung", "ovary"]

# Load general metadata
metadata = pd.read_csv(r"metadata/eGTExDNA_Pierce_Jan18.09-11-2021.tsv", sep = "\t")

for tissue in tissues_to_run:
    res_dir = f"results/3.feature_selection/{tissue}"
    if not os.path.exists(res_dir): 
        os.mkdir(res_dir)

    if tissue == "lung":
        meth_data = load_lung_data()
    elif tissue == "ovary": 
        meth_data = load_ovary_data()
    else: 
        print("Error.. No tissue with name, ", tissue)
        next

    # Load age data
    age_data = tissue_param[tissue]["age_data"]
    age_data = pd.read_csv(age_data)

    # Load test set
    test_data = tissue_param[tissue]["test_data"]
    test_set = pd.read_csv(test_data)

    #Divide in train and test set
    X_train, X_test, y_train, y_test = split_in_train_test(meth_data, age_data, test_set)
    
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
        train_data.to_csv(f'results/3.feature_selection/{tissue}/fold_{i}_train.csv', index=False)
        test_data.to_csv(f'results/3.feature_selection/{tissue}/fold_{i}_test.csv', index=False)
            


"""Run Elastic Net Regression on each tissue on all features. Save features selected in each fold"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import ElasticNet


from common_functions import split_in_train_test


def load_folds_multi_modal(tissue, num_folds = 5): 
    num_folds = 5
    recreated_folds = []

    for i in range(num_folds):
        # Load train and test data from CSV files
        train_data = pd.read_csv(f'results/3.feature_selection_multimodal/{tissue}/fold_{i}_train.csv', index_col=0)
        test_data = pd.read_csv(f'results/3.feature_selection_multimodal/{tissue}/fold_{i}_test.csv', index_col=0)

        # Append the train and test index to the folds list
        recreated_folds.append([train_data["sample"], test_data["sample"]])
    return recreated_folds

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
            },
        }


tissues_to_run = ["lung", "ovary"]

for tissue in tissues_to_run:
    print("Running Analysis for ", tissue) 
    tissue_name = tissue_param[tissue]["name"]
    
    gexp_data_file = tissue_param[tissue]["gene_exp_data"]
    
    # Load data
    gexp_data = pd.read_csv(gexp_data_file,
                               header=0, index_col=0)
    
    # Load age data
    age_data = tissue_param[tissue]["age_data"]
    age_data = pd.read_csv(age_data, sep = "\t")

    # Load test data
    test_data = tissue_param[tissue]["test_data"]
    test_set = pd.read_csv(test_data)
 
    #Split in Train and test data
    X_train, X_test, y_train2, y_test = split_in_train_test(gexp_data, age_data, test_set)
    # y_train here will be ignore and recomputed later
    
    #Load folds for cross validation
    n_folds = 5
    folds = load_folds_multi_modal(tissue, num_folds=n_folds)
    ## Model Training ##
    pipe = Pipeline([
        ("QuantileTransformer", QuantileTransformer(n_quantiles = 100, random_state=SEED, output_distribution = "normal")),
        ("elasticNet", ElasticNet())
    ])

    print("Computing EN features for each fold")

    cv_scores = {}
    idx = 0

    # Load methylation exclusive samples 
    exclusive_gexp_samples = pd.read_csv(f'results/3.feature_selection_multimodal/{tissue}/exclusive_gene_expression_train.csv', index_col=0)
    exclusive_gexp_samples = exclusive_gexp_samples.index
    
    #X_exclusive = meth_t.loc[exclusive_gexp_samples.Sample_ID]
    #y_exclusive = exclusive_gexp_samples.AGE.tolist()
    y_train = age_data.copy()
    y_train.index = y_train.tissue_sample_id
    y_train.drop(columns="tissue_sample_id", axis=1,  inplace=True)

    print("Fitting model in each fold")
    for cv_train_idx, cv_valid_idx in folds: 
        cv_train_idx = cv_train_idx.str.replace("-SM-.{4,5}", "").tolist()
        cv_train_idx.extend(exclusive_gexp_samples.tolist())

        X_fold_train = X_train.loc[cv_train_idx]
        y_fold_train = np.array(y_train.loc[cv_train_idx].age.tolist())
        
        print(X_fold_train.shape)

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
        probes = X_fold_train.columns
        model_coef =  pd.DataFrame({"probe":probes, "coef" : coef})

        model_coef.to_csv(f"results/3.feature_selection_multimodal/results/{tissue_name}/fold_{idx}/linear_model_coef_gexp.csv")
        idx += 1

                    
    print("Fiting whole model")
    
    y_train = np.array(y_train.age.tolist()) 
    pipe.fit(X_train, y_train2)

    #Save coefficients
    coef = pipe[1].coef_
    probes = X_train.columns
    model_coef =  pd.DataFrame({"probe":probes, "coef" : coef})

    model_coef.to_csv(f"results/3.feature_selection_multimodal/results/{tissue_name}/train/linear_model_coef_gexp.csv")

    print("Finished %s", tissue)
    
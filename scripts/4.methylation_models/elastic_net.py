"""Run lightGBM on each tissue. As a feature selection use probes from the EN. Do hyperparam optimization. Use LDS"""

import mlflow
import optuna

import joblib

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import ElasticNet

from mlflow import log_metric

from common_functions import (load_lung_data, 
                              load_colon_data, 
                              load_ovary_data, 
                              load_prostate_data, 
                              split_in_train_test,
                              compute_metrics, 
                              plot_model_fit,
                              convert_beta_to_m,
                              filter_M_and_XY_probes,
                              load_folds,
                              prepare_weights)

import logging

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('EN_LDS_meth.log'),
    ]
)

SEED = 42
N_TRIALS_OPTUNA = 50

EXPERIMENT_ID = "297330103957631652"

tissue_param = {"lung": {
                    "name": "Lung", 
                    "age_data": r"metadata/lung_annotation_meth.csv",
                    "test_data": "metadata/lung_test_metadata.csv"
                }, 
                "ovary": {
                    "name": "Ovary", 
                    "age_data": r"metadata/ovary_annotation_meth.csv",
                    "test_data": "metadata/ovary_test_metadata.csv",
                }
            }


tissues_to_run = ["lung", "ovary"]

# Load general metadata
metadata = pd.read_csv(r"metadata/eGTExDNA_Pierce_Jan18.09-11-2021.tsv", sep = "\t")
#Load probe info
probe_info = pd.read_csv(r"metadata/methylation_epic_v1.0b5.csv")
probe_info = probe_info[probe_info["CHR_hg38"].notna()] #Remove NA probes in the CHR_hg38 genome


for tissue in tissues_to_run:
    logging.info("Running Analysis for %s", tissue) 

    if tissue == "lung":
        meth_data = load_lung_data()
    elif tissue == "colon_transverse": 
        meth_data = load_colon_data()
    elif tissue == "ovary": 
        meth_data = load_ovary_data()
    elif tissue == "prostate": 
        meth_data = load_prostate_data()
    else: 
        logging.info("Error.. No tissue with name %s", tissue)
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
 
    #Split in Train and test data
    logging.info("Spliting in train and test data..")
    X_train, X_test, y_train, y_test = split_in_train_test(meth_data, age_data, test_set)
    logging.info("Train dimensions: %s", X_train.shape)
    logging.info("Test dimensions: %s", X_test.shape)    
    
    #Load folds for cross validation
    n_folds = 5
    folds = load_folds(tissue, num_folds=n_folds)

    logging.info("Optimizing Hyperparams....")

    def objective(trial, X, y):   

        search_params = { 
            'alpha': trial.suggest_categorical("alpha", [0.01, 0.1, 1, 10, 100]),
            'l1_ratio': trial.suggest_categorical("l1_ratio", [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1]),
            'kernel': trial.suggest_categorical("kernel", ['gaussian', 'triang', 'laplace']),
            'reweight': trial.suggest_categorical("reweight", ['sqrt_inv', 'inverse']),
            'lds_ks': trial.suggest_int("lds_ks", 3, 8), 
            "lds_sigma": trial.suggest_int("lds_sigma", 1, 4),
        }

        cv_scores = np.empty(5)
        idx = 0
        for cv_train_idx, cv_valid_idx in folds: 
            # Feature selection
            feature_folder_name = tissue_param[tissue].get("name2", tissue_param[tissue]["name"])
            feature_file = "results/3.feature_selection/results/" + feature_folder_name + "/fold_" + str(idx) + "/DML_results.csv"
            features_dataset = pd.read_csv(feature_file)
            features_to_keep = features_dataset[features_dataset["adj.P.Val"] < 0.05]
            features_to_keep = features_to_keep.iloc[:, 0].tolist()
            
            features_to_keep = [feat for feat in features_to_keep if feat in X_train.columns]

            X_train_filtered = X[features_to_keep]

            # Get data from fold
            X_fold_train, X_fold_valid = X_train_filtered.iloc[cv_train_idx], X_train_filtered.iloc[cv_valid_idx]
            y_fold_train, y_fold_valid = np.array(y)[cv_train_idx], np.array(y)[cv_valid_idx]

            #Get weights 
            w = prepare_weights(labels = y_fold_train, 
                                reweight = search_params["reweight"], 
                                lds=True, 
                                lds_kernel=search_params["kernel"], 
                                lds_ks=search_params["lds_ks"],
                                lds_sigma=search_params["lds_sigma"])

            # Fit 
            pipe = Pipeline([
                ("QuantileTransformer", QuantileTransformer(n_quantiles = 100, random_state=SEED, output_distribution = "normal")),
                ("elasticNet", ElasticNet(alpha=search_params["alpha"], l1_ratio=search_params["l1_ratio"]))
            ])
            
            pipe.fit(X_fold_train, y_fold_train, elasticNet__sample_weight = w)

            preds = pipe.predict(X_fold_valid)
            cv_scores[idx] = mean_absolute_error(y_fold_valid, preds)
            idx += 1

        return np.nanmean(cv_scores)
        
    study = optuna.create_study(direction="minimize", study_name="EN")
    func = lambda trial: objective(trial, X_train, y_train)
    study.optimize(func, n_trials=N_TRIALS_OPTUNA)
    
    #Save the study dataframe
    study.trials_dataframe().to_csv(f"results/4.methylation_models/{tissue}/5.2_optuna_table.csv")
    
    best_params = study.best_params

    logging.info("Fitting with cross validation")
    idx = 0
    cv_scores = {}
    for cv_train_idx, cv_valid_idx in folds: 
        # Feature selection (pre-computed features)
        feature_folder_name = tissue_param[tissue].get("name2", tissue_param[tissue]["name"])
        feature_file = "results/3.feature_selection/results/" + feature_folder_name + "/fold_" + str(idx) + "/DML_results.csv"

        features_dataset = pd.read_csv(feature_file)
        features_to_keep = features_dataset[features_dataset["adj.P.Val"] < 0.05]
        features_to_keep = features_to_keep.iloc[:, 0].tolist()        
        features_to_keep = [feat for feat in features_to_keep if feat in X_train.columns]

        X_train_filtered = X_train[features_to_keep]

        # Get data from fold
        X_fold_train, X_fold_valid = X_train_filtered.iloc[cv_train_idx], X_train_filtered.iloc[cv_valid_idx]
        y_fold_train, y_fold_valid = np.array(y_train)[cv_train_idx], np.array(y_train)[cv_valid_idx]
    
        #Get weights 
        w = prepare_weights(labels = y_fold_train, 
                            reweight = best_params["reweight"], 
                            lds=True, 
                            lds_kernel=best_params["kernel"], 
                            lds_ks=best_params["lds_ks"],
                            lds_sigma=best_params["lds_sigma"])

        # Fit 
        pipe = Pipeline([
            ("QuantileTransformer", QuantileTransformer(n_quantiles = 100, random_state=SEED, output_distribution = "normal")),
            ("elasticNet", ElasticNet(alpha=best_params["alpha"], l1_ratio=best_params["l1_ratio"]))
        ])
            
        pipe.fit(X_fold_train, y_fold_train, elasticNet__sample_weight = w)
        preds = pipe.predict(X_fold_valid)
        metrics_cv = compute_metrics(y_fold_valid, preds)
        for metric, value in metrics_cv.items(): 
            if metric not in cv_scores.keys() : 
                cv_scores[metric] = np.empty(5)   
                cv_scores[metric][0] = value
            else: 
                cv_scores[metric][idx] = value
        idx += 1

    with mlflow.start_run(run_name = tissue + "_elastic_net_dml_optimized_LDS", experiment_id = EXPERIMENT_ID) as _:
        # Save the cross-validation metrics
        for key, item in cv_scores.items(): 
            log_metric("cv_" + key + "_mean", cv_scores[key].mean())
            log_metric("cv_" + key + "_std",  cv_scores[key].std())
    
        logging.info("Fiting whole model")
        
        mlflow.sklearn.autolog() #autologs the model

        # Feature selection
        feature_folder_name = tissue_param[tissue].get("name2", tissue_param[tissue]["name"])
        feature_file = "results/3.feature_selection/results/" + feature_folder_name + "/train" + "/DML_results.csv"
        features_dataset = pd.read_csv(feature_file)
        features_to_keep = features_dataset[features_dataset["adj.P.Val"] < 0.05]
        features_to_keep = features_to_keep.iloc[:, 0].tolist()        

        features_to_keep = [feat for feat in features_to_keep if feat in X_train.columns]

        X_train_filtered = X_train[features_to_keep]
        X_test_filtered = X_test[features_to_keep]
        
        #Get weights 
        w = prepare_weights(labels = y_train, 
                            reweight = best_params["reweight"], 
                            lds=True, 
                            lds_kernel=best_params["kernel"], 
                            lds_ks=best_params["lds_ks"],
                            lds_sigma=best_params["lds_sigma"])
                            
        # Fit 
        pipe = Pipeline([
            ("QuantileTransformer", QuantileTransformer(n_quantiles = 100, random_state=SEED, output_distribution = "normal")),
            ("elasticNet", ElasticNet(alpha=best_params["alpha"], l1_ratio=best_params["l1_ratio"]))
        ])
        pipe.fit(X_train_filtered, y_train, elasticNet__sample_weight = w)

        #Compute metrics
        logging.info("Logging metrics...")
        y_train_pred = pipe.predict(X_train_filtered)
        metrics_train = compute_metrics(y_train, y_train_pred)
        train_prediction = pd.DataFrame({"ind": X_train_filtered.index, "true": y_train, "pred": y_train_pred})
        train_prediction.to_csv(f"results/4.methylation_models/{tissue}/5.2_train_prediction.csv")

        y_test_pred = pipe.predict(X_test_filtered)
        metrics_test = compute_metrics(y_test, y_test_pred)
        test_prediction = pd.DataFrame({"ind": X_test_filtered.index, "true": y_test, "pred": y_test_pred})
        test_prediction.to_csv(f"results/4.methylation_models/{tissue}/5.2_test_prediction.csv")

        for model_metric in metrics_train.keys(): 
            log_metric(model_metric + "_train", metrics_train[model_metric])
        
        for model_metric in metrics_test.keys(): 
            log_metric(model_metric + "_test", metrics_test[model_metric])

        #Plot model Fit
        plot_model_fit(y_train, 
                       y_train_pred, 
                       data_set="Train", 
                       fig_output_path= f"aging_notes/figures/4.methylation_models/5.2_{tissue}_GBDT_fit_train.pdf")


        plot_model_fit(y_test, 
                    y_test_pred, 
                    data_set="Test", 
                    fig_output_path= f"aging_notes/figures/4.methylation_models/5.2_{tissue}_GBDT_fit_test.pdf")
        
    
        #Save coefficients
        coef = pipe[1].coef_
        model_coef =  pd.DataFrame({"probe":features_to_keep, "coef" : coef})       
        model_coef.to_csv(f"results/4.methylation_models/{tissue}/GBDT_feature_importance.csv")

        # Save model 
        joblib.dump(pipe, f'results/4.methylation_models/{tissue}/pipeline.pkl', compress = 1)

    logging.info("Finished %s", tissue)
    
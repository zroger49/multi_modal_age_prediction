"""Run lightGBM on each tissue. As a feature selection use probes from the EN. Do hyperparam optimization (Exploit SMOGN AND LDS)"""

import mlflow
import optuna

import joblib

import numpy as np
import pandas as pd
import lightgbm as lgbm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_absolute_error

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
                              prepare_weights,
                              smogn)

import logging

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('LGBM_EN_LDS_meth.log'),
    ]
)

SEED = 42
N_TRIALS_OPTUNA = 250

EXPERIMENT_ID = "514083558578029078"

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


tissues_to_run = ["lung", "colon_transverse", "ovary", "prostate"]

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
            # Methylation feature selection
            'meth_features': trial.suggest_categorical("meth_features", ["DML", "EN"]),
            # LightGBM parameters
            'n_estimators': trial.suggest_int("n_estimators", 20, 160),
            'learning_rate': trial.suggest_float("learning_rate", 1e-3,5e-1,log=True),
            'reg_alpha': trial.suggest_float("reg_alpha", 1e-3, 10.0, log = True),
            'reg_lambda': trial.suggest_float("reg_lambda", 1e-3, 10.0, log = True),
            'num_leaves': trial.suggest_int("num_leaves", 2, 12),
            'max_depth': trial.suggest_int("max_depth", 3, 5),
            'subsample': trial.suggest_float("subsample", 0.2, 0.8),
            'feature_fraction': trial.suggest_float("feature_fraction", 0.2, 0.8),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 20),
            # LDS hyperparams
            'use_lds': trial.suggest_categorical("use_lds", [True, False]),
            'kernel': trial.suggest_categorical("kernel", ['gaussian', 'triang', 'laplace']),
            'reweight': trial.suggest_categorical("reweight", ['sqrt_inv', 'inverse']),
            'lds_ks': trial.suggest_int("lds_ks", 3, 8), 
            "lds_sigma": trial.suggest_int("lds_sigma", 1, 4),
        }

        cv_scores = np.empty(5)
        idx = 0
        for cv_train_idx, cv_valid_idx in folds: 
            # Feature selection (pre-computed features)

            if search_params["meth_features"] == "DML": 
                feature_folder_name = tissue_param[tissue].get("name2", tissue_param[tissue]["name"])
                feature_file = "results/3.feature_selection/results/" + feature_folder_name + "/fold_" + str(idx) + "/DML_results.csv"
                features_dataset = pd.read_csv(feature_file)
                features_to_keep = features_dataset[features_dataset["adj.P.Val"] < 0.05]
                features_to_keep = features_to_keep.iloc[:, 0].tolist()
            elif search_params["meth_features"] == "EN": 
                feature_folder_name = tissue_param[tissue].get("name2", tissue_param[tissue]["name"])
                feature_file = "results/3.feature_selection/results/" + feature_folder_name + "/fold_" + str(idx) + "/linear_model_coef.csv"
                features_dataset = pd.read_csv(feature_file)
                features_to_keep = features_dataset[features_dataset["coef"] != 0]
                features_to_keep = features_to_keep.probe.tolist()

            features_to_keep = [feat for feat in features_to_keep if feat in X_train.columns]
            X_train_filtered = X[features_to_keep]

            # Get data from fold
            X_fold_train, X_fold_valid = X_train_filtered.iloc[cv_train_idx], X_train_filtered.iloc[cv_valid_idx]
            y_fold_train, y_fold_valid = np.array(y_train)[cv_train_idx], np.array(y_train)[cv_valid_idx]
            
            #Get weights 
            if search_params["use_lds"] == True: 
                w = prepare_weights(labels = y_fold_train, 
                                    reweight = search_params["reweight"], 
                                    lds=True, 
                                    lds_kernel=search_params["kernel"], 
                                    lds_ks=search_params["lds_ks"],
                                    lds_sigma=search_params["lds_sigma"])

            model = lgbm.LGBMRegressor(
                objective="regression",
                random_state = SEED,
                extra_trees = True,
                n_estimators=search_params["n_estimators"],
                learning_rate  = search_params["learning_rate"],
                reg_alpha = search_params["reg_alpha"],
                reg_lambda = search_params["reg_lambda"],
                num_leaves=search_params["num_leaves"],
                max_depth = search_params["max_depth"],
                subsample = search_params["subsample"],
                feature_fraction = search_params["feature_fraction"],
                min_child_samples = search_params["min_child_samples"]
            )

            pipe = Pipeline([
                ("QuantileTransformer", QuantileTransformer(n_quantiles = 100, random_state=SEED, output_distribution = "normal")),
                ("GBDT", model)
            ])

            if search_params["use_lds"] == True: 
                pipe.fit( 
                    X_fold_train,
                    y_fold_train,
                    GBDT__sample_weight = w,
                    GBDT__eval_metric="mae"
                )
            else: 
                pipe.fit( 
                        X_fold_train,
                        y_fold_train,
                        GBDT__eval_metric="mae"
                    )
                

            preds = pipe.predict(X_fold_valid, num_iteration = pipe[1].best_iteration_)
            cv_scores[idx] = mean_absolute_error(y_fold_valid, preds)
            idx += 1

        return np.nanmean(cv_scores)
        
    study = optuna.create_study(direction="minimize", study_name="LGBM")
    func = lambda trial: objective(trial, X_train, y_train)
    study.optimize(func, n_trials=N_TRIALS_OPTUNA)
    
    #Save the study dataframe
    study.trials_dataframe().to_csv(f"results/4.methylation_models/{tissue}/12.2_optuna_table.csv")
    
    best_params = study.best_params

    logging.info("Fitting with cross validation")
    idx = 0
    cv_scores = {}
    for cv_train_idx, cv_valid_idx in folds: 
        # Feature selection (pre-computed features)
        if best_params["meth_features"] == "DML": 
                feature_folder_name = tissue_param[tissue].get("name2", tissue_param[tissue]["name"])
                feature_file = "results/3.feature_selection/results/" + feature_folder_name + "/fold_" + str(idx) + "/DML_results.csv"
                features_dataset = pd.read_csv(feature_file)
                features_to_keep = features_dataset[features_dataset["adj.P.Val"] < 0.05]
                features_to_keep = features_to_keep.iloc[:, 0].tolist()
        elif best_params["meth_features"] == "EN": 
            feature_folder_name = tissue_param[tissue].get("name2", tissue_param[tissue]["name"])
            feature_file = "results/3.feature_selection/results/" + feature_folder_name + "/fold_" + str(idx) + "/linear_model_coef.csv"
            features_dataset = pd.read_csv(feature_file)
            features_to_keep = features_dataset[features_dataset["coef"] != 0]
            features_to_keep = features_to_keep.probe.tolist()
        
         
        features_to_keep = [feat for feat in features_to_keep if feat in X_train.columns]
        X_train_filtered = X_train[features_to_keep]

        # Get data from fold
        X_fold_train, X_fold_valid = X_train_filtered.iloc[cv_train_idx], X_train_filtered.iloc[cv_valid_idx]
        y_fold_train, y_fold_valid = np.array(y_train)[cv_train_idx], np.array(y_train)[cv_valid_idx]
    
        #Get weights 
        # Apply the LDS (Conditional)
        if best_params["use_lds"] == True: 
            w = prepare_weights(labels = y_fold_train, 
                        reweight = best_params["reweight"], 
                        lds=True, 
                        lds_kernel=best_params["kernel"], 
                        lds_ks=best_params["lds_ks"],
                        lds_sigma=best_params["lds_sigma"])
        

        model = lgbm.LGBMRegressor(
            objective="regression",
            random_state = SEED,
            extra_trees = True,
            n_estimators=best_params["n_estimators"],
            learning_rate  = best_params["learning_rate"],
            reg_alpha = best_params["reg_alpha"],
            reg_lambda = best_params["reg_lambda"],
            num_leaves=best_params["num_leaves"],
            max_depth = best_params["max_depth"],
            subsample = best_params["subsample"],
            feature_fraction = best_params["feature_fraction"],
            min_child_samples = best_params["min_child_samples"]
        )

        
        pipe = Pipeline([
            ("QuantileTransformer", QuantileTransformer(n_quantiles = 100, random_state=SEED, output_distribution = "normal")),
            ("GBDT", model)
        ])


        if best_params["use_lds"] == True: 
            pipe.fit( 
                X_fold_train,
                y_fold_train,
                GBDT__sample_weight = w,
                GBDT__eval_metric="mae"
            )
        else: 
            pipe.fit( 
                    X_fold_train,
                    y_fold_train,
                    GBDT__eval_metric="mae"
                )

        preds = pipe.predict(X_fold_valid, num_iteration = pipe[1].best_iteration_)
        metrics_cv = compute_metrics(y_fold_valid, preds)
        for metric, value in metrics_cv.items(): 
            if metric not in cv_scores.keys() : 
                cv_scores[metric] = np.empty(5)   
                cv_scores[metric][0] = value
            else: 
                cv_scores[metric][idx] = value
        idx += 1

        
       

    with mlflow.start_run(run_name = tissue + "_lgbm_en_hp_LDS_feature_sel_optuna", experiment_id = EXPERIMENT_ID) as _:
        # Save the cross-validation metrics
        for key, item in cv_scores.items(): 
            log_metric("cv_" + key + "_mean", cv_scores[key].mean())
            log_metric("cv_" + key + "_std",  cv_scores[key].std())
    
        logging.info("Fiting whole model")
        
        mlflow.sklearn.autolog() #autologs the model

        # Feature selection
        if best_params["meth_features"] == "DML": 
            feature_folder_name = tissue_param[tissue].get("name2", tissue_param[tissue]["name"])
            feature_file = "results/3.feature_selection/results/" + feature_folder_name + "/train/DML_results.csv"
            features_dataset = pd.read_csv(feature_file)
            features_to_keep = features_dataset[features_dataset["adj.P.Val"] < 0.05]
            features_to_keep = features_to_keep.iloc[:, 0].tolist()
        elif best_params["meth_features"] == "EN": 
            feature_folder_name = tissue_param[tissue].get("name2", tissue_param[tissue]["name"])
            feature_file = "results/3.feature_selection/results/" + feature_folder_name + "/train/linear_model_coef.csv"
            features_dataset = pd.read_csv(feature_file)
            features_to_keep = features_dataset[features_dataset["coef"] != 0]
            features_to_keep = features_to_keep.probe.tolist()
 
        features_to_keep = [feat for feat in features_to_keep if feat in X_train.columns]

        X_train_filtered = X_train[features_to_keep]
        X_test_filtered = X_test[features_to_keep]

         #Get weights 
        if best_params["use_lds"] == True: 
            w = prepare_weights(labels = y_train, 
                        reweight = best_params["reweight"], 
                        lds=True, 
                        lds_kernel=best_params["kernel"], 
                        lds_ks=best_params["lds_ks"],
                        lds_sigma=best_params["lds_sigma"])

        model = lgbm.LGBMRegressor(
            objective="regression",
            random_state = SEED,
            extra_trees = True,
            n_estimators=best_params["n_estimators"],
            learning_rate  = best_params["learning_rate"],
            reg_alpha = best_params["reg_alpha"],
            reg_lambda = best_params["reg_lambda"],
            num_leaves=best_params["num_leaves"],
            max_depth = best_params["max_depth"],
            subsample = best_params["subsample"],
            feature_fraction = best_params["feature_fraction"],
            min_child_samples = best_params["min_child_samples"]
        )

        pipe = Pipeline([
            ("QuantileTransformer", QuantileTransformer(n_quantiles = 100, random_state=SEED, output_distribution = "normal")),
            ("GBDT", model)
        ])
        if best_params["use_lds"] == True: 
            pipe.fit(X_train_filtered, y_train, GBDT__eval_metric = "mae", GBDT__sample_weight = w)
        else: 
            pipe.fit(X_train_filtered, y_train, GBDT__eval_metric = "mae")
            

        #Compute metrics
        logging.info("Logging metrics...")
        y_train_pred = pipe.predict(X_train_filtered)
        metrics_train = compute_metrics(y_train, y_train_pred)
        train_prediction = pd.DataFrame({"ind": X_train_filtered.index, "true": y_train, "pred": y_train_pred})
        train_prediction.to_csv(f"results/4.methylation_models/{tissue}/12.2_train_prediction.csv")

        y_test_pred = pipe.predict(X_test_filtered)
        metrics_test = compute_metrics(y_test, y_test_pred)
        test_prediction = pd.DataFrame({"ind": X_test_filtered.index, "true": y_test, "pred": y_test_pred})
        test_prediction.to_csv(f"results/4.methylation_models/{tissue}/12.2_test_prediction.csv")

        for model_metric in metrics_train.keys(): 
            log_metric(model_metric + "_train", metrics_train[model_metric])
        
        for model_metric in metrics_test.keys(): 
            log_metric(model_metric + "_test", metrics_test[model_metric])

        #Plot model Fit
        plot_model_fit(y_train, 
                       y_train_pred, 
                       data_set="Train", 
                       fig_output_path= f"aging_notes/figures/4.methylation_models/12.2_{tissue}_GBDT_fit_train.pdf")


        plot_model_fit(y_test, 
                    y_test_pred, 
                    data_set="Test", 
                    fig_output_path= f"aging_notes/figures/4.methylation_models/12.2_{tissue}_GBDT_fit_test.pdf")
        
    
        #Save coefficients
        importance_gain = pipe[1].feature_importances_
        probes = features_to_keep
        model_coef =  pd.DataFrame({"probe":probes, "coef" : importance_gain})

        model_coef.to_csv(f"results/4.methylation_models/{tissue}/12.2.GBDT_feature_importance.csv")

        # Save model 
        joblib.dump(pipe, f'results/4.methylation_models/{tissue}/12.1.pipeline.pkl', compress = 1)

    logging.info("Finished %s", tissue)
    
"""Run EN on each tissue using gexp 
Do hyperparam optimization and use LDS. 
Run the analysis with the samples in common for Gexp and Histology"""

import mlflow
import optuna

import joblib
import pickle

import numpy as np
import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_absolute_error

import lightgbm as lgbm


from mlflow import log_metric

from common_functions import (split_in_train_test,
                              compute_metrics, 
                              plot_model_fit,
                              load_folds,
                              prepare_weights)

import logging

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('9.4_gexp_downsample_EN_LDS.log'),
    ]
)

SEED = 42
N_TRIALS_OPTUNA = 250

#EXPERIMENT_ID  = mlflow.create_experiment(f"multi_modal_models_gexp_hist")
EXPERIMENT_ID = "895740904223637091"

tissue_param = {"lung": {
                    "name": "Lung", 
                    "age_data": r"metadata/gene_expression_metadata/metadata_lung.tsv",
                    "test_data": "metadata/lung_test_metadata.csv",
                    "hist_data": "data/features_histology/lung/lung_features_mean256_cls4k.pkl"
                }, 
                "ovary": {
                    "name": "Ovary", 
                    "age_data": r"metadata/gene_expression_metadata/metadata_ovary.tsv",
                    "test_data": "metadata/ovary_test_metadata.csv",
                    "hist_data": "data/features_histology/ovary/ovary_features_mean256_cls4k.pkl"
                },
            }


tissues_to_run = ["lung", "ovary"]


for tissue in tissues_to_run:
    logging.info("Running Analysis for %s", tissue) 

    tissue_name = tissue_param[tissue]["name"]
    
    ## Metadata 
    age_data = tissue_param[tissue]["age_data"]
    age_data = pd.read_csv(age_data, sep = "\t")

    # Load Gexp data
    X_coding_log = pd.read_csv(f"data/X_coding_{tissue}_log2.csv",
                            header=0, index_col=0)
    
    #Note: (Gene expression data is already filtered and pre-processed)
    # Load common samples to the dataset
    multi_modal_table = "metadata/sample_ids_multiomics_updated_tl_data.csv"
    multi_modal_table = pd.read_csv(multi_modal_table)

    filtered_multi_modal_table = multi_modal_table[multi_modal_table['tissue'] == tissue_name]
    complete_samples = filtered_multi_modal_table[(filtered_multi_modal_table["metadata"]) == 1 & (filtered_multi_modal_table["gene_expression"])]
    
    ## Since some of the subjects were not processed, load the histology dataset and keep only common samples    
    # Load histology data
    hist_file = tissue_param[tissue]["hist_data"]
    hist_data = pickle.load(open(hist_file, "rb"))
    # Subset
    complete_samples = complete_samples[complete_samples.sample_id.isin(hist_data.index)]
    age_data_multi_modal = age_data.loc[age_data["tissue_sample_id"].isin(complete_samples["sample_id"])]
    
    # Load test data
    test_data = tissue_param[tissue]["test_data"]
    test_set = pd.read_csv(test_data)
 
    #Split in Train and test data.
    logging.info("Spliting methylation data into train and test data..")
    X_train, X_test, y_train, y_test = split_in_train_test(X_coding_log, age_data_multi_modal, test_set)
    

    #Load folds for cross validation
    n_folds = 5
    folds = load_folds(tissue, num_folds=n_folds)

    logging.info("Optimizing Hyperparams....")

        
    def objective(trial, X):   
        complete_samples_age = complete_samples.copy()
        complete_samples_age = complete_samples_age.set_index("sample_id")
        
        complete_samples_age_train = complete_samples_age[~complete_samples_age.index.isin(test_set["sample_id"])]
        
        search_params = { 
            # gene expression features
            'gexp_features': trial.suggest_categorical("gexp_features", ["EN", "DEG"]), 
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
            "lds_sigma": trial.suggest_int("lds_sigma", 1, 4)
        }

    
        cv_scores = np.empty(5)
        idx = 0
        for cv_train_samples, cv_valid_samples in folds: 
            # Feature selection for Gexp features (This is a parameter to optimize)
            if search_params["gexp_features"] == "EN":
                en = pd.read_csv(f'results/3.feature_selection_gexp_hist/results/{tissue_name}/fold_{idx}/linear_model_coef_gexp.csv',
                                    index_col=0)
                feat_sel_gexp = en[en["coef"] != 0]["probe"].to_list() # here it says probe but its genes
            if search_params["gexp_features"] == "DEG":
                degs = pd.read_csv(f'results/3.feature_selection_gexp_hist/results/{tissue_name}/fold_{idx}/DEG_results.csv',
                                    index_col=0)
                feat_sel_gexp = degs[degs["adj.P.Val"] < 0.05].index.tolist()
            
            X_train_filtered = X[feat_sel_gexp]
            
            # Get data from fold
            X_fold_train, X_fold_valid = X_train_filtered.loc[cv_train_samples], X_train_filtered.loc[cv_valid_samples]
            y_fold_train, y_fold_valid = np.array(complete_samples_age.loc[cv_train_samples]["age"]), np.array(complete_samples_age.loc[cv_valid_samples]["age"])  
 

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
                
            preds = pipe.predict(X_fold_valid)
            cv_scores[idx] = mean_absolute_error(y_fold_valid, preds)
            idx += 1

        return np.nanmean(cv_scores)
        
    study = optuna.create_study(direction="minimize", study_name="EN")
    func = lambda trial: objective(trial, X_train)
    study.optimize(func, n_trials=N_TRIALS_OPTUNA)
    
    #Save the study dataframe
    study.trials_dataframe().to_csv(f"results/9.multi_modal_models_gexp_hist/{tissue}/4_optuna_table.csv")
    best_params = study.best_params
    
 
    logging.info("Fitting with cross validation")
    idx = 0
    cv_scores = {}

    complete_samples_age = complete_samples.copy()
    complete_samples_age = complete_samples_age.set_index("sample_id")    

    complete_samples_age_train = complete_samples_age[~complete_samples_age.index.isin(test_set["sample_id"])]
    complete_samples_age_test = complete_samples_age[complete_samples_age.index.isin(test_set["sample_id"])]

    
    for cv_train_samples, cv_valid_samples in folds: 
        # Feature selection for Gexp features (This is a parameter to optimize)
        if best_params["gexp_features"] == "EN":
            en = pd.read_csv(f'results/3.feature_selection_gexp_hist/results/{tissue_name}/fold_{idx}/linear_model_coef_gexp.csv',
                                index_col=0)
            feat_sel_gexp = en[en["coef"] != 0]["probe"].to_list() # here it says probe but its genes
        if best_params["gexp_features"] == "DEG":
            degs = pd.read_csv(f'results/3.feature_selection_gexp_hist/results/{tissue_name}/fold_{idx}/DEG_results.csv',
                                index_col=0)
            feat_sel_gexp = degs[degs["adj.P.Val"] < 0.05].index.tolist()

    
        X_train_filtered = X_train[feat_sel_gexp]

        # Get data from fold
        X_fold_train, X_fold_valid = X_train_filtered.loc[cv_train_samples], X_train_filtered.loc[cv_valid_samples]
        y_fold_train, y_fold_valid = np.array(complete_samples_age.loc[cv_train_samples]["age"]), np.array(complete_samples_age.loc[cv_valid_samples]["age"])  
        
        #Get weights 
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
    
        
        preds = pipe.predict(X_fold_valid)
        metrics_cv = compute_metrics(y_fold_valid, preds)
        for metric, value in metrics_cv.items(): 
            if metric not in cv_scores.keys() : 
                cv_scores[metric] = np.empty(5)   
                cv_scores[metric][0] = value
            else: 
                cv_scores[metric][idx] = value
        idx += 1
    
    with mlflow.start_run(run_name = tissue + "_GBT_multi_modal_gexp_downsample", experiment_id = EXPERIMENT_ID) as _:
        # Save the cross-validation metrics
        for key, item in cv_scores.items(): 
            log_metric("cv_" + key + "_mean", cv_scores[key].mean())
            log_metric("cv_" + key + "_std",  cv_scores[key].std())

        # Compute score (optimizing MAE)
        for idxd in range(5):
            log_metric(f"cv_mae_fold_{idxd}", cv_scores["mae"][idxd])
        
        logging.info("Fiting whole model")
        
        mlflow.sklearn.autolog() #autologs the model

        if best_params["gexp_features"] == "EN":
            en = pd.read_csv(f'results/3.feature_selection_gexp_hist/results/{tissue_name}/train/linear_model_coef_gexp.csv',
                                    index_col=0)
            feat_sel_gexp = en[en["coef"] != 0]["probe"].to_list() # here it says probe but its genes
        if best_params["gexp_features"] == "DEG":
            degs = pd.read_csv(f'results/3.feature_selection_gexp_hist/results/{tissue_name}/train/DEG_results.csv',
                                    index_col=0)
            feat_sel_gexp = degs[degs["adj.P.Val"] < 0.05].index.tolist()

        X_train_filtered = X_train[feat_sel_gexp]
        X_test_filtered = X_test[feat_sel_gexp]

        hist_data_file = f"data/features_histology/{tissue}/{tissue}_features_mean256_cls4k.pkl"
        hist_data_df =  pickle.load(open(hist_data_file, "rb"))
    
        # Subset for common samples 
        hist_data_df_train = hist_data_df.loc[complete_samples_age_train.index]
        common_samples =  set(X_train_filtered.index.tolist()).intersection(set(hist_data_df_train.index.tolist()))

        X_train_filtered = X_train_filtered.loc[list(common_samples)]

        y_train = np.array(complete_samples_age.loc[list(common_samples)]["age"])
        y_test = np.array(complete_samples_age.loc[X_test.index]["age"])  

        #Get weights 
        if best_params["use_lds"] == True: 
            w = prepare_weights(labels = y_train, 
                                reweight = best_params["reweight"], 
                                lds=True, 
                                lds_kernel=best_params["kernel"], 
                                lds_ks=best_params["lds_ks"],
                                lds_sigma=best_params["lds_sigma"])



        # Fit 
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
                X_train_filtered,
                y_train,
                GBDT__sample_weight = w,
                GBDT__eval_metric="mae"
            )
        else: 
            pipe.fit( 
                    X_train_filtered,
                    y_train,
                    GBDT__eval_metric="mae"
                )
    

        #Compute metrics
        #logging.info("Logging metrics...")
        y_train_pred = pipe.predict(X_train_filtered)
        metrics_train = compute_metrics(y_train, y_train_pred)
        train_prediction = pd.DataFrame({"ind": X_train_filtered.index, "true": y_train, "pred": y_train_pred})
        train_prediction.to_csv(f"results/9.multi_modal_models_gexp_hist/{tissue}/3_gexp_en_train_prediction.csv")

        y_test_pred = pipe.predict(X_test_filtered)
        metrics_test = compute_metrics(y_test, y_test_pred)
        test_prediction = pd.DataFrame({"ind": X_test_filtered.index, "true": y_test, "pred": y_test_pred})
        test_prediction.to_csv(f"results/9.multi_modal_models_gexp_hist/{tissue}/3_gexp_en_test_prediction.csv")

        for model_metric in metrics_train.keys(): 
            log_metric(model_metric + "_train", metrics_train[model_metric])

        for model_metric in metrics_test.keys(): 
            log_metric(model_metric + "_test", metrics_test[model_metric])

        #Plot model Fit
        plot_model_fit(y_train, 
                        y_train_pred, 
                        data_set="Train", 
                        fig_output_path= f"aging_notes/figures/9.multi_modal_models_gexp_hist/4_{tissue}_GBT_gexp_fit_train.pdf")


        plot_model_fit(y_test, 
                    y_test_pred, 
                    data_set="Test", 
                    fig_output_path= f"aging_notes/figures/9.multi_modal_models_gexp_hist/4_{tissue}_GBT_gexp_fit_test.pdf")


        #Save coefficients
        coef = pipe[1].feature_importances_
        features = X_train_filtered.columns
        model_coef =  pd.DataFrame({"features":features, "coef" : coef})         
        model_coef.to_csv(f"results/9.multi_modal_models_gexp_hist/{tissue}/4_GBT_gexp_feature_importance.csv")

        # Save model 
        joblib.dump(pipe, f'results/9.multi_modal_models_gexp_hist/{tissue}/4_GBT_gexp.pipeline.pkl', compress = 1)

        mlflow.sklearn.autolog(disable = True) #autologs the model

    logging.info(f"Finished {tissue}")
    
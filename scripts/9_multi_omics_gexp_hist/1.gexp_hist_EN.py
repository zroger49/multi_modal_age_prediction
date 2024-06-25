"""Run EN on each tissue using features from histology and gexp 
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
from sklearn.linear_model import ElasticNet

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
        logging.FileHandler('9.1_hist_gexp_EN_LDS.log'),
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
    
    # Histology data will be added later
    multi_omics_data = X_coding_log

    # Load test data
    test_data = tissue_param[tissue]["test_data"]
    test_set = pd.read_csv(test_data)
 
    #Split in Train and test data.
    logging.info("Spliting methylation data into train and test data..")
    X_train, X_test, y_train, y_test = split_in_train_test(multi_omics_data, age_data_multi_modal, test_set)
    

    #Load folds for cross validation
    n_folds = 5
    folds = load_folds(tissue, num_folds=n_folds)

    logging.info("Optimizing Hyperparams....")

        
    def objective(trial, X):   
        complete_samples_age = complete_samples.copy()
        complete_samples_age = complete_samples_age.set_index("sample_id")
        
        complete_samples_age_train = complete_samples_age[~complete_samples_age.index.isin(test_set["sample_id"])]
        
        search_params = { 
            # EN params
            'alpha': trial.suggest_categorical("alpha", [0.01, 0.1, 1, 10, 100]),
            'l1_ratio': trial.suggest_categorical("l1_ratio", [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1]),
            # gene expression features
            'gexp_features': trial.suggest_categorical("gexp_features", ["EN", "DEG"]), 
            # histology_features parameters
            'histological_features': trial.suggest_categorical("histological_features", ["mean256_cls4k", "mean256", "cls4k"]), 
            # LDS hyperparams
            'use_lds': trial.suggest_categorical("use_lds", [True, False]),
            'kernel': trial.suggest_categorical("kernel", ['gaussian', 'triang', 'laplace']),
            'reweight': trial.suggest_categorical("reweight", ['sqrt_inv', 'inverse']),
            'lds_ks': trial.suggest_int("lds_ks", 3, 8), 
            "lds_sigma": trial.suggest_int("lds_sigma", 1, 4)
        }

        # Before CV, load the histological features to use
        hist_data_file = f"data/features_histology/{tissue}/{tissue}_features_{search_params['histological_features']}.pkl"
        hist_data_df =  pickle.load(open(hist_data_file, "rb"))
        hist_data_df.columns = [f'hist_feature_{i}' for i in range(len(hist_data_df.columns))]
        
        # Subset for common samples 
        hist_data_df = hist_data_df.loc[complete_samples_age_train.index]


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
            
            # Concatenate data from histology
            X_train_filtered = pd.concat([X_train_filtered, hist_data_df], axis = 1)

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
            
            # Fit 
            pipe = Pipeline([
                ("QuantileTransformer", QuantileTransformer(n_quantiles = 100, random_state=SEED, output_distribution = "normal")),
                ("elasticNet", ElasticNet(alpha=search_params["alpha"], l1_ratio=search_params["l1_ratio"]))
            ])
            
            if search_params["use_lds"] == True: 
                pipe.fit(X_fold_train, y_fold_train, elasticNet__sample_weight = w)
            else: 
                pipe.fit(X_fold_train, y_fold_train)
            preds = pipe.predict(X_fold_valid)
            cv_scores[idx] = mean_absolute_error(y_fold_valid, preds)
            idx += 1

        return np.nanmean(cv_scores)
        
    study = optuna.create_study(direction="minimize", study_name="EN")
    func = lambda trial: objective(trial, X_train)
    study.optimize(func, n_trials=N_TRIALS_OPTUNA)
    
    #Save the study dataframe
    study.trials_dataframe().to_csv(f"results/9.multi_modal_models_gexp_hist/{tissue}/1_optuna_table.csv")
    best_params = study.best_params
    
 
    logging.info("Fitting with cross validation")
    idx = 0
    cv_scores = {}

    complete_samples_age = complete_samples.copy()
    complete_samples_age = complete_samples_age.set_index("sample_id")    

    complete_samples_age_train = complete_samples_age[~complete_samples_age.index.isin(test_set["sample_id"])]
    complete_samples_age_test = complete_samples_age[complete_samples_age.index.isin(test_set["sample_id"])]


    hist_data_file = f"data/features_histology/{tissue}/{tissue}_features_{best_params['histological_features']}.pkl"
    hist_data_df =  pickle.load(open(hist_data_file, "rb"))
    hist_data_df.columns = [f'hist_feature_{i}' for i in range(len(hist_data_df.columns))]
    
    # Subset for common samples 
    hist_data_df_train = hist_data_df.loc[complete_samples_age_train.index]
    hist_data_df_test = hist_data_df.loc[test_set.sample_id]

    
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

        # Concatenate data from histology
        X_train_filtered = pd.concat([X_train_filtered, hist_data_df_train], axis = 1)

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
        
        
        
        # Fit 
        pipe = Pipeline([
            ("QuantileTransformer", QuantileTransformer(n_quantiles = 100, random_state=SEED, output_distribution = "normal")),
            ("elasticNet", ElasticNet(alpha=best_params["alpha"], l1_ratio=best_params["l1_ratio"]))
        ])
        if best_params["use_lds"] == True: 
            pipe.fit(X_fold_train, y_fold_train, elasticNet__sample_weight = w)
        else: 
            pipe.fit(X_fold_train, y_fold_train)
        
        preds = pipe.predict(X_fold_valid)
        metrics_cv = compute_metrics(y_fold_valid, preds)
        for metric, value in metrics_cv.items(): 
            if metric not in cv_scores.keys() : 
                cv_scores[metric] = np.empty(5)   
                cv_scores[metric][0] = value
            else: 
                cv_scores[metric][idx] = value
        idx += 1
    
    with mlflow.start_run(run_name = tissue + "_elastic_net_multi_modal_gexp_hist", experiment_id = EXPERIMENT_ID) as _:
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
    
        # Concatenate data from histology
        hist_data_df_train = hist_data_df_train.loc[X_train_filtered.index]
        
        X_train_filtered = pd.concat([X_train_filtered, hist_data_df_train], axis = 1)
        X_test_filtered = pd.concat([X_test_filtered, hist_data_df_test], axis = 1)
        
        #Get weights 
        if best_params["use_lds"] == True: 
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
        if best_params["use_lds"] == True: 
            pipe.fit(X_train_filtered, y_train, elasticNet__sample_weight = w)
        else: 
            pipe.fit(X_train_filtered, y_train)

        #Compute metrics
        #logging.info("Logging metrics...")
        y_train_pred = pipe.predict(X_train_filtered)
        metrics_train = compute_metrics(y_train, y_train_pred)
        train_prediction = pd.DataFrame({"ind": X_train_filtered.index, "true": y_train, "pred": y_train_pred})
        train_prediction.to_csv(f"results/9.multi_modal_models_gexp_hist/{tissue}/1_hist_gexp_en_train_prediction.csv")

        y_test_pred = pipe.predict(X_test_filtered)
        metrics_test = compute_metrics(y_test, y_test_pred)
        test_prediction = pd.DataFrame({"ind": X_test_filtered.index, "true": y_test, "pred": y_test_pred})
        test_prediction.to_csv(f"results/9.multi_modal_models_gexp_hist/{tissue}/1_hist_gexp_en_test_prediction.csv")

        for model_metric in metrics_train.keys(): 
            log_metric(model_metric + "_train", metrics_train[model_metric])

        for model_metric in metrics_test.keys(): 
            log_metric(model_metric + "_test", metrics_test[model_metric])

        #Plot model Fit
        plot_model_fit(y_train, 
                        y_train_pred, 
                        data_set="Train", 
                        fig_output_path= f"aging_notes/figures/9.multi_modal_models_gexp_hist/1_{tissue}_EN_hist_gexp_fit_train.pdf")


        plot_model_fit(y_test, 
                    y_test_pred, 
                    data_set="Test", 
                    fig_output_path= f"aging_notes/figures/9.multi_modal_models_gexp_hist/1_{tissue}_EN_hist_gexp_fit_test.pdf")


        #Save coefficients
        coef = pipe[1].coef_
        features = X_train_filtered.columns
        model_coef =  pd.DataFrame({"features":features, "coef" : coef})       
        model_coef.to_csv(f"results/9.multi_modal_models_gexp_hist/{tissue}/1_EN_hist_gexp_feature_importance.csv")

        # Save model 
        joblib.dump(pipe, f'results/9.multi_modal_models_gexp_hist/{tissue}/1_EN_hist_gexp.pipeline.pkl', compress = 1)

        mlflow.sklearn.autolog(disable = True) #autologs the model

    logging.info(f"Finished {tissue}")
    
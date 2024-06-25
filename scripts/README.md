## Analysis scripts

### Pre-processing script

Pre-process the metadata

- 1.preprocess folder 
    - Process metadata from each tissue (01_process_metadata_table.R)
    - Plot sample size (02_plot_sample_size.Rmd)
    - Split data into train and test data (03_train_test_split.ipynb)

### Aging clocks

To run this scripts, the data from Oliva et al. should be downloaded
Addional data required to run the Aging clock is provided

- 2.aging_clocks_benchmark 
    - Run Horvarth clock (1.run_Horvarth_model.Rmd)
    - Run Hannum clock (2.run_Hannum_model.rmd)
    - Run AltumAge (3.1 and 3.2 prepare and normalize the data. 3.3 to run the model in the GTEx data)
    - Missing probes analysis (4.missing_probes_analysis.R)

### Feature selection 

Apply feature selection to each tissue. 


- 3.feature_selection
    - Generate folds to apply models. Feature selection is applied in each of these folds and in the whole train set, to avoid data leakage when training the model (01.generate_folds.py )
    - Prepare methylation data to use in differential analysis (02.split_data_per_tissue.R )
    - Run differential analysis in each fold (03.model.R)
    - Run Elastic Net model in each tissue (04.fit_linear_model_on_feature_space.py)

Similar folders for Gene expression feature selection and for multi-omics samples are also provided (folder structures are the same)

### Models from each data type

To run models, mlflow is required


- Methylation and Gene expression models
    - Elastic net models (elastic_net.py)
    - LightGBM (lighGBM.py)

- Telemores
    - Split telemore data in tissues (1_split_telemore_data.R)
    - Train telemore data (2.train_telemore.ipynb)

- Neural networks
    Neural networks models are in its own folder. Open 7.neural_networks for more information. The main file is main.ipynb.

### Models integrating multiple data types 

For Elastic net and GBT (lightGBM) models

- Using methylation data (8_multi_omics_models_meth): 
    - Model with Methylation + Hist (1.methylation_hist_EN.py + 2.methylation_hist_GBT.py)
    - Model with Methylation + Gexp (3.methylation_gexp_EN.py + 4.methylation_gexp_GBT.py)
    - Models with Methylation + Gexp + Hist (5.methylation_gexp_hist_EN.py + 6.methylation_gexp_hist_GBT.py)
    - Models with Methylation + Gexp + Hist + TL (7.methylation_gexp_hist_tl_EN.py + 8.methylation_gexp_hist_tl_GBT.py)
    - Methylation downsample models (9.methylation_EN_downsample.py + 10.methylation_GBT_downsample.py)
    

- Using histology and Gene expression (9_multi_omics_gexp_hist)
    - Models with Gene expression and Histology (1.gexp_hist_EN.py + 2.gexp_hist_GBT.py)
    - Gene expression with downsample (3.gexp_downsample_en.py + 4.gtex_GBT_downsample.py)
    - Histology with downsample (5.hist_en_downsample.py + 6.hist_GBT_downsample.py)


For neural networks based models, see folder 7.neural_networks

### SHAP analysis 

- SHAP analysis in each data type: 
    - Methylation (shap_analysis_en.ipynb)
    - Gene expression (shap_analysis_gexp.ipynb)
    - Multi-modal (shap_analysis_multi_modal_EN.ipynb)
    
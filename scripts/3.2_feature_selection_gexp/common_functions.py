import pandas as pd

def load_gexp_data(file_name) -> pd.DataFrame: 
    """Split the dataset into train and test"""
    return pd.read_csv(file_name)

def split_in_train_test(gexp, age_data, test_set):
    """Split the dataset into train and test"""
    #Metadata
    metadata_test = age_data.loc[age_data["tissue_sample_id"].isin(test_set["sample_id"])]
    metadata_train = age_data.loc[~age_data["tissue_sample_id"].isin(test_set["sample_id"])]
    #Age data
    y_test = list(metadata_test["age"])
    y_train = list(metadata_train["age"])
    #Gexp data data
    X_train = gexp.loc[metadata_train.tissue_sample_id]
    X_test = gexp.loc[metadata_test.tissue_sample_id]

    return(X_train, X_test, y_train, y_test)

def load_folds(tissue, num_folds = 5): 
    num_folds = 5
    recreated_folds = []

    for i in range(num_folds):
        # Load train and test data from CSV files
        train_data = pd.read_csv(f'results/3.feature_selection_gexp/{tissue}/fold_{i}_train.csv', index_col=0)
        test_data = pd.read_csv(f'results/3.feature_selection_gexp/{tissue}/fold_{i}_test.csv', index_col=0)

        # Append the train and test index to the folds list
        recreated_folds.append([train_data.index, test_data.index])
    return recreated_folds

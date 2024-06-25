from pathlib import Path
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
import numpy as np
import pandas as pd
#import umap
import torch
#import os
from sklearn.decomposition import KernelPCA
import torch


def list_non_empty_dir_names(directory_path):
    path = Path(directory_path)
    non_empty_dir_names = [
        item.name for item in path.iterdir() if item.is_dir() and item.name != "embeddings" and any(item.iterdir())
    ]
    
    return non_empty_dir_names

class CustomStratifiedKFold(BaseCrossValidator):
    def __init__(self, df, patient_id_col, stratify_on='age', min_count=2, n_splits=5, random_state=None):
        self.df = df
        self.patient_id_col = patient_id_col
        self.stratify_on = stratify_on
        self.min_count = min_count
        self.n_splits = n_splits
        self.random_state = random_state

    # def split(self):
    #     np.random.seed(self.random_state)

    #     # Extract patient IDs and stratify column
    #     #patient_ids = self.df[self.patient_id_col].unique()
    #     stratify_values = self.df.groupby(self.patient_id_col)[self.stratify_on].first()

    #     # Split patient IDs into small and large groups based on stratify column counts
    #     counts = self.df[self.stratify_on].value_counts()
    #     large_strata = counts[counts > self.min_count].index
    #     large_group_patients = stratify_values[stratify_values.isin(large_strata)].index.values
    #     small_group_patients = stratify_values[~stratify_values.isin(large_strata)].index.values
        
    #     # Shuffle large group patient IDs for splitting
    #     np.random.shuffle(large_group_patients)
        
    #     # Calculate fold sizes for large groups, ensuring approximately equal distribution
    #     fold_sizes = np.full(self.n_splits, len(large_group_patients) // self.n_splits, dtype=int)
    #     fold_sizes[:len(large_group_patients) % self.n_splits] += 1
        
    #     current = 0
    #     for fold_size in fold_sizes:
    #         start, stop = current, current + fold_size
    #         test_ids = large_group_patients[start:stop]
            
    #         # Add a proportion of small group patients to each fold
    #         if len(small_group_patients) > 0:
    #             small_group_split = np.array_split(small_group_patients, self.n_splits)
    #             test_ids = np.concatenate((test_ids, small_group_split[0]))
    #             # Rotate small groups to ensure different small patients in each fold
    #             small_group_patients = np.roll(small_group_patients, -len(small_group_split[0]))
            
    #         train_ids = np.setdiff1d(large_group_patients, test_ids)
            
    #         assert len(list(set(train_ids).intersection(test_ids))) == 0
    #         # Yield patient IDs instead of DataFrame indices
    #         yield train_ids, test_ids
    #         current = stop
    def split(self):
        np.random.seed(self.random_state)

        # Extract patient IDs and stratify column
        stratify_values = self.df.groupby(self.patient_id_col)[self.stratify_on].first()

        # Split patient IDs into small and large groups based on stratify column counts
        counts = self.df[self.stratify_on].value_counts()
        large_strata = counts[counts > self.min_count].index
        large_group_patients = stratify_values[stratify_values.isin(large_strata)].index.values
        small_group_patients = stratify_values[~stratify_values.isin(large_strata)].index.values
        
        # Shuffle large group patient IDs for splitting
        np.random.shuffle(large_group_patients)
        
        # Calculate fold sizes for large groups, ensuring approximately equal distribution
        fold_sizes = np.full(self.n_splits, len(large_group_patients) // self.n_splits, dtype=int)
        fold_sizes[:len(large_group_patients) % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_ids = large_group_patients[start:stop]
            
            # For train IDs, include the rest of large group patients not in test_ids
            train_ids = np.setdiff1d(large_group_patients, test_ids)
            
            # Now add small group patients to the train set
            # This ensures small group patients are always in the training set
            train_ids = np.concatenate((train_ids, small_group_patients))
            
            assert len(list(set(train_ids).intersection(test_ids))) == 0
            # Yield patient IDs instead of DataFrame indices
            yield train_ids, test_ids
            current = stop

    def get_n_splits(self):
        return self.n_splits

def customStratifiedKFoldOld(df, stratify, k, random_state=42, shuffle=False):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)

    df_cv = df.groupby(["subject_id"])[stratify].max().reset_index()

    counts = df_cv.age.value_counts()
    non_rep_ages = list(counts[counts <= 4].index)

    non_rep_subjs = list(df_cv[df_cv.age.isin(non_rep_ages)].subject_id)

    X = df_cv[~df_cv["subject_id"].isin(non_rep_subjs)].drop(stratify, axis=1)
    y = df_cv[~df_cv["subject_id"].isin(non_rep_subjs)][stratify]

    assert len(set(X.subject_id).intersection(set(non_rep_subjs))) == 0

    kfolds = []
    for i, (train_index, val_index) in enumerate(skf.split(X, y)):
    
        train_subjects = list(X.iloc[train_index]["subject_id"]) + non_rep_subjs
        val_subjects = list(X.iloc[val_index]["subject_id"])

        assert len(set(train_subjects).intersection(val_subjects)) == 0
        
        kfolds.append((train_subjects, val_subjects))
    
    return kfolds

class CustomStratifiedKFoldOld:
    def __init__(
            self, 
            df, 
            stratify_on, 
            subject_id_col='subject_id', 
            n_splits=5, 
            random_state=None, 
            shuffle=False
            ):
        
        self.df = df
        self.stratify_col = stratify_on
        self.subject_id_col = subject_id_col
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle

    def split(self):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

        # Aggregate to find the maximum or representative value of the stratify column for each subject
        df_cv = self.df.groupby([self.subject_id_col])[self.stratify_col].max().reset_index()

        # Identify non-representative ages (or other stratify criteria) and their subjects
        counts = df_cv[self.stratify_col].value_counts()
        non_rep_values = counts[counts <= 4].index
        non_rep_subjects = df_cv[df_cv[self.stratify_col].isin(non_rep_values)][self.subject_id_col].tolist()

        # Prepare data for subjects not in non-representative groups
        X = df_cv[~df_cv[self.subject_id_col].isin(non_rep_subjects)].drop(self.stratify_col, axis=1)
        y = df_cv[~df_cv[self.subject_id_col].isin(non_rep_subjects)][self.stratify_col]

        # Ensure non-representative subjects are not mistakenly included
        assert len(set(X[self.subject_id_col]).intersection(set(non_rep_subjects))) == 0

        for train_index, test_index in skf.split(X, y):
            # Get subjects for training and validation
            train_subjects = X.iloc[train_index][self.subject_id_col].tolist() + non_rep_subjects
            test_subjects = X.iloc[test_index][self.subject_id_col].tolist()

            # Ensure there's no overlap between training and test subjects
            assert len(set(train_subjects).intersection(test_subjects)) == 0

            # Convert subject_ids back to original dataframe indices
            #train_ids = self.df[self.df[self.subject_id_col].isin(train_subjects)].index
            #test_ids = self.df[self.df[self.subject_id_col].isin(test_subjects)].index

            yield train_subjects, test_subjects
    
    def get_n_splits(self):
        return self.n_splits

class StandardScalerTransform:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, sample):
        # Convert PyTorch tensor to numpy array if necessary
        if isinstance(sample, torch.Tensor):
            sample = sample.numpy()
        self.scaler.fit(sample)

    def __call__(self, sample):
        if self.scaler is None:
            raise RuntimeError("Scaler has not been fitted.")
        # Convert PyTorch tensor to numpy array for transformation
        if isinstance(sample, torch.Tensor):
            sample = sample.numpy()
        transformed_sample = self.scaler.transform(sample)
        # Convert back to PyTorch tensor
        return torch.tensor(transformed_sample, dtype=torch.float32)

class QuantileTransform:
    def __init__(self, n_quantiles):
        self.scaler = QuantileTransformer(n_quantiles = n_quantiles, random_state=42, output_distribution = "normal")

    def fit(self, sample):
        # Convert PyTorch tensor to numpy array if necessary
        if isinstance(sample, torch.Tensor):
            sample = sample.numpy()
        self.scaler.fit(sample)

    def __call__(self, sample):
        if self.scaler is None:
            raise RuntimeError("Scaler has not been fitted.")
        # Convert PyTorch tensor to numpy array for transformation
        if isinstance(sample, torch.Tensor):
            sample = sample.numpy()
        transformed_sample = self.scaler.transform(sample)
        # Convert back to PyTorch tensor
        return torch.tensor(transformed_sample, dtype=torch.float32)

class UMAPTransform:
    def __init__(self, n_components=2):
        self.reducer = umap.UMAP(
            n_jobs=-1, 
            n_components=n_components, 
            #random_state=42
            )
        
        self.fitted = False

    def fit(self, sample):
        # Convert PyTorch tensor to numpy array if necessary
        if isinstance(sample, torch.Tensor):
            sample = sample.numpy()
        self.reducer.fit(sample)
        self.fitted = True

    def __call__(self, sample):
        if not self.fitted:
            raise RuntimeError("UMAP reducer has not been fitted.")
        # Convert PyTorch tensor to numpy array for transformation
        if isinstance(sample, torch.Tensor):
            sample = sample.numpy()
        transformed_sample = self.reducer.transform(sample)
        # Convert back to PyTorch tensor
        return torch.tensor(transformed_sample, dtype=torch.float32)

class KernelPCATransform:
    def __init__(self, n_components=2, kernel='rbf', **kwargs):
        """
        Initialize KernelPCA with the given parameters.
        """
        self.n_components = n_components
        self.kernel = kernel
        self.kwargs = kwargs
        self.kernel_pca = KernelPCA(n_components=self.n_components, kernel=self.kernel, **self.kwargs)
        self.fitted = False

    def fit(self, X):
        """
        Fit the KernelPCA model. Expected X to be a NumPy array.
        """
        self.kernel_pca.fit(X)
        self.fitted = True

    def __call__(self, sample):
        """
        Apply the KernelPCA transformation to the PyTorch tensor 'sample'.
        The sample is first converted to a NumPy array, transformed, and then converted back to a tensor.
        """
        if not self.fitted:
            raise RuntimeError("KernelPCA must be fitted before it can be used to transform data.")
        
        # Ensure sample is a NumPy array for transformation
        if isinstance(sample, torch.Tensor):
            sample = sample.numpy()
        
        transformed_sample = self.kernel_pca.transform(sample)
        
        # Convert back to tensor before returning
        return torch.tensor(transformed_sample, dtype=torch.float32)

class MethFeatures():
    def __init__(self, fold_number, technique, tissue):
        self.fold_number = fold_number
        self.technique = technique
        self.tissue = tissue
        if not self.fold_number and self.fold_number != 0:
            print(f"[+] {fold_number} -> Whole train.")
            self.folder = f"train"
        else:
            self.folder = f"fold_{self.fold_number}"
        
        if self.technique == "LIN":
            self.file = "linear_model_coef.csv"
        elif self.technique == "DML":
            self.file = "DML_results.csv"
        
        self.path = os.path.join(f"./data/methylation/3.feature_selection_multimodal/results/{self.tissue}/{self.folder}/{self.file}")
        self.data = None

    def _get_data(self):
        try:
            df_selected_features = pd.read_csv(self.path)
            if self.technique == "DML":
                df_selected_features = df_selected_features.rename({"Unnamed: 0": "probe"}, axis=1)
            self.data = df_selected_features.copy()
        except FileNotFoundError:
            print(f"File not found: {self.path}")

    def _apply_rule(self, N):
        if self.technique == "LIN":
            features_to_keep = self.data[self.data["coef"] != 0]
            top_features = (
                features_to_keep.loc[
                    features_to_keep["coef"]
                        .abs()
                        .sort_values(ascending=False)
                        .index
                    ]
                ).head(N)
            self.features = set(top_features["probe"])
        
        elif self.technique == "DML":
            features_to_keep = self.data[self.data["adj.P.Val"] < 0.05]
            sorted_features = (
                features_to_keep.assign(
                    abs_logFC=features_to_keep['logFC'].abs()
                    ).sort_values(by="abs_logFC", ascending=False)
                    )
            self.features = set(sorted_features.head(N)["probe"])

    def get_features(self, N):
        self._get_data()
        self._apply_rule(N)
        return self.features

class GeneFeatures():
    def __init__(self, fold_number, technique, tissue):
        self.fold_number = fold_number
        self.technique = technique
        self.tissue = tissue
        if not self.fold_number and self.fold_number != 0:
            print(f"[+] {fold_number} -> Whole train.")
            self.folder = f"train"
        else:
            self.folder = f"fold_{self.fold_number}"
        
        if self.technique == "LIN":
            self.file = "linear_model_coef_gexp.csv"
        elif self.technique == "DEG":
            self.file = "DEG_results.csv"
        
        self.path = os.path.join(f"./data/gene_expression/3.feature_selection_multimodal/results/{self.tissue}/{self.folder}/{self.file}")
        self.data = None

    def _get_data(self):
        try:
            df_selected_features = pd.read_csv(self.path)
            if self.technique == "DEG":
                df_selected_features = df_selected_features.rename({"Unnamed: 0": "probe"}, axis=1)
            self.data = df_selected_features.copy()
        except FileNotFoundError:
            print(f"File not found: {self.path}")

    def _apply_rule(self, N):
        if self.technique == "LIN":
            features_to_keep = self.data[self.data["coef"] != 0]
            top_features = (
                features_to_keep.loc[
                    features_to_keep["coef"]
                        .abs()
                        .sort_values(ascending=False)
                        .index
                    ]
                ).head(N)
            self.features = set(top_features["probe"])
        
        elif self.technique == "DEG":
            features_to_keep = self.data[self.data["adj.P.Val"] < 0.05]
            sorted_features = (
                features_to_keep.assign(
                    abs_logFC=features_to_keep['logFC'].abs()
                    ).sort_values(by="abs_logFC", ascending=False)
                    )
            self.features = set(sorted_features.head(N)["probe"])

    def get_features(self, N):
        self._get_data()
        self._apply_rule(N)
        return self.features

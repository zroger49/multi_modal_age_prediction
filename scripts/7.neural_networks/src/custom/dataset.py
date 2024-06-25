import numpy as np
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
#from ..custom import utils
import yaml
from sklearn.preprocessing import QuantileTransformer
def get_parameters(path):
    with open(path) as params:
        params_dict = yaml.safe_load(params)
    return params_dict

params__global = get_parameters("./conf/params__global.yml")

DEVICE = params__global['DEVICE']

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))
    
    return kernel_window

class SlideFeaturesDatasetWeighted(Dataset):
    def __init__(
            self, 
            features, 
            labels,
            max_target, 
            reweight='none', 
            transform=None,
            lds=False, 
            lds_kernel='gaussian', 
            lds_ks=5, # can try from 3 to 15
            lds_sigma=2 # can try from 0.5 to 3
            ):
        """
        Args:
            features (numpy.ndarray): Numpy array of features.
            labels (numpy.ndarray): Numpy array of labels.
            reweight (str): Method for reweighting the samples.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert features.shape[0] == labels.shape[0], "Features and labels must have the same number of samples"
        
        self.features = features
        self.labels = labels.astype('float32')
        self.transform = transform
        self.max_target = max_target + 1

        # self.weights = self._prepare_weights(
        #     reweight=reweight, 
        #     max_target=self.max_target
        #     )
        
        self.weights = self._prepare_weights(
            reweight=reweight,
            max_target=self.max_target,
            lds=lds, 
            lds_kernel=lds_kernel, 
            lds_ks=lds_ks, 
            lds_sigma=lds_sigma
            )
        

        if self.transform:
            self.features = self.transform(self.features)
        
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        # Get the feature and label at the specified index
        feature = self.features[index]
        label = self.labels[index]
        #label = np.expand_dims(self.labels[index], axis=0)
        
        # Get the weight for the sample
        weight = np.asarray(self.weights[index]).astype('float32') if self.weights is not None else np.asarray([np.float32(1.)])
        weight = torch.tensor(weight, dtype=torch.float32, device=DEVICE)
        return feature, label, weight

    def _prepare_weights(
            self, 
            reweight, 
            max_target,#51, 
            lds=False, 
            lds_kernel='gaussian', 
            lds_ks=5, 
            lds_sigma=2
            ):
        
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        value_dict = {x: 0 for x in range(max_target)}
        labels = self.labels.tolist()

        # mbr
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1

        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}

        elif reweight == 'inverse': # Vanilla method
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None
        #print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            #print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), 
                weights=lds_kernel_window, 
                mode='constant'
                )
            
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights

class SlideFeaturesDataset(Dataset):
    def __init__(
            self, 
            features, 
            labels,
            transform=None,
            ):
        """
        Args:
            features (numpy.ndarray): Numpy array of features.
            labels (numpy.ndarray): Numpy array of labels.
            reweight (str): Method for reweighting the samples.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert features.shape[0] == labels.shape[0], "Features and labels must have the same number of samples"
        
        self.features = features
        self.labels = labels.astype('float32')
        self.transform = transform

        if self.transform:
            self.features = self.transform(self.features)
        
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        # Get the feature and label at the specified index
        feature = self.features[index]
        label = self.labels[index]
        #label = np.expand_dims(self.labels[index], axis=0)
        feature = torch.tensor(feature, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return feature, label
    
class MethylationDataset(Dataset):

    def __init__(
            self, 
            features, 
            labels, 
            transform=None,
            ):
        
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        methylation = self.features[idx]
        age = self.labels[idx]
        age = torch.tensor(age, dtype=torch.float32)

        if self.transform:
            methylation = self.transform(methylation.unsqueeze(0)).squeeze(0)

        return methylation, age

class MethylationDatasetWeighted(Dataset):

    def __init__(
            self, 
            features, 
            labels, 
            max_target, 
            reweight='inverse', 
            lds=True,
            lds_kernel='gaussian', 
            lds_ks=5, # can try from 3 to 15
            lds_sigma=2, # can try from 0.5 to 3
            transform=None
            ):
        
        self.features = features
        self.labels = labels
        self.transform = transform
        self.max_target = max_target + 1
        
        self.weights = self._prepare_weights(
            reweight=reweight,
            max_target=self.max_target,
            lds=lds, 
            lds_kernel=lds_kernel, 
            lds_ks=lds_ks, 
            lds_sigma=lds_sigma
            )
        
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        methylation = self.features[idx]
        age = self.labels[idx]
        age = torch.tensor(age, dtype=torch.float32)

        if self.transform:
            methylation = self.transform(methylation.unsqueeze(0)).squeeze(0)

        weight = np.asarray(self.weights[idx]).astype('float32') if self.weights is not None else np.asarray([np.float32(1.)])
        weight = torch.tensor(weight, dtype=torch.float32, device=DEVICE)

        return methylation, age, weight
    
    def _prepare_weights(
            self, 
            reweight, 
            max_target,#51, 
            lds=False, 
            lds_kernel='gaussian', 
            lds_ks=5, 
            lds_sigma=2
            ):
        
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        value_dict = {x: 0 for x in range(max_target)}
        labels = self.labels.tolist()

        # mbr
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1

        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}

        elif reweight == 'inverse': # Vanilla method
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None
        #print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            #print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), 
                weights=lds_kernel_window, 
                mode='constant'
                )
            
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights
    
class CombinedDataset(Dataset):
    def __init__(
            self, 
            dataset1, 
            dataset2,
            labels,
            max_target, 
            reweight='inverse', 
            lds=True,
            lds_kernel='gaussian', 
            lds_ks=5, # can try from 3 to 15
            lds_sigma=2 # can try from 0.5 to 3
            ):
        # Ensure datasets are of the same size
        assert len(dataset1) == len(dataset2)
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.labels = labels.astype('float32')

        self.max_target = max_target + 1

        self.weights = self._prepare_weights(
            reweight=reweight,
            max_target=self.max_target,
            lds=lds, 
            lds_kernel=lds_kernel, 
            lds_ks=lds_ks, 
            lds_sigma=lds_sigma
            )
        
    def __len__(self):
        return len(self.dataset1)  # Assuming both datasets are of the same size

    def __getitem__(self, idx):
        x1, y1 = self.dataset1[idx]
        x2, y2 = self.dataset2[idx]
        
        x_combined = torch.cat((x1, x2), dim=-1)  
        
        weight = np.asarray(self.weights[idx]).astype('float32') if self.weights is not None else np.asarray([np.float32(1.)])

        x_combined = x_combined.clone().detach().to(dtype=torch.float32, device=DEVICE)
        y1 = y1.clone().detach().to(dtype=torch.float32, device=DEVICE)    
        weight = torch.tensor(weight, dtype=torch.float32, device=DEVICE)    
        return x_combined, y1, weight
    
    def _prepare_weights(
            self, 
            reweight, 
            max_target,#51, 
            lds=False, 
            lds_kernel='gaussian', 
            lds_ks=5, 
            lds_sigma=2
            ):
        
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        value_dict = {x: 0 for x in range(max_target)}
        labels = self.labels.tolist()

        # mbr
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1

        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}

        elif reweight == 'inverse': # Vanilla method
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None
        #print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            #print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), 
                weights=lds_kernel_window, 
                mode='constant'
                )
            
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights

class CustomDataset():
    def __init__(self, X_train, X_val, y_train, y_val, modality, **params):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

        self.normalize = params.get('normalize', False)
        self.reweight = params.get('reweight', "sqrt_inv")
        self.kernel = params.get('lds_kernel', 'gaussian')
        self.ks = params.get('lds_ks', 2)
        self.sigma = params.get('lds_sigma', 1)
        self.batch_size = params.get('batch_size', 16)

        self.modality = modality

    def _prepare(self):
        if self.modality == "histology":
            if self.normalize:
                scaler = StandardScaler().fit(self.X_train)
                self.X_train = scaler.transform(self.X_train)
                self.X_val = scaler.transform(self.X_val)

            self.train_dataset = SlideFeaturesDatasetWeighted(
                features=self.X_train, 
                labels=self.y_train,
                max_target=int(self.y_train.max()),
                reweight=self.reweight,
                lds=True,
                lds_kernel=self.kernel,
                lds_ks=self.ks,
                lds_sigma=self.sigma,
                transform=None
            )

            self.val_dataset = SlideFeaturesDatasetWeighted(
                features=self.X_val, 
                labels=self.y_val,
                max_target=int(self.y_train.max()),
                transform=None
            )

        elif self.modality == "methylation":
            X_train_np = self.X_train.cpu().numpy()
            X_val_np = self.X_val.cpu().numpy()
            quantile_transform = QuantileTransformer(
                n_quantiles=X_train_np.shape[0], 
                random_state=42, 
                output_distribution="normal"
            )
            
            self.X_train_np = quantile_transform.fit_transform(X_train_np)
            self.X_val_np = quantile_transform.transform(X_val_np)

            self.X_train = torch.from_numpy(self.X_train_np).float()
            self.X_val = torch.from_numpy(self.X_val_np).float()

            self.train_dataset = MethylationDatasetWeighted(
                features=self.X_train, 
                labels=self.y_train,
                max_target=int(self.y_train.max()),
                reweight=self.reweight,
                lds=True,
                lds_kernel=self.kernel,
                lds_ks=self.ks,
                lds_sigma=self.sigma,
                # transform=transform
                )
            
            self.val_dataset = MethylationDatasetWeighted(
                features=self.X_val, 
                labels=self.y_val,
                max_target=int(self.y_train.max()),
                # transform=transform
                )

        elif self.modality == "gene_expression":
        
            X_train_np = self.X_train.cpu().numpy()
            X_val_np = self.X_val.cpu().numpy()
            quantile_transform = QuantileTransformer(
                n_quantiles=X_train_np.shape[0], 
                random_state=42, 
                output_distribution="normal"
            )
            
            self.X_train_np = quantile_transform.fit_transform(X_train_np)
            self.X_val_np = quantile_transform.transform(X_val_np)

            self.X_train = torch.from_numpy(self.X_train_np).float()
            self.X_val = torch.from_numpy(self.X_val_np).float()

            self.train_dataset = MethylationDatasetWeighted(
                features=self.X_train, 
                labels=self.y_train,
                max_target=int(self.y_train.max()),
                reweight=self.reweight,
                lds=True,
                lds_kernel=self.kernel,
                lds_ks=self.ks,
                lds_sigma=self.sigma,
                # transform=transform
                )
            
            self.val_dataset = MethylationDatasetWeighted(
                features=self.X_val, 
                labels=self.y_val,
                max_target=int(self.y_train.max()),
                # transform=transform
                )

    def get_dataloader(self):
        self._prepare()
        train_loader_hist = DataLoader(self.train_dataset, self.batch_size, shuffle=False)
        val_loader_hist = DataLoader(self.val_dataset, self.batch_size, shuffle=False)

        return train_loader_hist, val_loader_hist
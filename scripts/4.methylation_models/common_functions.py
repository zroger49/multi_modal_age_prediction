import pandas as pd
import numpy as np

from joblib import Memory
from scipy.stats import linregress
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

from scipy.ndimage import convolve1d

memory = Memory('cache')

@memory.cache
def load_lung_data() -> pd.DataFrame:
    return pd.read_csv("data/methylation_lung.csv")

@memory.cache
def load_ovary_data() -> pd.DataFrame:
    return pd.read_csv("data/methylation_ovary.csv")

def split_in_train_test(meth, age_data, test_set):
    """Split the dataset into train and test"""
    #Metadata
    metadata_test = age_data.loc[age_data["tissue_sample_id"].isin(test_set["sample_id"])]
    metadata_train = age_data.loc[~age_data["tissue_sample_id"].isin(test_set["sample_id"])]
    #Age data
    y_test = list(metadata_test["AGE"])
    y_train = list(metadata_train["AGE"])
    #Methylation data
    meth_t = meth.transpose()
    X_train = meth_t.loc[metadata_train.Sample_ID]
    X_test = meth_t.loc[metadata_test.Sample_ID]

    return(X_train, X_test, y_train, y_test)

def load_folds(tissue, num_folds = 5): 
    num_folds = 5
    recreated_folds = []

    for i in range(num_folds):
        # Load train and test data from CSV files
        train_data = pd.read_csv(f'results/3.feature_selection/{tissue}/fold_{i}_train.csv', index_col=0)
        test_data = pd.read_csv(f'results/3.feature_selection/{tissue}/fold_{i}_test.csv', index_col=0)

        # Append the train and test index to the folds list
        recreated_folds.append([train_data.index, test_data.index])
    return recreated_folds

def filter_M_and_XY_probes(meth_data, probe_info, tissue): 
    probe_info = probe_info[probe_info["CHR_hg38"].notna()] #Remove NA probes in the CHR_hg38 genome
    if tissue == "lung" or tissue == "colon_transverse": 
        # Filter out probes in the X, Y chromossomes
        chrxy_probes = probe_info[(probe_info['CHR_hg38'] == 'chrX') | (probe_info['CHR_hg38'] == 'chrY')]['Name'].tolist()
        meth_data = meth_data[~meth_data.index.isin(chrxy_probes)]
        
    # Filter ou probes in the M chromossome
    chrM_probes = probe_info[(probe_info['CHR_hg38'] == 'chrM')]['Name'].tolist()
    meth_data = meth_data[~meth_data.index.isin(chrM_probes)]

    return meth_data

def convert_beta_to_m(meth):
    """Convert beta values to M values"""
    # Assuming all columns in the DataFrame are beta values
    beta_columns = meth.columns

    # Create a list to store DataFrames for each column
    m_dfs = []

    for beta_column in beta_columns:
        if beta_column == "probe":
            m_dfs.append(meth[beta_column])
        else:
            m_dfs.append(pd.DataFrame(np.log2(meth[beta_column] / (1 - meth[beta_column])), columns=[beta_column]))

    # Concatenate all DataFrames in the list along axis=1
    m_dataframe = pd.concat(m_dfs, axis=1)

    return m_dataframe.copy()

def compute_metrics(y_test, y_test_pred):
    metrics = {}
    metrics["r2"] = r2_score(y_test, y_test_pred)
    metrics["rmse"] = mean_squared_error(y_test, y_test_pred, squared=False)
    metrics["mae"] = mean_absolute_error(y_test, y_test_pred)
    metrics["med"] = median_absolute_error(y_test, y_test_pred)
    lr = linregress(y_test, y_test_pred)
    metrics["slope"] = lr.slope
    metrics["intercept"] = lr.intercept
    metrics["cor"] = lr.rvalue
    return metrics

def iterate_metrics_metrics(metrics, _metrics): 
    """Add the model metrics to a dictionary"""
    for model_metric in _metrics.keys(): 
        if model_metric not in metrics.keys(): 
            metrics[model_metric] = []
        metrics[model_metric].append(_metrics[model_metric])
    return metrics

def plot_model_fit(y_test, y_test_pred,
                data_modality="DNA Methylation", 
                data_set="Validation",
                smoker_status=None, 
                color="#55812C",
                model = "Elastic Net",
                title_override=None,
                fig_output_path="scatterplot_methylation.pdf"):
        
    metrics = compute_metrics(y_test, y_test_pred)

    # kind="reg" is not supported when using hue;
    # as a workaround we plot scatter separately.
    if smoker_status is not None:
        scatter = False
    else:
        scatter = True

    jointgrid = sns.jointplot(x=y_test, y=y_test_pred,
                                kind="reg",
                                truncate=False,
                                scatter=scatter,
                                fit_reg=True,
                                color=color,
                                xlim=(20, 70),
                                ylim=(20, 70))
    
    jointgrid.ax_joint.axline([0, 0], [1, 1], transform=jointgrid.ax_joint.transAxes,
                                linestyle="--", alpha=0.8, color='darkgray')

    if smoker_status is not None:
        sns.scatterplot(x=y_test, y=y_test_pred, hue=smoker_status, ax=jointgrid.ax_joint)
        sns.move_legend(jointgrid.ax_joint, "lower right")

    if title_override:
        plt.title(title_override)
    else:
        plt.title(f"{model} Model Fit for {data_modality} - {data_set} (N = {len(y_test)})")
    jointgrid.ax_joint.set_ylabel("Predicted Values")
    jointgrid.ax_joint.set_xlabel("True Value")
    t = plt.text(.05, .7,
                    'r²={:.3f}\nrmse={:.3f}\nmae={:.3f}\nmed={:.3f}\nslope={:.3f}\nintercept={:.3f}\ncor={:.3f}'.format(
                        metrics["r2"], metrics["rmse"], metrics["mae"], metrics["med"], metrics["slope"],
                        metrics["intercept"], metrics["cor"]),
                    transform=jointgrid.ax_joint.transAxes)
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor=color))
    jointgrid.fig.subplots_adjust(top=0.95)
    plt.tight_layout()
    if fig_output_path:
        plt.savefig(fig_output_path, format="pdf", bbox_inches="tight")


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

def prepare_weights(labels, reweight, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
    assert reweight in {'none', 'inverse', 'sqrt_inv'}
    assert reweight != 'none' if lds else True, \
        "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

    max_target = max(labels)
    value_dict = {x: 0 for x in range(max_target)}
    for label in labels:
        value_dict[min(max_target - 1, int(label))] += 1
    if reweight == 'sqrt_inv':
        value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
    elif reweight == 'inverse':
        value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
    num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
    if not len(num_per_label) or reweight == 'none':
        return None
    #print(f"Using re-weighting: [{reweight.upper()}]")

    if lds:
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        #print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
        smoothed_value = convolve1d(
            np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
        num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

    weights = [np.float32(1 / x) for x in num_per_label]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    return weights
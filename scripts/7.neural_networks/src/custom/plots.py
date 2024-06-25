import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, median_absolute_error #,root_mean_squared_error
import seaborn as sns
from scipy.stats import pearsonr, linregress
import pandas as pd

MAIN_COLOR = '#7FC97F'

def plot_scatter_cv(lt_y_true_cv, lt_y_preds_cv):
    # Number of plots
    K = len(lt_y_true_cv)
    
    # Setup figure and axes for the grid
    fig, axs = plt.subplots(1, K, figsize=(5*K, 8))  # Adjust figsize as needed
    
    for i in range(K):
        try:
            y_test = lt_y_true_cv[i]
            y_pred = lt_y_preds_cv[i]
            
            # # Skip if data contains NaN
            # if np.isnan(y_test).any() or np.isnan(y_pred).any():
            #     raise ValueError("Data contains NaNs")
            
            # Calculate metrics
            slope, intercept, r_value, _, _ = stats.linregress(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            med = median_absolute_error(y_test, y_pred)
            
            # Prepare for plotting
            combined_ages = np.concatenate([y_test, y_pred])
            min_age, max_age = np.min(combined_ages) - 5, np.max(combined_ages) + 5
            
            # Plotting
            axs[i].scatter(y_test, y_pred, alpha=0.6, label='True vs. Predicted')
            axs[i].plot(y_test, slope * y_test + intercept, color='red', label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
            axs[i].plot([min_age, max_age], [min_age, max_age], color='gray', linestyle='--', label='45° Reference Line')
            axs[i].set_xlim(min_age, max_age)
            axs[i].set_ylim(min_age, max_age)
            axs[i].set_xlabel('True Value')
            axs[i].set_ylabel('Predicted Value')
            axs[i].set_title(f'Fold {i+1}')
            axs[i].set_aspect('equal', adjustable='box')
            axs[i].legend(title=f'Corr: {r_value:.2f}\nSlope: {slope:.2f}\nIntercept: {intercept:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%\nR2: {r2:.2f}\nRMSE: {rmse:.2f}\nMED: {med:.2f}', loc='best')
            
        except Exception as e:
            # Plot an empty figure for the fold that fails
            axs[i].text(0.5, 0.5, 'Data not available\nor contains NaNs', horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes)
            axs[i].set_title(f'Fold {i+1}')
            axs[i].set_xlabel('True Value')
            axs[i].set_ylabel('Predicted Value')
            axs[i].set_xlim(0, 1)
            axs[i].set_ylim(0, 1)
            print(f"Error in fold {i+1}: {e}")
            
    plt.tight_layout()
    return fig

def plot_scatter(y_test, y_pred):
    # Calculate metrics
    slope, intercept, r_value, _, _ = stats.linregress(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    med = median_absolute_error(y_test, y_pred)

    # Prepare for plotting
    combined_ages = np.concatenate([y_test, y_pred])
    min_age, max_age = np.min(combined_ages) - 5, np.max(combined_ages) + 5

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.6, label='True vs. Predicted')
    ax.plot(y_test, slope * y_test + intercept, color='red', label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
    ax.plot([min_age, max_age], [min_age, max_age], color='gray', linestyle='--', label='45° Reference Line')
    ax.set_xlim(min_age, max_age)
    ax.set_ylim(min_age, max_age)
    ax.set_xlabel('True Age')
    ax.set_ylabel('Predicted Age')
    ax.set_title('True vs. Predicted Age Scatter Plot with Fitted Line')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(title=f'Corr: {r_value:.2f}\nSlope: {slope:.2f}\nIntercept: {intercept:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%\nR2: {r2:.2f}\nRMSE: {rmse:.2f}\nMED: {med:.2f}')

    return fig

def plot_hist(df_train,df_val):
    import seaborn as sns
    import pandas as pd
    # Combine train, val, test sets into a single DataFrame with an additional column for set type
    df_train['set'] = 'train'
    df_val['set'] = 'val'
    combined_df = pd.concat([df_train, df_val])

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # Plot 1: Age count
    sns.histplot(
        data=combined_df, 
        x='age', 
        hue='set', 
        element='step', 
        bins=30, 
        common_norm=False,
        kde=True,
        ax=axes[0]
    )
    axes[0].set_title('Age Count across Train, Validation, and Test Sets')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Count')

    # Plot 2: Age Distribution
    sns.histplot(
        data=combined_df, 
        x='age', 
        hue='set', 
        element='step', 
        stat='density', 
        bins=30, 
        common_norm=False,
        kde=True,
        ax=axes[1]
    )
    axes[1].set_title('Age Distribution across Train, Validation, and Test Sets')
    axes[1].set_xlabel('Age')
    axes[1].set_ylabel('Density')

    plt.tight_layout()
    plt.show()
    plt.close()

def compute_metrics(y_true, y_pred):
    metrics = {}
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['rmse'] = root_mean_squared_error(y_true, y_pred)
    slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)
    metrics['slope'] = slope
    metrics['intercept'] = intercept
    correlation, _ = pearsonr(y_true, y_pred)
    metrics['cor'] = correlation
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['med'] = median_absolute_error(y_true, y_pred)
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
        #plt.title(f"{model} Model Fit for {data_modality} - {data_set} (N = {len(y_test)})")
        plt.title(f"{model} | {data_modality} - {data_set} (N = {len(y_test)})")
    jointgrid.ax_joint.set_ylabel("Predicted Values")
    jointgrid.ax_joint.set_xlabel("True Value")
    t = plt.text(.05, .7,
                    'r²={:.3f}\nrmse={:.3f}\nmae={:.3f}\nmed={:.3f}\nslope={:.3f}\nintercept={:.3f}\ncor={:.3f}'.format(
                        metrics["r2"], metrics["rmse"], metrics["mae"], metrics["med"], metrics["slope"],
                        metrics["intercept"], metrics["cor"]),
                    transform=jointgrid.ax_joint.transAxes)
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor=color))
    jointgrid.figure.subplots_adjust(top=0.95)
    plt.tight_layout()
    if fig_output_path:
        plt.savefig(fig_output_path, format="pdf", bbox_inches="tight")
    else:
        plt.show()

def plot_age_data(y_train, y_test):
    y_train = y_train.to_frame("age").copy()
    y_test = y_test.to_frame("age").copy()
    y_train['set'] = 'train'
    y_test['set'] = 'test'
    
    combined_df = pd.concat([y_train, y_test])
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    
    # Plot 1: Age count
    sns.histplot(
        data=combined_df, 
        x='age', 
        hue='set', 
        element='step', 
        bins=30, 
        common_norm=False,
        kde=True,
        ax=axes[0]
    )
    axes[0].set_title('Age Count across Train and Test Sets With Lines')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Count')

    # Plot 2: Age Distribution
    sns.histplot(
        data=combined_df, 
        x='age', 
        hue='set', 
        element='step', 
        stat='density', 
        bins=30, 
        common_norm=False,
        kde=True,
        ax=axes[1]
    )
    axes[1].set_title('Age Distribution across Train and Test Sets With Lines')
    axes[1].set_xlabel('Age')
    axes[1].set_ylabel('Density')
    
    plt.tight_layout()
    
    plt.figure(figsize=(20, 6))
    sns.countplot(
        data=combined_df,         
        x='age', 
        hue='set'
    )
    plt.title('Age Count across Train and Test Sets')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.5)
    plt.tight_layout() 
    
    plt.show()



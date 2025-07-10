from FEniCS_simulations.toy_model.config import RANDOM_SEED, SCALER_DIR, TRAIN_MODE
import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import joblib
import os
from pathlib import Path
from matplotlib import rcParams



rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9
})

def latexify_param(name):
    """
    Convert parameter names into LaTeX-friendly format with proper subscripts.
    Handles both Greek-letter parameters (alpha_0 → $\alpha_0$) 
    and q-parameters (q1 → $q_1$).
    """
    greek_map = {
        'alpha': r'\alpha',
        'beta': r'\beta',
        'gamma': r'\gamma',
        'delta': r'\delta'
    }
    
    # Handle Greek-letter parameters (alpha_0, beta_1, etc.)
    for greek, latex in greek_map.items():
        if name.startswith(greek):
            parts = name.split('_')
            if len(parts) > 1:
                return rf'${latex}_{{{parts[1]}}}$'  # $\alpha_{0}$
            return rf'${latex}$'  # fallback if no subscript
    
    # Handle q-parameters (q1, q2, etc.)
    if name.startswith('q') and name[1:].isdigit():
        return rf'$q_{{{name[1:]}}}$'  # $q_{1}$
    
    # Handle existing underscores (for other parameters)
    if '_' in name:
        base, sub = name.split('_', 1)
        return rf'${base}_{{{sub}}}$'
    
    # Handle specific cardiac parameters
    cardiac_params = {
        'edv': 'EDV',
        'esv': 'ESV', 
        'ef': 'EF',
        'sv': 'SV'  # Stroke Volume
}
    if name.lower() in cardiac_params:
        return cardiac_params[name.lower()]
        
    return rf'${name}$'  # fallback for all other cases


def load_and_split_data(path, q_dim=4, y_dim=1, shuffle=True, random_state=RANDOM_SEED):
    """
    Load data from an EXCEL file and split it into features and target variables.

    Args:
        path (_type_): Path to the EXCEL file.
        q_dim (int, optional): Number of feature columns. Defaults to 4.
        y_dim (int, optional): Number of target columns. Defaults to 6.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        random_state (_type_, optional): Random state for reproducibility. Defaults to RANDOM_SEED.

    Raises:
        ValueError: Mismatch in sample count between q and y

    Returns:
        q_data (pd.DataFrame): Feature data.
        y_data (pd.DataFrame): Target data.
        y_cols (list): Output feature's name
    """

    df = pd.read_csv(path)

    if df.shape[1] != q_dim + y_dim:
        raise ValueError(f"Expected {q_dim + y_dim} columns, but got {df.shape[1]}")

    if shuffle:
        df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    q_df = df.iloc[:, :q_dim].copy()
    y_df = df.iloc[:, q_dim:].copy()

    y_cols = y_df.columns.to_list()

    q_data = np.array(q_df)
    y_data = np.array(y_df)

    assert q_data.shape[0] == y_data.shape[0], "Mismatch in sample count between q and y"

    return q_data, y_data, y_cols


def split_data(q_data, y_data, test_size=0.1, random_state=RANDOM_SEED):
    """
    Splits the data into training and testing sets.

    Args:
        q_data (pd.DataFrame): Feature data.
        y_data (pd.DataFrame): Target data.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.1.
        random_state (int, optional): Random state for reproducibility. Defaults to RANDOM_SEED.

    Returns:
        q1 (np.ndarray): Training feature data. Non-scaled.
        q2 (np.ndarray): Testing feature data. Non-scaled.
        y1 (np.ndarray): Training target data. Non-scaled.
        y2 (np.ndarray): Testing target data. Non-scaled.
    """
    
    q1, q2, y1, y2 = train_test_split(
        q_data, y_data, test_size=test_size, random_state=random_state, shuffle=True   # changed manually (default = True)
    )
    return q1, q2, y1, y2


def normalize_data(q1, q2, y1, y2, train_new_scalers=True):
    """
    Normalize the data using StandardScaler.

    Args:
        q1 (np.ndarray): Training feature data (unscaled).
        q2 (np.ndarray): Testing feature data (unscaled).
        y1 (np.ndarray): Training target data (unscaled).
        y2 (np.ndarray): Testing target data (unscaled).
        scaler_dir (Path or str): Directory path to save/load scalers.
        train_new_scalers (bool): If True, trains and saves new scalers. If False, loads existing scalers.

    Returns:
        X_train (np.ndarray): Scaled training feature data.
        X_test (np.ndarray): Scaled testing feature data.
        y_train (np.ndarray): Scaled training target data.
        y_test (np.ndarray): Scaled testing target data.
        q_scaler (StandardScaler): Feature scaler.
        y_scaler (StandardScaler): Target scaler.
    """

    q_scaler_path = os.path.join(SCALER_DIR, "q_scaler.pkl")
    y_scaler_path = os.path.join(SCALER_DIR, "y_scaler.pkl")

    if train_new_scalers:
        q_scaler = StandardScaler()
        y_scaler = StandardScaler()

        # Save scalers
        os.makedirs(SCALER_DIR, exist_ok=True)
        joblib.dump(q_scaler, q_scaler_path)
        joblib.dump(y_scaler, y_scaler_path)
    else:
        q_scaler = joblib.load(q_scaler_path)
        y_scaler = joblib.load(y_scaler_path)

    # Apply transformations
    X_train = q_scaler.fit_transform(q1)
    X_test = q_scaler.transform(q2)
    y_train = y_scaler.fit_transform(y1)
    y_test = y_scaler.transform(y2)

    return X_train, X_test, y_train, y_test, q_scaler, y_scaler


def load_or_train_gp_regressor(X_train, y_train, y_cols, save_dir, kernel=None, train=TRAIN_MODE):
    """
    Load or train Gaussian Process Regression models for each target column.

    Args:
        X_train (np.ndarray): Feature training data.
        y_train (np.ndarray): Target training data (2D array).
        y_cols (list): List of names for each output column.
        save_dir (str or Path): Directory path to save/load models.
        kernel (sklearn.gaussian_process.kernels.Kernel, optional): Custom kernel. Defaults to Matern.
        train (bool): Whether to train new models or load existing ones.

    Returns:
        dict: Dictionary of trained or loaded GPR models, keyed by column name.
    """
    os.makedirs(save_dir, exist_ok=True)

    gp_regressor = {}

    for i, col in enumerate(y_cols):
        model_path = os.path.join(save_dir, f"gp_regressor_{col}.pkl")

        if train:
            if kernel is None:
                kernel = Matern(length_scale=1.0, nu=2.5)  # Default Matern kernel

            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)
            gpr.fit(X_train, y_train[:, i])

            joblib.dump(gpr, model_path)
            gp_regressor[col] = gpr
        else:
            gp_regressor[col] = joblib.load(model_path)

    return gp_regressor


def evaluate_gp(model, X_test, y_scaler, return_std=False): 
    """Returns predictions (and std if requested) from each GPR model. 
    Serves as a lightweight utility for getting raw predictions from models."""

    n_targets = len(model)
    gpr_pred = np.zeros((X_test.shape[0], n_targets))
    std_pred = np.zeros_like(gpr_pred) if return_std else None

    for idx, model in enumerate(model.values()):
        if return_std:
            preds, stds = model.predict(X_test, return_std=True)
            gpr_pred[:, idx], std_pred[:, idx] = preds, stds
        else:
            gpr_pred[:, idx] = model.predict(X_test)

    # Always apply inverse scaling
    gpr_pred = y_scaler.inverse_transform(gpr_pred)
    if return_std:
        std_pred = std_pred * y_scaler.scale_  # correct std devs after inverse scaling

    return gpr_pred, std_pred


def compute_metrics(y_true, y_pred, y_cols):
    mse_dict = {col: mean_squared_error(y_true[:, i], y_pred[:, i]) for i, col in enumerate(y_cols)}
    r2_dict = {col: r2_score(y_true[:, i], y_pred[:, i]) for i, col in enumerate(y_cols)}
    return mse_dict, r2_dict


def predict_with_gp(gpr_models, X_test, y_true, y_cols, y_scaler,
                    return_std=True, return_metrics=True, verbose=False):

    gpr_pred, std_pred = evaluate_gp(gpr_models, X_test, y_scaler, return_std)

    mse_dict, r2_dict = None, None
    if return_metrics:
        mse_dict, r2_dict = compute_metrics(y_true=y_true, y_pred=gpr_pred, y_cols=y_cols)

        if verbose:
            print("GPR Evaluation Metrics:")
            for col in y_cols:
                print(f"  {col}: MSE = {mse_dict[col]:.4f}, R² = {r2_dict[col]:.4f}")


    return {
    "pred": gpr_pred,
    "std": std_pred,
    "mse": mse_dict,
    "r2": r2_dict
}


def add_performance_metrics(ax, mse, r2, feat, min_val, max_val):
    """Adds MSE and R² annotation to the plot."""
    if mse is None or r2 is None:
        return
    if feat not in mse or feat not in r2:
        return

    text_x = min_val + 0.7 * (max_val - min_val)
    text_y = min_val + 0.1 * (max_val - min_val)

    ax.text(
        text_x, text_y,
        rf"$\mathrm{{MSE}}={mse[feat]:.5g}$" + "\n" + rf"$R^2={r2[feat]:.5f}$",
        fontsize=9,
        bbox=dict(facecolor='white', alpha=0.8)
    )


def save_figure(fig, plot_name, save_dir, formats=('pdf', 'png')):
    """
    Saves figure in multiple formats and optionally saves associated data arrays.

    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        plot_name (str): Base name (without extension).
        save_dir (str or Path): Directory to save into.
        formats (tuple): File formats to save (pdf, png, svg, etc).
        data_dict (dict): Optional dictionary with arrays to save.
    """
    save_dir = Path(save_dir)
    fig_dir = save_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        fig.savefig(
            fig_dir / f"{plot_name}.{fmt}",
            dpi=300,
            bbox_inches='tight',
            metadata={"Title": plot_name}
        )


def plot_performance(y_true, y_pred, y_cols, std_pred, mse=None, r2=None, 
                     title=None, save_name=None, save_dir=None):
    """
    Plots true vs. predicted values for multiple targets with optional metrics and error bars.

    Args:
        y_true (ndarray): Ground truth values, shape (n_samples, n_features)
        y_pred (ndarray): Predicted values, shape (n_samples, n_features)
        y_cols (list): Feature names, length = n_features
        std_pred (ndarray, optional): Standard deviation of predictions
        mse (dict, optional): Dictionary with MSE per feature
        r2 (dict, optional): Dictionary with R² per feature
        title (str): Figure title
        save_name (str, optional): Base name for saving (without extension)
        save_dir (str or Path): Base directory to save into
    """
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have same shape"
    assert y_true.shape[1] == len(y_cols), "Mismatch between y_cols and y_true.shape[1]"
    if std_pred is not None:
        assert std_pred.shape == y_pred.shape, "std_pred must match y_pred shape"
    if mse or r2:
        assert all(k in mse and k in r2 for k in y_cols), "mse/r2 keys must match y_cols"

    n_outputs = len(y_cols)
    rows = 1#int(np.ceil(n_outputs / 3))
    cols = min(3, n_outputs)
    fig, axes = plt.subplots(rows, cols, figsize=(8, 5 * rows))
    axes = axes.flatten() if n_outputs > 1 else [axes]

    for i, feat in enumerate(y_cols):
        ax = axes[i]
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]
        std_vals = std_pred[:,i] if std_pred is not None else None

        # Plot predicted vs. true
        ax.scatter(
            true_vals,
            pred_vals,
            alpha=0.6,
            color='dodgerblue',
            edgecolor='white',
            s=60,
            label='Prediction',
            zorder=3
        )

        # Optionally add uncertainty
        if std_pred is not None:
            ax.errorbar(
                true_vals,
                pred_vals,
                yerr=1.96 * std_vals,
                fmt='none',
                ecolor='gray',
                alpha=0.4,
                capsize=2,
                zorder=2
            )
        

        # Perfect fit line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        margin = 0.1 * (max_val - min_val)
        ax.plot(
            [min_val - margin, max_val + margin],
            [min_val - margin, max_val + margin],
            'r--',
            label='Perfect Fit',
            linewidth=1,
            zorder=1
        )

        ax.set_title(latexify_param(feat), fontsize=12)
        ax.set_xlabel('True Value', fontsize=10)
        ax.set_ylabel('Predicted Value', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper left', fontsize=9)

        add_performance_metrics(ax, mse, r2, feat, min_val, max_val)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_name:
        save_figure(
            fig=fig,
            plot_name=save_name,
            save_dir=save_dir,
            formats=('pdf', 'svg')
        )












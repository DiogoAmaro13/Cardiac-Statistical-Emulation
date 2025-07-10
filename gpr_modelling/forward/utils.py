from matplotlib import rcParams
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import seaborn as sns
from datetime import datetime
import joblib
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

from gpr_modelling.forward.config import RANDOM_SEED, TRAIN_MODE, FEATURE_COLUMNS, TARGET_COLUMNS, MODEL_DIR

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
    
    return rf'${name}$'  # fallback for all other cases


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


def save_figure_and_data(fig, plot_name, save_dir, formats=('pdf', 'png'), data_dict=None):
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

    if data_dict:
        data_dir = save_dir / "plot_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for key, array in data_dict.items():
            header = ','.join([f"{key}_{i}" for i in range(array.shape[1])]) if array.ndim == 2 else key
            np.savetxt(
                data_dir / f"{plot_name}_{key}.csv",
                array,
                delimiter=',',
                header=header,
                comments=''
            )


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


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types with improved numerical handling"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.int16)):
            return int(obj)
        return super().default(obj)
    

def load_gpr_models(gpr_path):
    gpr_models = {}
    for filename in os.listdir(gpr_path):
        if filename.endswith(".pkl"):
            feat_name = filename.replace("gpr_model_", "").replace(".pkl", "")
            model_path = os.path.join(gpr_path, filename)
            loaded_data = joblib.load(model_path)
            gpr_models[feat_name] = loaded_data['model']

    # Then create a new ordered dictionary
    ordered_models = {}
    for i in range(3):  # For alpha/beta_0 through _2
        ordered_models[f'alpha_{i}'] = gpr_models[f'alpha_{i}']
        ordered_models[f'beta_{i}'] = gpr_models[f'beta_{i}']

    return ordered_models


def load_scalers(scalers_dir):
    """
    Load q_scaler and y_scaler from given directory.

    Args:
        scaler_dir (str): Path to the directory containing 'q_scaler.pkl' and 'y_scaler.pkl'.

    Returns:
        tuple: (q_scaler, y_scaler)
    """
    q_scaler_path = os.path.join(scalers_dir, 'q_scaler.pkl')
    y_scaler_path = os.path.join(scalers_dir, 'y_scaler.pkl')

    q_scaler_dict = joblib.load(q_scaler_path)
    y_scaler_dict = joblib.load(y_scaler_path)

    q_scaler = q_scaler_dict['scaler']
    y_scaler = y_scaler_dict['scaler']

    return q_scaler, y_scaler


def evaluate_gp(gpr_models, X_test, y_scaler, return_std=False): 
    """Returns predictions (and std if requested) from each GPR model. 
    Serves as a lightweight utility for getting raw predictions from models."""

    n_targets = len(gpr_models)
    gpr_pred = np.zeros((X_test.shape[0], n_targets))
    std_pred = np.zeros_like(gpr_pred) if return_std else None

    for idx, model in enumerate(gpr_models.values()):
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


def inverse_scale_predictions(preds, stds, y_scaler):

    if y_scaler is not None:
        preds = y_scaler.inverse_transform(preds)
        stds = stds * y_scaler.scale_ if stds is not None else None
        return preds, stds


def compute_metrics(y_true, y_pred, y_cols):
    mse_dict = {col: mean_squared_error(y_true[:, i], y_pred[:, i]) for i, col in enumerate(y_cols)}
    r2_dict = {col: r2_score(y_true[:, i], y_pred[:, i]) for i, col in enumerate(y_cols)}
    return mse_dict, r2_dict


def save_predictions_csv(y_true, y_pred, stds, y_cols, save_path, file_name="gpr_predictions"):

    save_path = Path(save_path)
    fig_dir = save_path / "predictions"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = [y_true, y_pred] + ([stds] if stds is not None else [])
    columns = (
        [f"true_{col}" for col in y_cols] +
        [f"pred_{col}" for col in y_cols] +
        ([f"std_{col}" for col in y_cols] if stds is not None else [])
    )
    df = pd.DataFrame(np.hstack(data), columns=columns)
    df.to_csv(fig_dir / f"{file_name}.csv", index=False)


def save_metrics(mse_dict, r2_dict, y_cols, save_path_prefix, file_name = "gpr_metrics"):

    save_path = Path(save_path_prefix)
    fig_dir = save_path / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)


    df = pd.DataFrame({
        "Parameter": y_cols,
        "MSE": [mse_dict[col] for col in y_cols],
        "R²": [r2_dict[col] for col in y_cols]
    })

    df.to_csv(fig_dir / f"{file_name}.csv", index=False)
    df.to_latex(fig_dir / f"{file_name}.tex", index=False, float_format="%.4f")


def save_metadata(train_mode=TRAIN_MODE, 
                      feature_names=FEATURE_COLUMNS, 
                      target_names=TARGET_COLUMNS, 
                      seed=RANDOM_SEED,
                      save_path='.',
                      model_paths=f"{MODEL_DIR}"):
    
    save_path = Path(save_path)
    fig_dir = save_path / "execution_info"
    fig_dir.mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "Last Execution": datetime.now().isoformat(),
        "Train Mode": train_mode,
        "Features": feature_names,
        "Targets": target_names,
        "Random Seed": seed,
        "Model Paths": model_paths,
    }

    
    with open(fig_dir / "run_info.json", "w") as f:
        json.dump(run_metadata, f, indent=4)



# ======================== Start of inverse utils ==================================

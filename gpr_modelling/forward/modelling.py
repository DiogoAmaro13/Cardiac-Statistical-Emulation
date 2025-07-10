import os
import numpy as np
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import joblib

from gpr_modelling.forward.utils import compute_metrics, save_predictions_csv, save_metrics, load_gpr_models, evaluate_gp

def load_or_train_gp_models(X_train, y_train, y_cols, save_dir, kernel=None, train=False):
    """
    Train Gaussian Process Regression models for each target variable in y_train.
    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target variables.
        y_cols (list): List of target variable names.
        save_dir (str): Directory to save the trained models.
        kernel (sklearn.gaussian_process.kernels.Kernel, optional): Kernel for GPR. Defaults to Matern kernel.
    Returns:
        dict: Dictionary of trained GPR models.
    """
    if train:
        os.makedirs(save_dir, exist_ok=True)
        if kernel is None:
            #kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
            kernel = Matern(length_scale=1.0, nu=2.5)
        
        gpr_models = {}

        for i, col in enumerate(y_cols):
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)
            gpr.fit(X_train, y_train[:, i])

            model_info = {
                'model': gpr,
                'training_stats': {
                    'X_mean': X_train.mean(axis=0),
                    'X_std': X_train.std(axis=0),
                    'y_mean': y_train[:, i].mean(),
                    'y_std': y_train[:, i].std(),
                },
                'timestamp': datetime.now().isoformat()
            }
            gpr_models[col] = gpr
            joblib.dump(model_info, os.path.join(save_dir, f"gpr_model_{col}.pkl"))

        return gpr_models
    else:
        gpr_models = load_gpr_models(save_dir)
        return gpr_models


def predict_with_gp(gpr_models, X_test, y_true, y_cols, y_scaler,
                    return_std=True, return_metrics=True, verbose=False, save_dir='.'):

    gpr_pred, std_pred = evaluate_gp(gpr_models, X_test, y_scaler, return_std)

    mse_dict, r2_dict = None, None
    if return_metrics:
        mse_dict, r2_dict = compute_metrics(y_true=y_true, y_pred=gpr_pred, y_cols=y_cols)

        if verbose:
            print("GPR Evaluation Metrics:")
            for col in y_cols:
                print(f"  {col}: MSE = {mse_dict[col]:.4f}, R² = {r2_dict[col]:.4f}")

    if save_dir and y_true is not None:
        save_predictions_csv(y_true, gpr_pred, std_pred, y_cols, save_path=save_dir)
        
        if return_metrics:
            save_metrics(mse_dict, r2_dict, y_cols, save_path_prefix=save_dir)

    return {
    "pred": gpr_pred,
    "std": std_pred,
    "mse": mse_dict,
    "r2": r2_dict
}

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
from gpr_modelling.forward.utils import load_scalers
from gpr_modelling.forward.config import SCALER_DIR, RANDOM_SEED



def load_and_split_data(path, q_dim=4, y_dim=6, shuffle=True, random_state=RANDOM_SEED):
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

    df = pd.read_excel(path, index_col=0)

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


def normalize_data(q1, q2, y1, y2, train_new=False, save_dir=None):
    """
    Normalize the data using StandardScaler by loading pre-defined scalers.

    Args:
        q1 (np.ndarray): Training feature data. Non-scaled.
        q2 (np.ndarray): Testing feature data. Non-scaled.
        y1 (np.ndarray): Training target data. Non-scaled.
        y2 (np.ndarray): Testing target data. Non-scaled.

    Returns:
        X_train (np.ndarray): Scaled training feature data.
        X_test (np.ndarray): Scaled testing feature data.
        y_train (np.ndarray): Scaled training target data.
        y_test (np.ndarray): Scaled testing target data.
        q_scaler (StandardScaler): Scaler for feature data.
        y_scaler (StandardScaler): Scaler for target data.
    """
    if train_new:
        q_scaler = StandardScaler().fit(q1)
        y_scaler = StandardScaler().fit(y1)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            joblib.dump(q_scaler, os.path.join(save_dir, "q_scaler.pkl"))
            joblib.dump(y_scaler, os.path.join(save_dir, "y_scaler.pkl"))
    else:
        scaler_dir = SCALER_DIR

        q_scaler, y_scaler = load_scalers(scaler_dir)

        X_train = q_scaler.fit_transform(q1)
        X_test = q_scaler.transform(q2)
        y_train = y_scaler.fit_transform(y1)
        y_test = y_scaler.transform(y2)

        return X_train, X_test, y_train, y_test, q_scaler, y_scaler

import pytest
from pathlib import Path
from sklearn.base import BaseEstimator
import json

from gpr_modelling.forward.config import MODEL_DIR, SCALER_DIR, PROCESSED_DATA_DIR
from gpr_modelling.forward.data import load_and_split_data, split_data, normalize_data
from gpr_modelling.forward.utils import load_gpr_models, load_scalers, save_metadata
from gpr_modelling.forward.modelling import predict_with_gp, load_or_train_gp_models


# ================ Data ops unit tests ============================
def test_data_loading():
    data_path = Path(PROCESSED_DATA_DIR) / "datasets.xlsx"
    q_data, y_data, y_cols = load_and_split_data(data_path)

    assert q_data.shape[0] == y_data.shape[0], "q_data and y_data must have same number of rows"
    assert len(y_cols) > 0, "No target columns found"


def test_data_splitting():
    data_path = Path(PROCESSED_DATA_DIR) / "datasets.xlsx"
    q_data, y_data, _ = load_and_split_data(data_path)
    q1, q2, y1, y2 = split_data(q_data, y_data)

    assert q1.shape[0] == y1.shape[0], "Mismatch between q1 and y1"
    assert q2.shape[0] == y2.shape[0], "Mismatch between q2 and y2"
    assert q1.shape[0] + q2.shape[0] == q_data.shape[0], "Total rows mismatch"


def test_normalization_shapes():
    data_path = Path(PROCESSED_DATA_DIR) / "datasets.xlsx"
    q_data, y_data, _ = load_and_split_data(data_path)
    q1, q2, y1, y2 = split_data(q_data, y_data)

    q_train, q_test, y_train, y_test, q_scaler, y_scaler = normalize_data(q1, q2, y1, y2)

    assert q_train.shape == q1.shape
    assert q_test.shape == q2.shape
    assert y_train.shape == y1.shape
    assert y_test.shape == y2.shape


# =========== Test GPR Model Loading and Specs ===========
def test_load_gpr_model():
    
    models = load_gpr_models(MODEL_DIR)
    
    assert isinstance(models, dict), "Expected a dictionary of models"
    assert len(models) > 0, "No models were loaded"
    for name, model in models.items():
        assert hasattr(model, "predict"), f"Model {name} does not have predict method"
        assert hasattr(model, "kernel_")


# =========== Test Scaler Loading ===========
def test_load_scalers():
    q_scaler, y_scaler = load_scalers(SCALER_DIR) 

    assert q_scaler is not None, "q_scaler is None"
    assert y_scaler is not None, "y_scaler is None"


    assert isinstance(q_scaler, BaseEstimator), "q_scaler is not an sklearn estimator"
    assert isinstance(y_scaler, BaseEstimator), "y_scaler is not an sklearn estimator"

    assert hasattr(q_scaler, "transform"), "q_scaler has no 'transform' method"
    assert hasattr(y_scaler, "transform"), "y_scaler has no 'transform' method"



# ================= Predictions test ==========================
def test_model_prediction():
    data_path = Path(PROCESSED_DATA_DIR) / "datasets.xlsx"
    q_data, y_data, y_cols = load_and_split_data(data_path)
    q1, q2, y1, y2 = split_data(q_data, y_data)

    q_train, q_test, y_train, y_test, q_scaler, y_scaler = normalize_data(q1, q2, y1, y2)

    models = load_or_train_gp_models(q_train, y_train, y_cols, train=False, save_dir=MODEL_DIR)

    results = predict_with_gp(models, q_test, y2, y_cols, y_scaler, return_std=True, return_metrics=True, verbose=False)

    assert "pred" in results
    assert results["pred"].shape == y2.shape

    if "std" in results:
        assert results["std"].shape == y2.shape

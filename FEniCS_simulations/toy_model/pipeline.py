import time
import logging
from pathlib import Path
from FEniCS_simulations.toy_model.utils import (
    load_and_split_data,
    split_data,
    normalize_data,
    load_or_train_gp_regressor,
    predict_with_gp,
    plot_performance
)
from FEniCS_simulations.toy_model.config import (
    DATA_DIR,
    MODEL_DIR,
    TOY_MODEL_RESULTS_DIR
)


class GPToyModelPipeline:
    def __init__(self, train_gp: bool = False):
        self.train_gp = train_gp
        self.data_path = DATA_DIR / "lhs_dataset.csv"
        self.model_dir = MODEL_DIR
        self.results_dir = TOY_MODEL_RESULTS_DIR

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
        self.logger = logging.getLogger(__name__)

    def run(self):
        self.logger.info("Starting GP Toy Model Pipeline")

        # --- Load and normalize data ---
        q_data, y_data, y_cols = load_and_split_data(self.data_path)
        q1, q2, y1, y2 = split_data(q_data, y_data)
        X_train, X_test, y_train, y_test, q_scaler, y_scaler = normalize_data(
            q1, q2, y1, y2, train_new_scalers=False
        )
        self.logger.info("Data loaded and normalized successfully")

        # --- Train or load GP model ---
        regressor = load_or_train_gp_regressor(
            X_train, y_train, y_cols, save_dir=self.model_dir, train=self.train_gp
        )

        # --- Predict and time the inference ---
        self.logger.info("Starting predictions")
        start_time = time.time()
        results = predict_with_gp(
            regressor, X_test, y2, y_cols, y_scaler,
            return_std=True, return_metrics=True, verbose=False
        )
        elapsed_time = time.time() - start_time
        self.logger.info(f"Predictions completed in {elapsed_time:.6f} seconds")

        # --- Plot performance ---
        plot_performance(
            y2,
            y_pred=results['pred'],
            y_cols=y_cols,
            std_pred=results['std'],
            mse=results['mse'],
            r2=results['r2'],
            title=None,
            save_name="toy_model_performance",
            save_dir=self.results_dir
        )

        self.logger.info("Pipeline finished successfully")

from pathlib import Path
from gpr_modelling.forward.config import PROCESSED_DATA_DIR, MODEL_DIR, FORWARD_RESULTS_DIR, TRAIN_MODE
from gpr_modelling.forward.data import load_and_split_data, split_data, normalize_data
from gpr_modelling.forward.modelling import load_or_train_gp_models, predict_with_gp
from gpr_modelling.forward.plotting import plot_predictions, revised_plot_gp_predicted_relationships
from gpr_modelling.forward.utils import save_metadata
from gpr_modelling.logger import get_logger



class GPRForwardPipeline:
    def __init__(self, data_path=None, train_mode=TRAIN_MODE, model_dir=MODEL_DIR, results_dir=FORWARD_RESULTS_DIR):
        self.logger = get_logger()
        self.train_mode = train_mode
        self.data_path = data_path or (Path(PROCESSED_DATA_DIR) / "datasets.xlsx")
        self.model_dir = model_dir
        self.results_dir = results_dir

    def run(self):
        if not self.data_path.exists():
            self.logger.error("Data path not found.")
            return
        else:
            self.logger.info("Path found and data loaded. Processing starting.")

        q_data, y_data, y_cols = load_and_split_data(self.data_path)

        q1, q2, y1, y2 = split_data(q_data=q_data, y_data=y_data)

        q_train, q_test, y_train, y_test, q_scaler, y_scaler = normalize_data(q1=q1, q2=q2, y1=y1, y2=y2)

        gpr_models = load_or_train_gp_models(
            X_train=q_train, y_train=y_train, y_cols=y_cols,
            save_dir=self.model_dir, train=self.train_mode
        )

        if self.train_mode:
            self.logger.info("Model training successfully initialized.")
        else:
            self.logger.info("Models loaded from disk.")

        dict_results = predict_with_gp(
            gpr_models=gpr_models,
            X_test=q_test,
            y_true=y2,
            y_cols=y_cols,
            y_scaler=y_scaler,
            return_std=True,
            return_metrics=True,
            verbose=False,
            save_dir=self.results_dir
        )

        self.logger.info(f"Predictions and metrics saved to: {self.results_dir}")

        plot_predictions(
            y_true=y2,
            y_pred=dict_results["pred"],
            y_cols=y_cols,
            std_pred=dict_results.get("std", None),
            mse=dict_results.get("mse", None),
            r2=dict_results.get("r2", None),
            title=None,
            save_name="gpr_predictions",
            save_dir=self.results_dir
        )

        save_metadata(save_path=self.results_dir)
        self.logger.info("Plot saved and metadata written.")
        self.logger.info("Forward pipeline completed.")


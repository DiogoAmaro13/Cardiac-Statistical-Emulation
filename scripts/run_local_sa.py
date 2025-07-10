from gpr_modelling.sensitivity.local_sa import univariate_sa
from gpr_modelling.forward.utils import load_gpr_models, load_scalers
from gpr_modelling.forward.config import MODEL_DIR, SCALER_DIR, SA_RESULTS_DIR


def main():
    """
    Main function to run local sensitivity analysis using pre-trained GPR models.
    """
    # Load models and scalers
    gpr_models = load_gpr_models(MODEL_DIR)
    q_scaler, y_scaler = load_scalers(SCALER_DIR)

    # Run univariate sensitivity analysis
    univariate_sa(
        gpr_models=gpr_models,
        q_scaler=q_scaler,
        bounds=[[0.1, 5]] * 4,
        num_points=200,
        output_dir= SA_RESULTS_DIR / "local_sa",
    )

    print("Local sensitivity analysis successfully executed.")
    
if __name__ == "__main__":
    main()
from gpr_modelling.sensitivity.global_sa import sensitivity_analysis_gp, plot_sobol_results
from gpr_modelling.forward.config import MODEL_DIR, SCALER_DIR, SA_RESULTS_DIR
from gpr_modelling.forward.utils import load_gpr_models, load_scalers

def main():
    """
    Main function to run global sensitivity analysis using pre-trained GPR models.
    """
    # Load models and scalers
    gpr_models = load_gpr_models(MODEL_DIR)
    q_scaler, y_scaler = load_scalers(SCALER_DIR)

    # Run Sobol sensitivity analysis
    sa_results = sensitivity_analysis_gp(
        gpr_models=gpr_models,
        q_scaler=q_scaler,
        sample_size=1024  # should be a power of 2 for Saltelli
    )

    # Plot and save results
    plot_output_dir = SA_RESULTS_DIR / "global_sa"
    plot_sobol_results(sa_results, plot_output_dir)

    print("Global sensitivity analysis successfully executed.")

if __name__ == "__main__":
    main()

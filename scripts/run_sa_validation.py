from gpr_modelling.forward.config import MODEL_DIR, SCALER_DIR, SA_RESULTS_DIR
from gpr_modelling.sensitivity.validation import kde_analysis, plot_sobol_convergence
from gpr_modelling.forward.utils import load_gpr_model, load_scalers

def main():
    """
    Main function to run sensitivity analysis validation using pre-trained GPR models.
    """
    # Load models and scalers
    gpr_models = load_gpr_model(MODEL_DIR)
    q_scaler, y_scaler = load_scalers(SCALER_DIR)

    # Define the output names
    y_cols = list(gpr_models.keys())

    # Define input bounds and parameters
    input_bounds = {
        'q1': [0.1, 5],
        'q2': [0.1, 5],
        'q3': [0.1, 5],
        'q4': [0.1, 5]
    }

    # Define the problem
    problem = {
        'num_vars': 4,
        'names': list(input_bounds.keys()),
        'bounds': list(input_bounds.values())
    }
    
    # Perform KDE analysis
    kde_analysis(
        q_scaler=q_scaler,
        y_scaler=y_scaler,
        gpr_models=gpr_models,
        y_cols=y_cols,
        input_bounds=input_bounds,
        save_dir=SA_RESULTS_DIR / "sa_validation",
        num_samples=200
    )

    # Perform Sobol convergence analysis
    plot_sobol_convergence(
        gpr_models=gpr_models,
        problem=problem,
        q_scaler=q_scaler,
        sample_sizes=None,    # None for default values of [64, 128, 256, 512, 1024]; must be equal to 2^n
        output_dir=SA_RESULTS_DIR / "sa_validation"
    )
    print("Sensitivity analysis validation successfully executed.")

if __name__ == "__main__":
    main()
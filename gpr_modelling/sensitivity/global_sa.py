import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SALib.analyze import sobol
from SALib.sample import saltelli
import json

from gpr_modelling.forward.utils import latexify_param, load_gpr_models, load_scalers
from gpr_modelling.forward.config import MODEL_DIR, SCALER_DIR, SA_RESULTS_DIR

gpr_models = load_gpr_models(MODEL_DIR)
q_scaler, y_scaler = load_scalers(SCALER_DIR)


def evaluate_model(params):
        predictions = np.zeros((params.shape[0], len(gpr_models)))
        std_devs = np.zeros_like(predictions)
        
        for i, (feat, model) in enumerate(gpr_models.items()):
            pred, std = model.predict(params, return_std=True)
            predictions[:, i] = pred.ravel()
            std_devs[:, i] = std.ravel()
            
        return y_scaler.inverse_transform(predictions)


def sensitivity_analysis_gp(gpr_models, q_scaler, sample_size):
    """
    Robust Sobol sensitivity analysis with validation checks.
    
    Args:
        gpr_models: Dict of trained GP models {output_name: model}
        q_scaler: Fitted StandardScaler for inputs
        y_scaler: Fitted StandardScaler for outputs
        sample_size: Base sample count (N) for Saltelli sampling
        
    Returns:
        Dict containing Sobol indices and validation metrics
    """
    # 1. Problem definition with bounds checking
    problem = {
        'num_vars': 4,
        'names': ['q1', 'q2', 'q3', 'q4'],
        'bounds': [[0.1, 5]]*4 
    }
    
    # 2. Generate samples with sample size validation
    if sample_size > 2048:
        print("Warning: Large sample size may cause memory issues")
    param_values = saltelli.sample(problem, sample_size)
    
    # 3. Input scaling with bounds verification
    norm_pm = q_scaler.transform(param_values)
    
    # 4. Model evaluation
    Y = evaluate_model(norm_pm)
    
    # 5. Sobol analysis with validation
    sobol_indices = {'results': {}, 'validation': {}}
    
    for feat in gpr_models:
        Si = sobol.analyze(problem, Y[:, list(gpr_models.keys()).index(feat)], 
                          print_to_console=False)
        
        # Validation checks
        if np.any(Si['S1'] < -0.01):
            print(f"Warning: Negative S1 for {feat}")
        if np.any(Si['ST'] < Si['S1'] - 0.01):
            print(f"Warning: ST < S1 for {feat}")
        
        # Store results
        sobol_indices['results'][feat] = {
            'S1': np.maximum(Si['S1'], 0).tolist(),
            'ST': np.maximum(Si['ST'], 0).tolist(),
            'S2': Si['S2'].tolist() if Si['S2'] is not None else None
        }
        with open(SA_RESULTS_DIR/"global_sa"/"sobol_indices.json", 'w') as f:
            json.dump(sobol_indices, f)

    return sobol_indices


def plot_sobol_results(sa_results, output_dir):
    """Generate all diagnostic plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. First-order and total-order indices
    for feat in sa_results['results']:
        df = pd.DataFrame({
            'First-Order': sa_results['results'][feat]['S1'],
            'Total-Order': sa_results['results'][feat]['ST']
        }, index=[f'$q_{i+1}$' for i in range(4)])
        
        ax = df.plot(kind='bar', figsize=(10, 5), width=0.8)
        ax.set_title(f"Sobol Indices for {latexify_param(feat)}", pad=20)
        ax.set_ylabel("Sensitivity Index", labelpad=10)
        ax.set_xlabel("Input Parameters", labelpad=10)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sobol_{feat}.pdf", dpi=300)
        plt.close()
        
        # 2. Interaction heatmaps if available
        #if sa_results['results'][feat]['S2'] is not None:
        #    plt.figure(figsize=(6, 5))
        #    sns.heatmap(
        #        np.array(sa_results['results'][feat]['S2']),
        #        annot=True, fmt=".2f",
        #        xticklabels=[f'$q_{i+1}$' for i in range(4)],
        #        yticklabels=[f'$q_{i+1}$' for i in range(4)],
        #        cmap="vlag", center=0,
        #        square=True, cbar_kws={'label': 'Interaction Strength'}
        #    )
        #    plt.title(f"Second-Order Interactions\n{latexify_param(feat)}", pad=15)
        #    plt.tight_layout()
        #    plt.savefig(f"{output_dir}/interactions_{feat}.pdf", dpi=300)
        #    plt.close()
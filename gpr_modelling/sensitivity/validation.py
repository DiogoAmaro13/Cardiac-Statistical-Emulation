import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from SALib.analyze import sobol
from SALib.sample import saltelli

from gpr_modelling.sensitivity.global_sa import evaluate_model
from gpr_modelling.forward.utils import latexify_param, NumpyEncoder


def kde_analysis(q_scaler, y_scaler, gpr_models, y_cols, input_bounds, save_dir, num_samples):
    """
    Enhanced KDE analysis with sensitivity validation features.
    
    Args:
        q_scaler: Fitted StandardScaler for inputs
        y_scaler: Fitted StandardScaler for outputs  
        input_bounds: Dict of parameter bounds {'q1': [min,max], ...}
        save_dir: Directory to save plots
        num_samples: Number of points per parameter sweep
    """
    os.makedirs(save_dir, exist_ok=True)
    input_params = list(input_bounds.keys())
    

    for vary_param in input_params:
        # Create parameter sweep
        X_fixed = np.ones((num_samples, len(input_params)))
        vary_idx = input_params.index(vary_param)
        varied_values = np.linspace(*input_bounds[vary_param], num_samples)
        X_fixed[:, vary_idx] = varied_values
        
        # Scale and predict
        X_scaled = q_scaler.transform(X_fixed)
        preds = np.zeros((num_samples, len(y_cols)))
        
        for i, feat in enumerate(y_cols):
            preds[:, i]= gpr_models[feat].predict(X_scaled, return_std=False)
        
        preds = y_scaler.inverse_transform(preds)
        
        # Identify alpha and beta parameters
        alpha_indices = [i for i, name in enumerate(y_cols) if 'alpha' in name]
        beta_indices = [i for i, name in enumerate(y_cols) if 'beta' in name]

        # Plotting
        for group, indices in [('alpha', alpha_indices), ('beta', beta_indices)]:
            plt.figure(figsize=(10, 6))
            
            for i in indices:
                # KDE plot
                sns.kdeplot(preds[:, i], fill=True, alpha=0.3, 
                            label=f"{latexify_param(y_cols[i])}")
                
            
            plt.title(f"Effect of {latexify_param(vary_param)} on ${group}$ parameters")
            plt.xlabel(f"{latexify_param(vary_param)} value")
            plt.ylabel("Output density")
            plt.legend(title="Parameters:", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/kde_{vary_param}_{group}.pdf", bbox_inches='tight', dpi=300)
            plt.close()



def plot_sobol_convergence(gpr_models, problem, q_scaler, 
                          sample_sizes, output_dir):
    """
    Test and plot convergence of Sobol indices across sample sizes.
    
    Args:
        gpr_models: Dictionary of trained GP models {output_name: model}
        problem: SALib problem dictionary
        q_scaler: Fitted input scaler
        y_scaler: Fitted output scaler
        sample_sizes: List of sample sizes to test (default: [64, 128, 256, 512, 1024])
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_sizes = sample_sizes or [64, 128, 256, 512, 1024]
    
    # Initialize results storage
    convergence_data = {
        feat: {
            'S1': np.zeros((len(sample_sizes), problem['num_vars'])),
            'ST': np.zeros((len(sample_sizes), problem['num_vars'])),
            'elapsed_times': []
        } 
        for feat in gpr_models
    }

    # Run convergence tests
    for i, N in enumerate(sample_sizes):
        print(f"Testing N={N}...")
        start_time = time.time()
        
        param_values = saltelli.sample(problem, N)
        Y = evaluate_model(q_scaler.transform(param_values))
        
        for feat in gpr_models:
            Si = sobol.analyze(
                problem, 
                Y[:, list(gpr_models.keys()).index(feat)],
                print_to_console=False
            )
            convergence_data[feat]['S1'][i] = np.maximum(Si['S1'], 0)
            convergence_data[feat]['ST'][i] = np.maximum(Si['ST'], 0)
        
        elapsed = time.time() - start_time
        for feat in gpr_models:
            convergence_data[feat]['elapsed_times'].append(elapsed/len(gpr_models))

    # Generate convergence plots
    param_labels = [f"${latexify_param(name)}$" for name in problem['names']]
    
    for feat in gpr_models:
        plt.figure(figsize=(12, 6))
        
        # S1 convergence
        plt.subplot(1, 2, 1)
        for j in range(problem['num_vars']):
            plt.plot(sample_sizes, convergence_data[feat]['S1'][:, j], 
                     'o-', label=param_labels[j])
        
        plt.xscale('log')
        plt.xlabel('Sample Size (log scale)')
        plt.ylabel('First-Order Index (S1)')
        plt.title(f'S1 Convergence: {latexify_param(feat)}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # ST convergence
        plt.subplot(1, 2, 2)
        for j in range(problem['num_vars']):
            plt.plot(sample_sizes, convergence_data[feat]['ST'][:, j],
                     's--', label=param_labels[j])
        
        plt.xscale('log')
        plt.xlabel('Sample Size (log scale)')
        plt.ylabel('Total-Order Index (ST)')
        plt.title(f'ST Convergence: {latexify_param(feat)}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/convergence_{feat}.pdf", dpi=300)
        plt.close()
    
    # Save numerical results
    with open(f"{output_dir}/convergence_data.json", 'w') as f:
        json.dump({
            feat: {
                'S1': convergence_data[feat]['S1'].tolist(),
                'ST': convergence_data[feat]['ST'].tolist(),
                'elapsed_times': convergence_data[feat]['elapsed_times'],
                'sample_sizes': sample_sizes
            }
            for feat in gpr_models
        }, f, indent=4, cls=NumpyEncoder)
    
    return convergence_data
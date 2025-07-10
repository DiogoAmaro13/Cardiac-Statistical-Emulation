import os
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

from gpr_modelling.forward.utils import latexify_param, load_gpr_models, load_scalers
from gpr_modelling.forward.config import MODEL_DIR, SCALER_DIR
from gpr_modelling.sensitivity.global_sa import evaluate_model

gpr_models = load_gpr_models(MODEL_DIR)
q_scaler, y_scaler = load_scalers(SCALER_DIR)

"""
def univariate_sa(
    gpr_models, q_scaler,
    bounds, num_points, output_dir
):
    os.makedirs(output_dir, exist_ok=True)
    param_names = [f"q{i+1}" for i in range(4)]
    output_names = list(gpr_models.keys())

    for i in range(4):
        q_vals = np.linspace(bounds[i][0], bounds[i][1], num_points)
        all_params = np.ones((num_points, 4)) 
        all_params[:, i] = q_vals  

        # Scale inputs
        scaled_params = q_scaler.transform(all_params)

        # Predict outputs
        preds = evaluate_model(scaled_params) 

        # Plot each output
        plt.figure(figsize=(12, 6))
        texts=[]
        for j in range(preds.shape[1]):
            plt.plot(q_vals, preds[:, j], label=f"${latexify_param(output_names[j])}$")

            x_text = q_vals[-1]
            y_text = preds[-1, j]
            texts.append(plt.text(
                x_text,
                y_text,
                f"${latexify_param(output_names[j])}$",
                fontsize=10,
                color=plt.gca().lines[-1].get_color()
            ))

        adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle='-', color='gray'))

        plt.xlabel(f"${latexify_param(param_names[i])}$")
        plt.ylabel("Output")
        plt.yscale("log")
        plt.title(f"Univariate Sensitivity Sweep: ${latexify_param(param_names[i])}$")
        plt.grid(True, linestyle="--", alpha=0.6)
        # plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/vary_{param_names[i]}.pdf", dpi=300)
        plt.close()
"""


def univariate_sa(gpr_models, q_scaler, bounds, num_points, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    param_names = [f"q{i+1}" for i in range(4)]
    output_names = list(gpr_models.keys())

    for i in range(4):
        q_vals = np.linspace(bounds[i][0], bounds[i][1], num_points)
        all_params = np.ones((num_points, 4))
        all_params[:, i] = q_vals

        # Scale inputs
        scaled_params = q_scaler.transform(all_params)

        # Predict outputs
        preds = evaluate_model(scaled_params)

        # Separate outputs into α and β
        alpha_indices = [j for j, name in enumerate(output_names) if "alpha" in name]
        beta_indices = [j for j, name in enumerate(output_names) if "beta" in name]

        def plot_group(indices, group_label):
            plt.figure(figsize=(10, 5))
            texts = []

            for j in indices:
                y_vals = preds[:, j]
                color = plt.gca()._get_lines.get_next_color()
                plt.plot(q_vals, y_vals, label=f"${latexify_param(output_names[j])}$", color=color)
                # Add floating label near end
                texts.append(plt.text(
                    q_vals[-1],
                    y_vals[-1],
                    f"${latexify_param(output_names[j])}$",
                    fontsize=9,
                    color=color
                ))

            adjust_text(
                texts,
                only_move={'points': 'y', 'texts': 'y'},
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5)
            )

            plt.xlabel(f"${latexify_param(param_names[i])}$")
            plt.ylabel("Model Output")
            plt.title(f"Univariate LSA Sweep: {group_label} vs ${latexify_param(param_names[i])}$")
            plt.grid(True, linestyle="--", alpha=0.5)
            #plt.yscale("log")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/vary_{param_names[i]}_{group_label.lower()}.pdf", dpi=300)
            plt.close()

        plot_group(alpha_indices, "Alpha")
        plot_group(beta_indices, "Beta")
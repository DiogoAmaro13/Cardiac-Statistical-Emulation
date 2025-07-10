import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from gpr_modelling.forward.utils import save_figure_and_data, add_performance_metrics, latexify_param


def plot_predictions(y_true, y_pred, y_cols, std_pred, mse=None, r2=None, 
                     title=None, save_name=None, save_dir=None):
    """
    Plots true vs. predicted values for multiple targets with optional metrics and error bars.

    Args:
        y_true (ndarray): Ground truth values, shape (n_samples, n_features)
        y_pred (ndarray): Predicted values, shape (n_samples, n_features)
        y_cols (list): Feature names, length = n_features
        std_pred (ndarray, optional): Standard deviation of predictions
        mse (dict, optional): Dictionary with MSE per feature
        r2 (dict, optional): Dictionary with R² per feature
        title (str): Figure title
        save_name (str, optional): Base name for saving (without extension)
        save_dir (str or Path): Base directory to save into
    """
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have same shape"
    assert y_true.shape[1] == len(y_cols), "Mismatch between y_cols and y_true.shape[1]"
    if std_pred is not None:
        assert std_pred.shape == y_pred.shape, "std_pred must match y_pred shape"
    if mse or r2:
        assert all(k in mse and k in r2 for k in y_cols), "mse/r2 keys must match y_cols"

    n_outputs = len(y_cols)
    rows = int(np.ceil(n_outputs / 3))
    cols = min(3, n_outputs)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if n_outputs > 1 else [axes]

    for i, feat in enumerate(y_cols):
        ax = axes[i]
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]
        std_vals = std_pred[:,i] if std_pred is not None else None

        # Plot predicted vs. true
        ax.scatter(
            true_vals,
            pred_vals,
            alpha=0.6,
            color='dodgerblue',
            edgecolor='white',
            s=60,
            label='Prediction',
            zorder=3
        )

        # Optionally add uncertainty
        if std_pred is not None:
            ax.errorbar(
                true_vals,
                pred_vals,
                yerr=std_vals,
                fmt='none',
                ecolor='gray',
                alpha=0.4,
                capsize=2,
                zorder=2
            )
        

        # Perfect fit line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        margin = 0.1 * (max_val - min_val)
        ax.plot(
            [min_val - margin, max_val + margin],
            [min_val - margin, max_val + margin],
            'r--',
            label='Perfect Fit',
            linewidth=1,
            zorder=1
        )

        ax.set_title(latexify_param(feat), fontsize=12)
        ax.set_xlabel('True Value', fontsize=10)
        ax.set_ylabel('Predicted Value', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper left', fontsize=9)

        add_performance_metrics(ax, mse, r2, feat, min_val, max_val)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_name:
        data_dict = {
            'y_true': y_true,
            'y_pred': y_pred,
        }
        if std_pred is not None:
            data_dict['std_pred'] = std_pred

        save_figure_and_data(
            fig=fig,
            plot_name=save_name,
            save_dir=save_dir,
            formats=('pdf', 'svg'),
            data_dict=data_dict
        )
    
    # Violin Plot of Std Deviation
    if std_pred is not None:
        fig_violin, ax_v = plt.subplots(figsize=(10, 6))
        std_norm = (std_pred - std_pred.min(axis=0)) / (std_pred.max(axis=0) - std_pred.min(axis=0))
        sns.violinplot(data=std_norm, ax=ax_v, inner='box', cut=0)
        ax_v.set_xticks(np.arange(len(y_cols)))
        ax_v.set_xticklabels([latexify_param(c) for c in y_cols], rotation=0)
        # ax_v.set_yscale("log")
        ax_v.set_ylabel("Predictive Std Dev")
        ax_v.set_title("Distribution of Prediction Uncertainty (Violin Plot)")
        plt.tight_layout()

        save_figure_and_data(
            fig=fig_violin,
            plot_name=f"std_violin",
            save_dir=save_dir,
            formats=('pdf', 'svg')
        )



"""def plot_gp_predicted_relationships(gpr_pred, y_cols, y_scaler=None, N_curves=100, 
                                   pressure_range=(0.01, 8), figsize=(15, 4), save_name="TESTGPRELS", save_dir=None):
    
    #Plots physiological relationships between pressure and LV features using GP predictions.
    
    #Args:
    #    gpr_pred (np.ndarray): Array of shape (n_samples, 6) containing predicted parameters 
    #                         [α₀, β₀, α₁, β₁, α₂, β₂].
    #    y_cols (list): List of 6 output feature names matching gpr_pred columns.
    #    y_scaler (StandardScaler): Optional scaler to inverse-transform parameters.
    #    N_curves (int): Number of Monte Carlo curves to plot.
    #    pressure_range (tuple): Pressure range (min, max) in mmHg.
    #    figsize (tuple): Figure dimensions.
    

    # Input validation
    assert gpr_pred.shape[1] == 6, "gpr_pred must contain 6 columns (α₀,β₀,α₁,β₁,α₂,β₂)"
    assert len(y_cols) == 6, "y_cols must list 6 parameter names"
    
    # Inverse transform if y_scaler is passed
    if y_scaler is not None:
        params = y_scaler.inverse_transform(gpr_pred)
    else:
        params = gpr_pred.copy()
    
    # Parameter index mapping
    param_idx = {name: i for i, name in enumerate(y_cols)}
    required_params = ['alpha_0', 'beta_0', 'alpha_1', 'beta_1', 'alpha_2', 'beta_2']
    assert all(p in param_idx for p in required_params), "Missing required parameters in y_cols"
    
    # Linearly ramped pressure array
    p = np.linspace(*pressure_range, 100)  # mmHg
    
    # Figure initialization
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    # plt.suptitle(r"$\mathrm{Left\; ventricle\; pressure\; relationships\; with\; predicted\; parameters}$", y=1)
    
    # Plot configs
    plot_configs = [
        {
            'ax_idx': 0,
            'title': r"$\mathrm{\;}$",
            'xlabel': r"$\mathrm{Normalized\; volume}\ v_n$",
            'formula': r"$p = \alpha_0 v_n^{\beta_0}$",
            'alpha': params[:, param_idx['alpha_0']],
            'beta': params[:, param_idx['beta_0']],
            'color': 'royalblue'
        },
        {
            'ax_idx': 1,
            'title': r"$\mathrm{\;}$",
            'xlabel': r"$\mathrm{Strain}\ \bar{\varepsilon}_{max}$",
            'formula': r"$p = \alpha_1 (\bar{\varepsilon}_{max})^{\beta_1}$",
            'alpha': params[:, param_idx['alpha_1']],
            'beta': params[:, param_idx['beta_1']],
            'color': 'seagreen'
        },
        {
            'ax_idx': 2,
            'title': r"$\mathrm{\;}$",
            'xlabel': r"$\mathrm{Strain}\ |\bar{\varepsilon}_{min}|$",
            'formula': r"$p = \alpha_2 |\bar{\varepsilon}_{min}|^{\beta_2}$",
            'alpha': params[:, param_idx['alpha_2']],
            'beta': params[:, param_idx['beta_2']],
            'transform': np.abs,  # Absolute value for min strain
            'color': 'firebrick'
        }
    ]
    
    # Generate and plot relationships
    for config in plot_configs:
        ax = axes[config['ax_idx']]
        curves = []
        
        # MC sampling of parameter pairs
        for a, b in zip(config['alpha'][:N_curves], config['beta'][:N_curves]):
            with np.errstate(invalid='ignore'):
                x = (p / a) ** (1 / b)
            if 'transform' in config:
                x = config['transform'](x)
            curves.append(x)
        
        curves = np.array(curves)
        mean_curve = np.nanmean(curves, axis=0)
        std_curve = np.nanstd(curves, axis=0)
        
        # Plot mean 95% CI
        #ax.plot(mean_curve, p, color=config['color'], label=config['formula'])
        #ax.fill_betweenx(p, 
        #                mean_curve - 1.96*std_curve, 
        #                mean_curve + 1.96*std_curve,
        #                color=config['color'], alpha=0.2, label=r'$95\%\ \mathrm{CI}$')
        #ax.fill_betweenx(p, 
        #                mean_curve - 1.0*std_curve, 
        #               mean_curve + 1.0*std_curve,
        #                color=config['color'], alpha=0.5, label=r'$68\%\ \mathrm{CI}$')
                        
        
        # Formatting
        ax.set_title(config['title'], pad=0)
        ax.set_xlabel(config['xlabel'], labelpad=8)
        ax.set_ylabel(r"$\mathrm{Pressure\, (mmHg)}$", labelpad=8)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper right', framealpha=1)
        
        # Dynamic axis limits
        valid_x = mean_curve[~np.isnan(mean_curve)]
        if len(valid_x) > 0:
            ax.set_xlim(0.9*valid_x.min(), 1.1*valid_x.max())
        
        # Save plot data
        if save_name:
        # Extract curve data for saving
            p = np.linspace(*pressure_range, 100)
            curve_data = {}
            for config in plot_configs:
                curves = []
                for a, b in zip(config['alpha'][:N_curves], config['beta'][:N_curves]):
                    x = (p / a) ** (1 / b)
                    if 'transform' in config:
                        x = config['transform'](x)
                    curves.append(x)
                curve_data[f"{config['xlabel']}_curves"] = np.array(curves)
        
        save_figure_and_data(
            fig=plt.gcf(),
            plot_name=save_name,
            save_dir=save_dir,
            formats=('pdf', 'eps'),
            data_dict=None
        )
    
    plt.tight_layout()
    plt.show()"""


def revised_plot_gp_predicted_relationships(gpr_pred, y_cols, y_scaler=None, N_curves=100,
                                            pressure_range=(0.01, 8), figsize=(15, 4), seed=42):
    """
    Revised version: Plots physiological relationships between pressure and LV features
    using GP-predicted alpha/beta parameters. Fixes the logic by generating pressure
    as the independent variable (x), and plotting pressure on y.
    """
    assert gpr_pred.shape[1] == 6, "gpr_pred must contain 6 columns (α₀,β₀,α₁,β₁,α₂,β₂)"
    assert len(y_cols) == 6, "y_cols must list 6 parameter names"
    
    # Inverse transform if needed
    if y_scaler is not None:
        params = y_scaler.inverse_transform(gpr_pred)
    else:
        params = gpr_pred.copy()
    
    # Map column names to indices
    param_idx = {name: i for i, name in enumerate(y_cols)}
    
    np.random.seed(seed)
    selected_idx = np.random.choice(len(params), size=N_curves, replace=False)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    pressure = np.linspace(*pressure_range, 25)

    # Config for each subplot
    configs = [
        {
            "alpha": params[selected_idx, param_idx["alpha_0"]],
            "beta": params[selected_idx, param_idx["beta_0"]],
            "xlabel": "Normalized Volume $v_n$",
            "formula": r"$p = \alpha_0 \cdot e^{v_n \beta_0}$",
            "color": "royalblue",
            "ax": axes[0],
            "invert": False
        },
        {
            "alpha": params[selected_idx, param_idx["alpha_1"]],
            "beta": params[selected_idx, param_idx["beta_1"]],
            "xlabel": r"Max Strain $\bar{\varepsilon}_{max}$",
            "formula": r"$p = \alpha_1 \cdot e^{\bar{\varepsilon}_{max} \beta_1}$",
            "color": "seagreen",
            "ax": axes[1],
            "invert": False
        },
        {
            "alpha": params[selected_idx, param_idx["alpha_2"]],
            "beta": params[selected_idx, param_idx["beta_2"]],
            "xlabel": r"Abs Min Strain $|\bar{\varepsilon}_{min}|$",
            "formula": r"$p = \alpha_2 \cdot e^{|\bar{\varepsilon}_{min}| \beta_2}$",
            "color": "firebrick",
            "ax": axes[2],
            "invert": False
        }
    ]

    for cfg in configs:
        x_curves = []
        for a, b in zip(cfg["alpha"], cfg["beta"]):
            with np.errstate(over='ignore', invalid='ignore'):
                x = np.log(pressure / a) / b
                if cfg["invert"]:
                    x = np.abs(x)
                x_curves.append(x)
        
        x_curves = np.array(x_curves)
        mean_x = np.nanmean(x_curves, axis=0)
        std_x = np.nanstd(x_curves, axis=0)

        ax = cfg["ax"]
        ax.plot(mean_x, pressure, color=cfg["color"], label=cfg["formula"])
        ax.fill_betweenx(pressure, mean_x - std_x, mean_x + std_x,
                         color=cfg["color"], alpha=0.3, label="±1 SD")
        ax.set_xlabel(cfg["xlabel"])
        ax.set_ylabel("Pressure [mmHg]")
        ax.grid(True, linestyle=':')
        ax.legend()

    plt.tight_layout()
    return fig
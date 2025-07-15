# Cardiac Statistical Emulation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15858919.svg)](https://doi.org/10.5281/zenodo.15858919)

This repository implements a statistical emulator for cardiac finite element simulations using Gaussian Process Regression (GPR). The goal is to enable fast, surrogate modeling of the passive left ventricle mechanics for parameter inference and uncertainty quantification. The code supports my master's thesis: _"Statistical Emulation of Complex Cardiac Models using Gaussian Processes"_, which can be accessed [here](https://run.unl.pt/handle/10362/418?subject_page=1).

The code consists of three main sections: (1) the [forward](gpr_modelling/forward/) directory handles the emulation of the results from the simulator. Once we showed we could accurately replicate those results within a much shorter time frame, (2) we evaluated the level of uncertainty of those predictions through (global and local) sensitivity analysis, located in [sensitivity](gpr_modelling/sensitivity/); (3) finally, [parameter estimation](notebooks/03_parameter_inference.ipynb) was performed to evaluate whether the model could infer unknown material parameters based on unseen data. In a real context, this could mean feeding the model with (output) data from MRI or CT scans, for example, and assess the capability of the model to infer (input) parameters that caused the behaviors observed in the medical exams. 

The results from this three mains sections are segmented in [results](results/).

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/DiogoAmaro13/cardiac-statistical-emulation.git
cd cardiac-statistical-emulation

# 2. Set up the virtual environment
python -m venv venv-gpr_modelling
source venv-gpr_modelling/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Download pretrained models (see Main Execution below)

# 5. Run unit tests (optional)
python -m pytest tests/

# 6. Run main scripts
    # 6.1 Run forward predictions
    python -m scripts.run_forward_pipeline

    # 6.2 Run global sensitivity analysis
    python -m scripts.run_global_sa

    # 6.3 Run local sensitivity analysis
    python -m scripts.run_local_sa

    # 6.4 Run sensitivity analysis validation
    python -m scripts.run_sa_validation
```

---

## Project Structure

```
gpr_project_root/
├── gpr_modelling/         # Main GPR modeling package
├── data/                  # Input data and scalers (except large models)
├── FEniCS_simulations/    # Separate module to generate data using JAX + FEniCS
├── notebooks/             # Jupyter notebook for parameter inference
├── results/               # Outputs: plots, metrics, posterior traces, etc.
├── scripts/               # Script-based interface to run the pipeline
├── tests/                 # Unit tests for utility and model functions
```

---

## Environments

This project involves **two environments**:

### 1. Main Environment (pip + venv)

This is used for forward modeling, sensitivity analysis, and inference.

```bash
python -m venv venv-gpr_modelling
source venv-gpr_modelling/bin/activate
pip install -r requirements.txt
```

Main dependencies include:
- `numpy`, `scipy`, `matplotlib`, `pandas`
- `scikit-learn`, `gpflow`, `tensorflow`
- `openpyxl`, `seaborn`, `pytest`, `SALib`, `adjustText`

### 2. Secondary Environment (FEniCS Simulations)

Used **only** to regenerate simulation data using FEniCS + JAX (optional).

Instructions provided [here](FEniCS_simulations/jax-fem-env.yml):

```bash
conda create -n jax-fem-env python=3.9.18 numpy=1.24 scipy=1.11.3 matplotlib=3.8.0 pip=23.2.1
conda install -c conda-forge fenics
conda install -c conda-forge gmsh meshio
```

---
## Main Execution

Due to GitHub file size limits, trained `.pkl` models are hosted externally. Download them from [Zenodo](https://zenodo.org/records/15858919) and place them in:

```
data/objects/models/
```

If you'd rather train models from scratch, set the training flag in the [config file](gpr_modelling/forward/config.py) to

```python
TRAIN_MODE = True
```

---

## Available Scripts

Once the models are in place, the following commands are available:

```bash
# Run all unit tests
python -m pytest tests/

# Run GPR forward emulation
python -m scripts.run_forward_pipeline

# Run global sensitivity analysis (Sobol indices)
python -m scripts.run_global_sa

# Run local sensitivity analysis (gradients)
python -m scripts.run_local_sa

# Run validation of SA metrics (convergence tests, KDE analysis)
python -m scripts.run_sa_validation

# Run parameter inference
# 03_parameter_inference.ipynb
```

---

## Compatibility Notes

When running this project, you may encounter warnings related to the loading of .pkl files (e.g., ```Gaussian Process Regressor``` models or ```StandardScaler``` objects). These files were originally saved using scikit-learn version 1.6.1, and attempting to load them with a different version (1.7.0 or later) may trigger an ```InconsistentVersionWarning```. However, this warning does not affect execution in most cases. 

**Recommended**:
- If you encounter errors (not just warnings), set `TRAIN_MODE = True` and retrain the models and scalers locally using your current version.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
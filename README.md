
# Cardiac Statistical Emulation

This repository implements a statistical emulator for cardiac finite element simulations using Gaussian Process Regression (GPR). The goal is to enable fast, surrogate modeling of the passive left ventricle mechanics for parameter inference and uncertainty quantification. The code concerns my master thesis: "Statistical Emulation of Complex Cardiac Models using Gaussian Processes", which can be accessed [here](https://run.unl.pt/handle/10362/418?subject_page=1).

First, the [forward](gpr_modelling/forward/) directory handles the emulation of the results from the simulator. Once we showed we could accurately replicate those results within a much shorter time frame, we evaluated the level of uncertainty of those predictions through (global and local) sensitivity analysis, located in [sensitivity](gpr_modelling/sensitivity/). Finally, [parameter estimation](notebooks/03_parameter_inference.ipynb) was performed to evaluate whether the model could infer unknown material parameters based on unseen data. In a real context, this could mean feed the model with (output) data from MRI or CT scans, for example, and assess the capability of the model to infer (input) parameters that caused the behaviors observed in the medical exams. The results from this three mains sections are segmented in [results](results/).

## Project Structure
gpr_project_root/<br/>
├── gpr_modelling/ # Main GPR modeling package<br/>
├── data/ # Input data and scalers (except large models)<br/>
├── FEniCS_simulations/ # Separate module to generate data using JAX + FEniCS<br/>
├── notebooks/ # Jupyter notebook for inference section<br/>
├── results/ # Inference results, plots, sensitivity analysis, etc.<br/>
├── scripts/ # Script-based interface<br/>
├── tests/ # Unit tests

## Environment Setup

This project consists of **two environments**. The main one is where most of the code can be executed and its pip-based:

```bash
python -m venv venv-gpr_modelling
source venv-gpr_modelling/bin/activate
pip install -r requirements.txt
```

The second environment is fully isolated and only required if you intend to re-run the finite element simulations with FEniCS. It's conda based and has its own instructions [here](FEniCS_simulations/jax-fem-env.yml). To set it up:

```bash
conda env 
conda create -n jax-fem-env python=3.9.18 numpy=1.24 scipy=1.11.3 matplotlib=3.8.0 pip=23.2.1
conda install -c conda-forge fenics
conda install -c conda-forge gmsh meshio
conda activate jax-fem-env
```

# Main Execution

Due to file size, the trained .pkl models are not included in this repository. Before executing the code, download them from [here](https://zenodo.org/records/15858919) and place them into the [models directory](data/objects/models/).

If you want to train your own models, the previous step is not necessary and you should go into the [configuration file](gpr_modelling/forward/config.py) and change to ```python
TRAIN_MODE=True
```.
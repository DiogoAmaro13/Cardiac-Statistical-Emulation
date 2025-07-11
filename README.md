
# Cardiac Statistical Emulation

This repository implements a statistical emulator for cardiac finite element simulations using Gaussian Process Regression (GPR). The goal is to enable fast, surrogate modeling of the passive left ventricle mechanics for parameter inference and uncertainty quantification. The code concerns my master thesis: "Statistical Emulation of Complex Cardiac Models using Gaussian Processes", which can be accessed [here](https://run.unl.pt/handle/10362/418?subject_page=1).

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

This project consists of **two environments**:

### 1. Main GPR Modeling (pip-based)

This is the main modeling environment using a `venv` + `requirements.txt`.

#### Create and activate the environment:

```bash
python -m venv venv-gpr_modelling
source venv-gpr_modelling/bin/activate
pip install -r requirements.txt
```

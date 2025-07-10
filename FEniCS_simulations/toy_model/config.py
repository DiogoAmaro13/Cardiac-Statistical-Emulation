from pathlib import Path
import numpy as np

# Base paths
BASE_DIR = Path(__file__).resolve().parents[2]   # /home/amaro/gpr_modelling
DATA_DIR = BASE_DIR / "FEniCS_simulations" / "toy_model" / "data"
OBJECTS_DIR = DATA_DIR / "objects"

# Model directory
MODEL_DIR = OBJECTS_DIR / "gp_regressor"

# Scaler directory
SCALER_DIR = OBJECTS_DIR / "scalers"

# Results directory
RESULTS_DIR = BASE_DIR / "results"

# Predictions directory
TOY_MODEL_RESULTS_DIR = RESULTS_DIR / "toy_model"

# Default train models status
TRAIN_MODE = False

# Seed
RANDOM_SEED = 42

# Default feature/target columns
FEATURE_COLUMNS = ["a", "b", "af", "bf"]
TARGET_COLUMNS = ["edv"]
PRESSURE = np.array([[0., 0.06665, 0.1333, 0.19995, 0.2666, 0.33325, 0.3999, 
                      0.46655, 0.5332, 0.59985, 0.6665,  0.73315, 0.7998, 
                      0.86645, 0.9331, 0.99975, 1.0664, 1.13305, 1.1997, 
                      1.26635, 1.333]])
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parents[2]   # /home/amaro/root
DATA_DIR = BASE_DIR / "data"
OBJECTS_DIR = DATA_DIR / "objects"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# Model directory
MODEL_DIR = OBJECTS_DIR / "models"

# Scaler directory
SCALER_DIR = OBJECTS_DIR / "scalers"

# Results directory
RESULTS_DIR = BASE_DIR / "results"

# Predictions directory
FORWARD_RESULTS_DIR = RESULTS_DIR / "forward"

# Predictions metrics directory
METRICS_DIR =  FORWARD_RESULTS_DIR / "metrics"

# Sensitivity analysis directory
SA_RESULTS_DIR = RESULTS_DIR / "sensitivity_analysis"

# Inference directory
INVERSE_RESULTS_DIR = RESULTS_DIR / "inverse"

# Default train models status
TRAIN_MODE = False

# Seed
RANDOM_SEED = 42

# Default feature/target columns
FEATURE_COLUMNS = ["q1", "q2", "q3", "q4"]
TARGET_COLUMNS = ["alpha_0", "beta_0", "alpha_1", "beta_1", "alpha_2", "beta_2" ]

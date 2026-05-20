import os

BASE_PATH = os.getcwd()

# Data path
DATA_PATH = os.path.join(BASE_PATH, "data")
RAW_PATH = os.path.join(DATA_PATH, "raw")
PROCESSED_PATH = os.path.join(DATA_PATH, "processed")
PATCH_DIR = os.path.join(RAW_PATH, "official")

# Model path
MODEL_PATH = os.path.join(BASE_PATH, "model")

# Code path
SRC_PATH = os.path.join(BASE_PATH, "src")
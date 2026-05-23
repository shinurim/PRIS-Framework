import os

# project path
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data path
DATA_PATH = os.path.join(BASE_PATH, "data")
RAW_PATH = os.path.join(DATA_PATH, "raw")
PROCESSED_PATH = os.path.join(DATA_PATH, "processed")
PATCH_DIR = os.path.join(RAW_PATH, "official")

# Model path
MODEL_PATH = os.path.join(BASE_PATH, "model")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "model_save")  # 학습된 가중치(.pt) 저장 위치

# Code path
SRC_PATH = os.path.join(BASE_PATH, "src")

# directory
for _d in (PROCESSED_PATH, MODEL_SAVE_PATH):
    os.makedirs(_d, exist_ok=True)
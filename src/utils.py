import yaml, os
import random as random
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
   mean_squared_error,
   mean_absolute_error,
   mean_absolute_percentage_error
)
from scipy.stats import spearmanr

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_yaml(path: str):
  with open(path, "r", encoding = "utf-8") as f:
    return yaml.safe_load(f)

def save_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

def load_parquet(path: str):
    return pd.read_parquet(path)

def load_npy_files(path: str, fname: str):
    train = load_parquet(os.path.join(path, f"{fname}_train.parquet"))
    val   = load_parquet(os.path.join(path, f"{fname}_val.parquet"))
    test  = load_parquet(os.path.join(path, f"{fname}_test.parquet"))

    HC_COLS = ['hc_word_count', 'hc_readability', 'hc_subjectivity', 'hc_sentiment']
    META_COLS = ['author_num_reviews', 'author_num_games_owned',
                 'author_playtime_forever', 'timestamp_created',
                 'author_playtime_at_review', 'received_for_free']

    train_r    = np.array(train['review_bge'].tolist(), dtype=np.float32)
    train_p    = np.array(train['patch_bge'].tolist(),  dtype=np.float32)
    train_hc   = train[HC_COLS].values.astype(np.float32)
    train_meta = train[META_COLS].values.astype(np.float32)

    val_r    = np.array(val['review_bge'].tolist(), dtype=np.float32)
    val_p    = np.array(val['patch_bge'].tolist(),  dtype=np.float32)
    val_hc   = val[HC_COLS].values.astype(np.float32)
    val_meta = val[META_COLS].values.astype(np.float32)

    test_r    = np.array(test['review_bge'].tolist(), dtype=np.float32)
    test_p    = np.array(test['patch_bge'].tolist(),  dtype=np.float32)
    test_hc   = test[HC_COLS].values.astype(np.float32)
    test_meta = test[META_COLS].values.astype(np.float32)

    train_y = train['y_log'].values.astype(np.float32)
    val_y   = val['y_log'].values.astype(np.float32)
    test_y  = test['y_log'].values.astype(np.float32)

    return (train_r, train_p, train_hc, train_meta,
            val_r, val_p, val_hc, val_meta,
            test_r, test_p, test_hc, test_meta,
            train_y, val_y, test_y)

def get_metrics(preds, trues):
    preds = preds.flatten()
    trues = trues.flatten()

    mse  = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(trues, preds)
    mape = mean_absolute_percentage_error(trues, preds) * 100
    rho  = spearmanr(trues, preds).correlation if len(trues) > 1 else 0.0
    return mse, rmse, mae, mape, rho

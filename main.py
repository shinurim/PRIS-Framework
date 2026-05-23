import os

from src.utils import set_seed, load_yaml, load_npy_files, get_metrics
from src.path import SRC_PATH, PROCESSED_PATH, MODEL_SAVE_PATH
from src.data import DataProcessor
from model.pris import Proposed, train_model, predict

if __name__ == "__main__":
    config = load_yaml(os.path.join(SRC_PATH, "config.yaml"))
    data_config = config.get("data")
    bge_config = config.get("bge")
    args_config = config.get("args")
    set_seed(config.get("seed"))

    FNAME = data_config.get("fname")
    processed_data_list = set(os.listdir(PROCESSED_PATH))

    file_list = [f"{FNAME}_train.parquet", f"{FNAME}_val.parquet",
                 f"{FNAME}_test.parquet"]

    if set(file_list) - set(processed_data_list): # 파일 없으면
        data_processor = DataProcessor(**data_config, bge_config=bge_config)
        data_processor.run()

    (train_r, train_p, train_hc, train_meta,
     val_r, val_p, val_hc, val_meta,
     test_r, test_p, test_hc, test_meta,
     train_y, val_y, test_y) = load_npy_files(PROCESSED_PATH, FNAME)

    model = Proposed(**args_config).PRIS(
        emb_dim=args_config.get("emb_dim"),
        hc_dim=args_config.get("hc_dim"),
        meta_dim=args_config.get("meta_dim"),
    )

    # 학습 (best 가중치는 model_save/pris_best.pt 로 저장)
    print(f"Starting model training...")
    ckpt_path = os.path.join(MODEL_SAVE_PATH, "pris_best.pt")
    train_model(model,
                train_r, train_p, train_hc, train_meta, train_y,
                val_r, val_p, val_hc, val_meta, val_y,
                args_config, save_path=ckpt_path)

    # 평가
    print(f"Starting model Testing...")
    preds = predict(model, test_r, test_p, test_hc, test_meta,
                    batch_size=args_config.get("batch_size", 128))
    mse, rmse, mae, mape, rho = get_metrics(preds, test_y)
    print(f"[TEST] RMSE={rmse:.4f}  MSE={mse:.4f}  MAE={mae:.4f}  MAPE={mape:.4f}%  Spearman={rho:.4f}")

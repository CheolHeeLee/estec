import os
import sys

import numpy as np
from config import DEFAULT_TRAIN_CONFIG
from model import create_model
from preprocess import create_dataloaders, preprocess
from train_utils import atomic_joblib_dump, count_parameters, fit_model, get_device

CONFIG = DEFAULT_TRAIN_CONFIG.copy()


def build_union_data(config: dict) -> dict:
    weekday_data = preprocess(
        config["train_path"],
        seq_length=config["seq_length"],
        batch_size=config["batch_size"],
        validation_ratio=config["validation_ratio"],
    )

    X_train = np.concatenate([data["X_train"] for data in weekday_data.values()], axis=0)
    y_train = np.concatenate([data["y_train"] for data in weekday_data.values()], axis=0)
    loaders = create_dataloaders(
        X_train, y_train,
        np.concatenate([data["X_val"] for data in weekday_data.values()], axis=0),
        np.concatenate([data["y_val"] for data in weekday_data.values()], axis=0),
        config["batch_size"],
    )

    return {
        **loaders,
        "X_train": X_train,
        "y_train": y_train,
        "input_size": X_train.shape[2],
        "seq_length": config["seq_length"],
        "num_classes": 2,
        "task": "classification",
        "weekday_scalers": {day: data["scaler"] for day, data in weekday_data.items()},
        "scaler_fingerprints": {
            int(day): data["scaler_fingerprint"]
            for day, data in weekday_data.items()
        },
    }


def train_union_lstm_model(config: dict) -> dict:

    device = get_device()
    print(f"사용 디바이스: {device}")
    print("=" * 60)
    print("통합 LSTM 모델 학습 시작")
    print("=" * 60)

    data = build_union_data(config)
    os.makedirs(config["save_dir"], exist_ok=True)

    model = create_model(config).to(device)
    print(f"\nLSTM 파라미터 수: {count_parameters(model):,}")
    print(f"  은닉 크기: {config.get('hidden_size', 64)}")
    print(f"  LSTM 층: {config.get('num_layers', 2)}")

    save_path = os.path.join(config["save_dir"], "best_lstm_model_union.pth")
    best_acc = fit_model(
        model,
        data["train_loader"],
        data["val_loader"],
        config,
        device,
        save_path,
        checkpoint_metadata={
            "scaler_fingerprints": data["scaler_fingerprints"],
        },
    )

    day_names = {1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토", 7: "일"}
    for day_value, scaler in data["weekday_scalers"].items():
        day_name = day_names.get(day_value, str(day_value))
        scaler_save_path = os.path.join(
            config["save_dir"],
            f"scaler_{day_value}_{day_name}.joblib",
        )
        atomic_joblib_dump(scaler, scaler_save_path)
    print("통합 모델 학습 및 요일별 스케일러 저장 완료!")

    print("\n" + "=" * 60)
    print("통합 LSTM 모델 학습 완료!")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"저장 경로: {save_path}")
    print("=" * 60)

    return {
        "best_accuracy": best_acc,
        "best_validation_accuracy": best_acc,
        "save_path": save_path,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        import json
        with open(sys.argv[1]) as f:
            custom_config = json.load(f)
        CONFIG.update(custom_config)

    train_union_lstm_model(CONFIG)

import os
import sys

from config import DEFAULT_TRAIN_CONFIG
from model import create_model
from preprocess import preprocess
from train_utils import atomic_joblib_dump, count_parameters, fit_model, get_device

CONFIG = DEFAULT_TRAIN_CONFIG.copy()

def train_weekday_lstm_models(config: dict) -> dict:

    device = get_device()
    print(f"사용 디바이스: {device}")
    print("=" * 60)

    weekday_data = preprocess(
        config["train_path"],
        seq_length=config["seq_length"],
        batch_size=config["batch_size"],
        validation_ratio=config["validation_ratio"],
    )

    results = {}
    os.makedirs(config["save_dir"], exist_ok=True)

    for day, data in weekday_data.items():
        day_name = data["day_name"]
        print(f"\n{'=' * 60}")
        print(f"[{day_name}요일] LSTM 모델 학습 시작")
        print(f"{'=' * 60}")

        model = create_model(config).to(device)
        print(f"LSTM 파라미터 수: {count_parameters(model):,}")
        print(f"  은닉 크기: {config.get('hidden_size', 64)}")
        print(f"  LSTM 층: {config.get('num_layers', 2)}")

        save_path = os.path.join(
            config["save_dir"],
            f"best_lstm_model_{day}_{day_name}.pth",
        )

        best_acc = fit_model(
            model,
            data["train_loader"],
            data["val_loader"],
            config,
            device,
            save_path,
            checkpoint_metadata={
                "scaler_fingerprint": data["scaler_fingerprint"],
            },
        )

        scaler_save_path = os.path.join(
            config["save_dir"],
            f"scaler_{day}_{day_name}.joblib",
        )
        
        atomic_joblib_dump(data["scaler"], scaler_save_path)
        results[day] = {
            "day_name": day_name,
            "best_accuracy": best_acc,
            "best_validation_accuracy": best_acc,
            "save_path": save_path,
        }

    print("\n" + "=" * 60)
    print("LSTM 요일별 학습 완료")
    print("=" * 60)
    for result in results.values():
        print(
            f"  {result['day_name']}요일: "
            f"Best Validation Accuracy = {result['best_accuracy']:.4f}"
        )

    return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        import json
        with open(sys.argv[1]) as f:
            custom_config = json.load(f)
        CONFIG.update(custom_config)

    train_weekday_lstm_models(CONFIG)

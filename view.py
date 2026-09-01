from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from model import load_model_from_checkpoint
from preprocess import CLASS_NAMES, DAY_NAMES, TimeSeriesDataset, preprocess


matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIG = {
    "train_path": "data/train.csv",
    "test_path": "data/test.csv",
    "validation_ratio": 0.2,
    "seq_length": 60,
    "batch_size": 256,
    "checkpoint_dir": "checkpoints",
    "save_dir": "view",
}

WEEKDAYS = (1, 2, 3, 4, 5)
EXPECTED_WEEKDAY_SAMPLES = {
    1: 7140,
    2: 7200,
    3: 5760,
    4: 5760,
    5: 5760,
}
DISPLAY_LABELS = [f"{CLASS_NAMES[0]}(0)", f"{CLASS_NAMES[1]}(1)"]


def setup_font() -> None:
    try:
        fm.fontManager.addfont("./font/NanumGothic.ttf")
        plt.rc("font", family="NanumGothic")
    except Exception:
        print("한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 150


@torch.no_grad()
def predict_weekday(
    checkpoint_path: Path,
    data: dict,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model, config = load_model_from_checkpoint(checkpoint_path, device)

    if config["seq_length"] != data["seq_length"]:
        raise ValueError(
            f"seq_length 불일치: checkpoint={config['seq_length']}, "
            f"data={data['seq_length']}"
        )
    if config["input_size"] != data["X_test"].shape[2]:
        raise ValueError(
            f"input_size 불일치: checkpoint={config['input_size']}, "
            f"data={data['X_test'].shape[2]}"
        )

    loader = DataLoader(
        TimeSeriesDataset(data["X_test"], data["y_test"]),
        batch_size=batch_size,
        shuffle=False,
    )
    predictions = []
    for X_batch, _ in loader:
        logits = model(X_batch.to(device))
        predictions.append(torch.argmax(logits, dim=1).cpu().numpy())

    return data["y_test"], np.concatenate(predictions)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def add_metrics_legend(ax, metrics: dict[str, float]) -> None:
    handles = [
        Patch(
            facecolor="none",
            edgecolor="none",
            label=f"{name}: {value:.4f}",
        )
        for name, value in metrics.items()
    ]
    ax.legend(
        handles=handles,
        loc="center right",
        fontsize=7,
        frameon=True,
        framealpha=0.9,
        borderpad=0.45,
        handlelength=0,
        handletextpad=0,
        labelspacing=0.25,
    )


def draw_confusion_map(ax, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(matrix, display_labels=DISPLAY_LABELS)
    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )
    ax.set_title(f"{title}\nN={len(y_true):,}", fontsize=10)
    ax.set_xlabel("예측", fontsize=9)
    ax.set_ylabel("실제", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    add_metrics_legend(ax, calculate_metrics(y_true, y_pred))


def save_confusion_map(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    draw_confusion_map(ax, y_true, y_pred, title)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_weekday_subplots(records: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.5))
    for ax, weekday in zip(axes, WEEKDAYS):
        subset = records.loc[records["weekday"] == weekday]
        draw_confusion_map(
            ax,
            subset["y_true"].to_numpy(),
            subset["y_pred"].to_numpy(),
            f"{DAY_NAMES[weekday]}요일",
        )
    fig.suptitle("6월 요일별 혼동행렬", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def build_prediction_records(config: dict, device: torch.device) -> pd.DataFrame:
    weekday_data = preprocess(
        config["train_path"],
        config["test_path"],
        seq_length=config["seq_length"],
        batch_size=config["batch_size"],
        validation_ratio=config["validation_ratio"],
    )

    frames = []
    checkpoint_dir = Path(config["checkpoint_dir"])
    for weekday in WEEKDAYS:
        day_name = DAY_NAMES[weekday]
        checkpoint_path = checkpoint_dir / f"best_lstm_model_{weekday}_{day_name}.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"요일별 모델 파일이 없습니다: {checkpoint_path}")

        data = weekday_data[weekday]
        y_true, y_pred = predict_weekday(
            checkpoint_path,
            data,
            config["batch_size"],
            device,
        )
        datetimes = pd.to_datetime(data["test_datetime"])

        expected = EXPECTED_WEEKDAY_SAMPLES[weekday]
        if len(y_true) != expected:
            raise ValueError(
                f"{day_name}요일 샘플 수 불일치: expected={expected}, actual={len(y_true)}"
            )

        frames.append(
            pd.DataFrame(
                {
                    "datetime": datetimes,
                    "weekday": weekday,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
        )

    records = pd.concat(frames, ignore_index=True).sort_values("datetime")
    records["week"] = ((records["datetime"].dt.day - 1) // 7) + 1
    return records.reset_index(drop=True)


def main(config: dict) -> None:
    setup_font()
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    records = build_prediction_records(config, device)

    for week in range(1, 6):
        subset = records.loc[records["week"] == week]
        save_path = save_dir / f"week_{week}_confusion.png"
        save_confusion_map(
            subset["y_true"].to_numpy(),
            subset["y_pred"].to_numpy(),
            f"6월 {week}주차 월~금",
            save_path,
        )
        print(f"주차별 혼동행렬 저장: {save_path} ({len(subset):,}개)")

    for weekday in WEEKDAYS:
        subset = records.loc[records["weekday"] == weekday]
        day_name = DAY_NAMES[weekday]
        save_path = save_dir / f"weekday_{weekday}_{day_name}_confusion.png"
        save_confusion_map(
            subset["y_true"].to_numpy(),
            subset["y_pred"].to_numpy(),
            f"6월 {day_name}요일 전체",
            save_path,
        )
        print(f"요일별 혼동행렬 저장: {save_path} ({len(subset):,}개)")

    subplot_path = save_dir / "weekday_confusion_subplots_1x5.png"
    save_weekday_subplots(records, subplot_path)
    print(f"요일별 1x5 subplot 저장: {subplot_path}")

    june_path = save_dir / "june_weekdays_confusion.png"
    save_confusion_map(
        records["y_true"].to_numpy(),
        records["y_pred"].to_numpy(),
        "6월 전체 월~금",
        june_path,
    )
    print(f"6월 전체 혼동행렬 저장: {june_path} ({len(records):,}개)")


if __name__ == "__main__":
    main(CONFIG)

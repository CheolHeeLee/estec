import os
import tempfile
from pathlib import Path
from typing import Dict

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _temporary_artifact_path(target_path: str) -> str:
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    )
    handle.close()
    return handle.name


def atomic_torch_save(payload: dict, target_path: str) -> None:
    temp_path = _temporary_artifact_path(target_path)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, target_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def atomic_joblib_dump(value, target_path: str) -> None:
    temp_path = _temporary_artifact_path(target_path)
    try:
        joblib.dump(value, temp_path)
        os.replace(temp_path, target_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc="Train")
    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        correct = (preds == y_batch).sum().item()
        total_correct += correct
        total_samples += y_batch.size(0)
        total_loss += loss.item()

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct / y_batch.size(0):.4f}",
            }
        )

    return {
        "loss": total_loss / len(train_loader),
        "accuracy": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in data_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y_batch).sum().item()
        total_samples += y_batch.size(0)
        total_loss += loss.item()

    return {
        "loss": total_loss / len(data_loader),
        "accuracy": total_correct / total_samples,
    }

def create_scheduler(optimizer: optim.Optimizer, num_epochs: int, warmup_epochs: int):
    warmup_epochs = min(warmup_epochs, num_epochs)

    def lr_lambda(epoch):
        if warmup_epochs == 0:
            return 1
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1

    warmup = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine_epochs = num_epochs - warmup_epochs
    if cosine_epochs <= 0:
        return warmup

    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=1e-5,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )

def fit_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: dict,
    device: torch.device,
    save_path: str,
    checkpoint_metadata: dict = None,
) -> float:
    if config["num_epochs"] < 1:
        raise ValueError("num_epochs는 1 이상이어야 합니다.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = create_scheduler(
        optimizer,
        config["num_epochs"],
        config["warmup_epochs"],
    )

    best_acc = float("-inf")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(config["num_epochs"]):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip=config["grad_clip"],
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{config['num_epochs']}: "
            f"Train Loss={train_metrics['loss']:.4f}, "
            f"Train Acc={train_metrics['accuracy']:.4f}, "
            f"Val Loss={val_metrics['loss']:.4f}, "
            f"Val Acc={val_metrics['accuracy']:.4f}, "
            f"LR={current_lr:.6f}"
        )

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "validation_accuracy": best_acc,
                "config": config,
            }
            if checkpoint_metadata:
                reserved_keys = set(checkpoint) & set(checkpoint_metadata)
                if reserved_keys:
                    raise ValueError(
                        f"체크포인트 메타데이터가 필수 키를 덮어쓸 수 없습니다: {sorted(reserved_keys)}"
                    )
                checkpoint.update(checkpoint_metadata)
            atomic_torch_save(checkpoint, save_path)
            print(f"  -> Best 모델 저장 (Accuracy: {best_acc:.4f})")

    return best_acc

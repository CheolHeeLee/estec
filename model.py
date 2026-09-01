from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size : int,
        hidden_size : int,
        num_layers : int,
        dropout : float,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        return self.classifier(out)

def create_model(config: Dict):
    model_kwargs = {
        "input_size": config["input_size"],
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
        "dropout": config["dropout"],
    }
    return LSTMModel(**model_kwargs)

def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: torch.device,
) -> Tuple[LSTMModel, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = create_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.checkpoint_metadata = {
        "scaler_fingerprint": checkpoint.get("scaler_fingerprint"),
        "scaler_fingerprints": checkpoint.get("scaler_fingerprints"),
    }
    model.eval()
    return model, config

__all__ = ["LSTMModel", "create_model", "load_model_from_checkpoint"]

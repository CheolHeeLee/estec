
DEFAULT_TRAIN_CONFIG = {
    "train_path": "data/train.csv",
    "test_path": "data/test.csv",
    "validation_ratio": 0.2,

    "seq_length": 60,
    "batch_size": 64,

    "task": "classification",
    "input_size": 2,
    "hidden_size": 64,

    "num_layers": 2,
    "dropout": 0.1,

    "num_epochs": 30,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,

    "warmup_epochs": 5,

    "grad_clip": 1.0,
    "save_dir": "checkpoints",
}

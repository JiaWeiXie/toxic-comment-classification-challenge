from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
MODEL_LOG_PATH = CHECKPOINT_DIR / "lightning_logs"
MODEL_PATH = MODEL_LOG_PATH / "model_fit.ckpt"

BASE_MODEL_NAME = "roberta-base"

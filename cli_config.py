"""CLI path defaults and TrainConfig re-exports."""
from pathlib import Path

from core.train_config import TrainConfig, load_user_config, save_user_config, merge_config, load_effective_config  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_YAML = str(PROJECT_ROOT / "yaml" / "data.yaml")
MODEL_FILE = str(PROJECT_ROOT / "pretrained_models" / "yolov8n-seg.pt")
RESULTS_DIR = str(PROJECT_ROOT / "outputs" / "result")
LOG_DIR = str(PROJECT_ROOT / "outputs" / "logs")

PREDICT_DIR = str(PROJECT_ROOT / "predict")
BEST_SEG_MODEL = str(PROJECT_ROOT / "outputs" / "result" / "<your_experiment>" / "weights" / "best.pt")
TEST_IMAGES_DIR = str(PROJECT_ROOT / "data" / "<your_dataset>" / "images" / "test")

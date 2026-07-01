from dataclasses import dataclass
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_YAML = str(PROJECT_ROOT / "data.yaml")
MODEL_FILE = str(PROJECT_ROOT / "pretrained_models" / "yolov8n-seg.pt")
RESULTS_DIR = str(PROJECT_ROOT / "result")
LOG_DIR = str(PROJECT_ROOT / "train_logs")

PREDICT_DIR = str(PROJECT_ROOT / "predict")
BEST_SEG_MODEL = str(PROJECT_ROOT / "result" / "seg_dataset_all_pro_random__aug_e150_b16" / "weights" / "best.pt")
TEST_IMAGES_DIR = str(PROJECT_ROOT / "data" / "Source Data" / "datasets_all_pro" / "images" / "test")


@dataclass
class TrainConfig:

    # ===== 路径相关 =====
    data_yaml: str = DATA_YAML
    model_file: str = MODEL_FILE
    results_dir: str = RESULTS_DIR
    log_dir: str = LOG_DIR

    # ===== 超参数 =====
    epochs: int = 100
    imgsz: int = 640
    batch: int = 8
    device: int = 0
    lr0: float = 0.0005
    close_mosaic: int = 10
    multi_scale: float = 0.5

    experiment_name: str = "seg_dataset771_random__aug_e100"

    # ===== 数据增强相关 =====
    use_augment: bool = True
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0

    @property
    def save_dir(self) -> str:
        return os.path.join(self.results_dir, self.experiment_name)

    @property
    def last_pt(self) -> str:
        return os.path.join(self.save_dir, "weights", "last.pt")

    @property
    def best_pt(self) -> str:
        return os.path.join(self.save_dir, "weights", "best.pt")

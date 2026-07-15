#!/usr/bin/env python3
"""Re-validate all mix5 experiments with correct class filter."""
import sys
sys.path.insert(0, "/home/ljz/vibe_coding/YOLO/YOLO-LAB-CLI")

from config import DATA_YAML, LOG_DIR, RESULTS_DIR
from core.train_logger import append_full_val_log
from core.training import get_val_metrics, count_val_label_stats

_CLASSES_FILTER = [1, 2, 4, 6, 7, 8]

EXPERIMENTS = [
    "6cls_mix5_finetune_e30_lr0001_from_mix4",
    "6cls_mix5_finetune_e60_lr5e-5",
    "6cls_mix5_scratch_e150_lr0005",
]

class FakeConfig:
    def __init__(self, name):
        self.data_yaml = DATA_YAML
        self.log_dir = LOG_DIR
        self.results_dir = RESULTS_DIR
        self.experiment_name = name
        self.imgsz = 640
        self.batch = 8
        self.device = "0"

    @property
    def best_pt(self):
        import os
        return os.path.join(self.results_dir, self.experiment_name, "weights", "best.pt")

for exp_name in EXPERIMENTS:
    cfg = FakeConfig(exp_name)
    import os
    if not os.path.exists(cfg.best_pt):
        print(f"SKIP {exp_name}: no best.pt")
        continue
    print(f"Validating {exp_name} ...")
    try:
        metrics = get_val_metrics(cfg.best_pt, cfg, classes=_CLASSES_FILTER)
        class_image_counts, class_instance_counts = count_val_label_stats(cfg)
        append_full_val_log(
            config=cfg, mode="new_train", metrics=metrics,
            class_image_counts=class_image_counts,
            class_instance_counts=class_instance_counts,
            notes="修复class filter后重新验证",
            classes_filter=_CLASSES_FILTER,
        )
        print(f"  Done.")
    except Exception as e:
        print(f"  FAILED: {e}")

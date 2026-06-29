# Flatten Project Structure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all Python files from `scripts/` to project root, make `main.py` the real CLI entry point (no longer a passthrough to `train_segment.main`).

**Architecture:** Flat structure — all `.py` files at root. `main.py` handles i18n + arg parsing + mode menu. `train.py` is a pure library with training functions, receiving locale via `set_locale()` injection. `config.py` merges the old `paths.py` constants.

**Tech Stack:** Python 3, Ultralytics YOLO, PyYAML

## Global Constraints

- Do not change any training logic, validation logic, or logging behavior
- `data.yaml`, `locales/`, `dataset_tools/`, `pretrained_models/` remain untouched
- All resolved filesystem paths must stay identical to before

---

### Task 1: Create root `config.py` (merge paths.py)

**Files:**
- Create: `config.py`
- Delete: `scripts/paths.py`
- Keep: `scripts/config.py` (deleted later with scripts/)

**Interfaces:**
- Produces: `TrainConfig` dataclass, `DATA_YAML`, `MODEL_FILE`, `RESULTS_DIR`, `LOG_DIR`, `PREDICT_DIR`, `BEST_SEG_MODEL`, `TEST_IMAGES_DIR`

Take `scripts/config.py`, merge in the constants from `scripts/paths.py`, fix `PROJECT_ROOT` from `parent.parent` to `parent`.

- [ ] **Step 1: Write the merged config.py**

```python
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
```

- [ ] **Step 2: Verify PROJECT_ROOT resolves correctly**

```bash
cd /home/ljz/vibe_coding/YOLO/YOLO-LAB-CLI && python3 -c "
from config import PROJECT_ROOT, DATA_YAML, MODEL_FILE
print('PROJECT_ROOT:', PROJECT_ROOT)
print('DATA_YAML:', DATA_YAML)
import os
print('data.yaml exists:', os.path.exists(DATA_YAML))
"
```

Expected: `PROJECT_ROOT` is the project root dir, `data.yaml exists: True`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add root config.py merging paths.py constants"
```

---

### Task 2: Create root `train.py` (train_segment minus CLI)

**Files:**
- Create: `train.py`
- Source: `scripts/train_segment.py`

**Interfaces:**
- Consumes: `TrainConfig` from `config.py`, `append_train_log` + `append_full_val_log` from `train_logger.py`
- Produces: `set_locale(loc)`, `start_new_training(config)`, `resume_training(config)`, `train_from_previous_best(config)`, `find_latest_experiment_dir(results_dir, experiment_name)`

Copy `scripts/train_segment.py` content, then remove: i18n functions (`_detect_lang`, `_load_locale`, `_t`), `LOCALE_DIR`, `CONFIG` global, `parse_args()`, `override_config_from_args()`, `main()`, `if __name__ == "__main__"` block. Add `set_locale()`. Change `from config import TrainConfig` to `from config import TrainConfig`.

- [ ] **Step 1: Write train.py**

```python
import os
os.environ["MPLBACKEND"] = "Agg"

import re
import yaml
import shutil
from ultralytics import YOLO
from config import TrainConfig
from train_logger import append_train_log, append_full_val_log

# ── i18n (injected by main.py) ─────────────────────────────

_loc = {}

def set_locale(loc):
    """Called by main.py after loading locale JSON."""
    global _loc
    _loc = loc

def _t(loc, key, **kwargs):
    text = loc.get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass
    return text

# ── 工具函数 ──────────────────────────────────────────────

def ask_confirm_train(mode, pt_path, config):
    print(f"\n------------------------------")
    print(_t(_loc, "confirm.title", mode=mode))
    print(_t(_loc, "confirm.pt_file", path=pt_path))
    print(_t(_loc, "confirm.data_yaml", path=config.data_yaml))
    print(_t(_loc, "confirm.exp_name", name=config.experiment_name))
    print(_t(_loc, "confirm.epochs", epochs=config.epochs))
    print("------------------------------")

    confirm = input(_t(_loc, "confirm.prompt")).strip().lower()
    if confirm != "y":
        print(f"\n{_t(_loc, 'confirm.cancelled')}")
        return False
    return True


def list_experiments(results_dir):
    if not os.path.exists(results_dir):
        print(f"\n{_t(_loc, 'results.not_found', dir=results_dir)}")
        return []

    folders = []
    for name in os.listdir(results_dir):
        full_path = os.path.join(results_dir, name)
        if os.path.isdir(full_path):
            folders.append(name)
    folders.sort()
    return folders


def find_latest_experiment_dir(results_dir, experiment_name):
    if not os.path.exists(results_dir):
        return None

    pattern = re.compile(r'^' + re.escape(experiment_name) + r'(?:-(\d+))?$')

    best_dir = None
    best_suffix = -1

    for name in os.listdir(results_dir):
        full_path = os.path.join(results_dir, name)
        if not os.path.isdir(full_path):
            continue
        match = pattern.match(name)
        if match:
            suffix_str = match.group(1)
            suffix = int(suffix_str) if suffix_str else 0
            if suffix > best_suffix:
                best_suffix = suffix
                best_dir = name

    return best_dir


def ask_use_augment(config):
    status = _t(_loc, "augment.status_on") if config.use_augment else _t(_loc, "augment.status_off")
    print(f"\n------------------------------")
    print(_t(_loc, "augment.title"))
    print(_t(_loc, "augment.current", status=status))
    print("------------------------------")

    choice = input(_t(_loc, "augment.prompt")).strip().lower()
    if choice == "y":
        return True
    elif choice == "n":
        return False
    else:
        return config.use_augment


def build_train_kwargs(config, use_augment):
    kwargs = {
        "data": config.data_yaml,
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "device": config.device,
        "project": config.results_dir,
        "name": config.experiment_name,
        "plots": False,
    }
    if use_augment:
        kwargs.update({
            "hsv_h": config.hsv_h, "hsv_s": config.hsv_s, "hsv_v": config.hsv_v,
            "degrees": config.degrees, "translate": config.translate,
            "scale": config.scale, "shear": config.shear,
            "perspective": config.perspective, "flipud": config.flipud,
            "fliplr": config.fliplr, "mosaic": config.mosaic,
            "mixup": config.mixup, "copy_paste": config.copy_paste,
        })
    return kwargs


# ── 数据集与验证 ──────────────────────────────────────────

def get_class_names_from_data_yaml(data_yaml_path):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    else:
        return {}


def get_val_labels_dir(data_yaml_path):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root_path = data.get("path", "")
    val_path = data.get("val", "")
    if not val_path:
        return None
    if root_path and not os.path.isabs(val_path):
        val_path = os.path.join(root_path, val_path)
    val_path = os.path.normpath(val_path)
    parts = val_path.split(os.sep)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return os.path.normpath(os.sep.join(parts))
    parent_dir = os.path.dirname(os.path.dirname(val_path))
    val_name = os.path.basename(val_path)
    return os.path.join(parent_dir, "labels", val_name)


def count_val_label_stats(config):
    val_labels_dir = get_val_labels_dir(config.data_yaml)
    if not val_labels_dir or not os.path.exists(val_labels_dir):
        print(f"\n{_t(_loc, 'val.no_labels_dir', dir=val_labels_dir)}")
        return {}, {}

    class_names = get_class_names_from_data_yaml(config.data_yaml)
    class_image_counts = {name: 0 for name in class_names.values()}
    class_instance_counts = {name: 0 for name in class_names.values()}

    for file_name in os.listdir(val_labels_dir):
        if not file_name.endswith(".txt"):
            continue
        file_path = os.path.join(val_labels_dir, file_name)
        appeared_in_this_image = set()
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) < 1:
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                continue
            class_name = class_names.get(class_id, f"class_{class_id}")
            class_instance_counts[class_name] = class_instance_counts.get(class_name, 0) + 1
            appeared_in_this_image.add(class_name)
        for class_name in appeared_in_this_image:
            class_image_counts[class_name] = class_image_counts.get(class_name, 0) + 1

    return class_image_counts, class_instance_counts


def get_val_metrics(best_pt_path, config):
    model = YOLO(best_pt_path)
    val_name = f"{config.experiment_name}_tmp_val"
    val_dir = os.path.join(config.results_dir, val_name)
    try:
        metrics = model.val(
            data=config.data_yaml, imgsz=config.imgsz, batch=config.batch,
            device=config.device, plots=False, save_txt=False, save_json=False,
            visualize=False, project=config.results_dir, name=val_name,
        )
        return metrics
    finally:
        shutil.rmtree(val_dir, ignore_errors=True)


def log_validation_result(config, mode, notes=""):
    if not os.path.exists(config.best_pt):
        print(f"\n{_t(_loc, 'val.no_best_pt', path=config.best_pt)}")
        return
    try:
        metrics = get_val_metrics(config.best_pt, config)
        class_image_counts, class_instance_counts = count_val_label_stats(config)
        append_full_val_log(
            config=config, mode=mode, metrics=metrics,
            class_image_counts=class_image_counts,
            class_instance_counts=class_instance_counts, notes=notes,
        )
        print(f"\n{_t(_loc, 'val.logged')}")
    except Exception as e:
        print(f"\n{_t(_loc, 'val.failed', err=e)}")


# ── 训练流程 ──────────────────────────────────────────────

def start_new_training(config):
    mode_label = _t(_loc, "train.new_mode_label")
    if not ask_confirm_train(mode_label, config.model_file, config):
        return
    use_augment = ask_use_augment(config)
    aug_label = _t(_loc, "augment.status_on") if use_augment else _t(_loc, "augment.status_off")
    append_train_log(config, mode="new_train", status="started",
                     notes=_t(_loc, "log.new_started", aug=aug_label))

    try:
        model = YOLO(config.model_file)
        train_kwargs = build_train_kwargs(config, use_augment)
        model.train(**train_kwargs)
        append_train_log(config, mode="new_train", status="finished",
                         notes=_t(_loc, "log.new_finished", aug=aug_label))
        log_validation_result(config, mode="new_train", notes=_t(_loc, "log.new_val"))
    except Exception as e:
        if os.path.exists(config.best_pt):
            append_train_log(config, mode="new_train", status="finished",
                             notes=_t(_loc, "log.val_retry", err=e))
            print(f"\n{_t(_loc, 'train.completed_but_val_failed', err=e)}")
            log_validation_result(config, mode="new_train",
                                  notes=_t(_loc, "log.val_retry", err=e))
        else:
            append_train_log(config, mode="new_train", status="failed",
                             notes=_t(_loc, "log.failed", err=e))
            print(f"\n{_t(_loc, 'train.failed', err=e)}")


def resume_training(config):
    latest_dir = find_latest_experiment_dir(config.results_dir, config.experiment_name)
    if latest_dir is not None and latest_dir != config.experiment_name:
        config.experiment_name = latest_dir

    if not os.path.exists(config.last_pt):
        print(f"\n{_t(_loc, 'resume.not_found', path=config.last_pt)}")
        choice = input(_t(_loc, "resume.fallback_prompt")).strip().lower()
        if choice == "y":
            start_new_training(config)
        else:
            print(_t(_loc, "resume.cancelled"))
        return

    mode_label = _t(_loc, "train.resume_mode_label")
    if not ask_confirm_train(mode_label, config.last_pt, config):
        return

    append_train_log(config, mode="resume_train", status="started",
                     notes=_t(_loc, "log.resume_started"))
    try:
        model = YOLO(config.last_pt)
        model.train(resume=True)
        append_train_log(config, mode="resume_train", status="finished",
                         notes=_t(_loc, "log.resume_finished"))
        log_validation_result(config, mode="resume_train", notes=_t(_loc, "log.resume_val"))
    except Exception as e:
        append_train_log(config, mode="resume_train", status="failed",
                         notes=_t(_loc, "log.failed", err=e))
        print(f"\n{_t(_loc, 'resume.failed', err=e)}")


def train_from_previous_best(config):
    folders = list_experiments(config.results_dir)
    if not folders:
        print(f"\n{_t(_loc, 'history.empty')}")
        return

    print(f"\n{_t(_loc, 'history.list_title')}")
    for i, folder in enumerate(folders, 1):
        print(f"{i} - {folder}")

    choice = input(_t(_loc, "history.select_prompt")).strip()
    if not choice.isdigit():
        print(_t(_loc, "history.invalid_input"))
        return

    idx = int(choice) - 1
    if idx < 0 or idx >= len(folders):
        print(_t(_loc, "history.out_of_range"))
        return

    selected_exp = folders[idx]
    selected_best_pt = os.path.join(config.results_dir, selected_exp, "weights", "best.pt")
    if not os.path.exists(selected_best_pt):
        print(f"\n{_t(_loc, 'history.no_best_pt', path=selected_best_pt)}")
        return

    print(f"\n{_t(_loc, 'history.selected', name=selected_exp)}")

    mode_label = _t(_loc, "train.finetune_mode_label")
    if not ask_confirm_train(mode_label, selected_best_pt, config):
        return

    use_augment = ask_use_augment(config)
    aug_label = _t(_loc, "augment.status_on") if use_augment else _t(_loc, "augment.status_off")

    append_train_log(config, mode="train_from_best", status="started",
                     notes=_t(_loc, "log.finetune_started", exp=selected_exp, aug=aug_label))
    try:
        model = YOLO(selected_best_pt)
        train_kwargs = build_train_kwargs(config, use_augment)
        model.train(**train_kwargs)
        append_train_log(config, mode="train_from_best", status="finished",
                         notes=_t(_loc, "log.finetune_finished", exp=selected_exp, aug=aug_label))
        log_validation_result(config, mode="train_from_best",
                              notes=_t(_loc, "log.finetune_val", exp=selected_exp))
    except Exception as e:
        if os.path.exists(config.best_pt):
            append_train_log(config, mode="train_from_best", status="finished",
                             notes=_t(_loc, "log.val_retry", err=e))
            print(f"\n{_t(_loc, 'train.completed_but_val_failed', err=e)}")
            log_validation_result(config, mode="train_from_best",
                                  notes=_t(_loc, "log.val_retry", err=e))
        else:
            append_train_log(config, mode="train_from_best", status="failed",
                             notes=_t(_loc, "log.failed", err=e))
            print(f"\n{_t(_loc, 'train.failed', err=e)}")
```

- [ ] **Step 2: Verify train.py imports resolve**

```bash
cd /home/ljz/vibe_coding/YOLO/YOLO-LAB-CLI && python3 -c "
from train import set_locale, start_new_training, resume_training, train_from_previous_best
print('train.py imports OK')
"
```

Expected: `train.py imports OK`

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add root train.py (pure training library, locale injected)"
```

---

### Task 3: Rewrite root `main.py` as real CLI entry

**Files:**
- Modify: `main.py`

**Interfaces:**
- Consumes: `TrainConfig` from `config.py`, `set_locale`, `start_new_training`, `resume_training`, `train_from_previous_best` from `train.py`
- Produces: CLI entry point

- [ ] **Step 1: Write main.py**

```python
#!/usr/bin/env python3
"""YOLO-LAB-CLI entry point — i18n, arg parsing, mode menu."""

import sys
import os
import json
import locale
import argparse
from pathlib import Path

from config import TrainConfig
from train import set_locale, start_new_training, resume_training, train_from_previous_best

# ── i18n ──────────────────────────────────────────────────

LOCALE_DIR = Path(__file__).resolve().parent / "locales"


def _detect_lang():
    try:
        system_lang, _ = locale.getdefaultlocale()
        if system_lang:
            code = system_lang[:2].lower()
            if code in ("zh", "en", "fr", "es"):
                return code
    except Exception:
        pass
    return "en"


def _load_locale(lang):
    path = LOCALE_DIR / f"{lang}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _t(loc, key, **kwargs):
    text = loc.get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass
    return text


# ── CLI ──────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO training script")
    parser.add_argument("--epochs", type=int, default=None, help="training epochs")
    parser.add_argument("--imgsz", type=int, default=None, help="input image size")
    parser.add_argument("--batch", type=int, default=None, help="batch size")
    parser.add_argument("--device", type=str, default=None, help="device: 0 / 0,1 / cpu")
    parser.add_argument("--name", type=str, default=None, help="experiment name")
    parser.add_argument("--lang", type=str, default=None, help="language: zh/en/fr/es (auto-detect if not set)")
    return parser.parse_args()


def override_config_from_args(config, args):
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.imgsz is not None:
        config.imgsz = args.imgsz
    if args.batch is not None:
        config.batch = args.batch
    if args.device is not None:
        config.device = args.device
    if args.name is not None:
        config.experiment_name = args.name
    return config


def main():
    args = parse_args()
    lang = args.lang or _detect_lang()
    _loc = _load_locale(lang)

    # Inject locale into train module so all training functions can use it
    set_locale(_loc)

    config = TrainConfig()
    config = override_config_from_args(config, args)

    print(_t(_loc, "mode.select"))
    print(_t(_loc, "mode.1"))
    print(_t(_loc, "mode.2"))
    print(_t(_loc, "mode.3"))
    choice = input(_t(_loc, "mode.prompt") + "\n").strip()

    if choice == "1":
        start_new_training(config)
    elif choice == "2":
        resume_training(config)
    elif choice == "3":
        train_from_previous_best(config)
    else:
        print(_t(_loc, "mode.invalid"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test — verify main.py starts without import errors**

```bash
cd /home/ljz/vibe_coding/YOLO/YOLO-LAB-CLI && python3 -c "
from main import main, parse_args, _detect_lang
print('main.py imports OK')
print('detected lang:', _detect_lang())
"
```

Expected: `main.py imports OK` + detected language

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: rewrite main.py as real CLI entry point"
```

---

### Task 4: Move `train_logger.py` and `predict.py` to root, fix imports

**Files:**
- Move: `scripts/train_logger.py` → `train_logger.py` (no code changes)
- Move: `scripts/predict_test.py` → `predict.py` (fix import)
- Delete: `scripts/paths.py` (constants now in `config.py`)

- [ ] **Step 1: Copy train_logger.py (no changes)**

```bash
cp scripts/train_logger.py train_logger.py
```

- [ ] **Step 2: Write predict.py (fix import)**

```python
from dataclasses import dataclass
from pathlib import Path
import json

from ultralytics import YOLO
from config import PREDICT_DIR, BEST_SEG_MODEL, TEST_IMAGES_DIR


# =========================
# 只改这里：通用参数
# =========================
@dataclass
class InferConfig:
    model_path: str = BEST_SEG_MODEL
    source: str = TEST_IMAGES_DIR
    save_dir: str = str(Path(PREDICT_DIR) / "predict_result")

    conf: float = 0.406
    imgsz: int = 640

    # 外置任务参数文件
    task_param_file: str = "infer_task_params.json"

    # 输出文件后缀
    out_suffix: str = "_overlay.jpg"


class TaskParamLoader:
    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self.params = self._load_json()

    def _load_json(self) -> dict:
        if not self.json_path.exists():
            raise FileNotFoundError(f"找不到任务参数文件: {self.json_path}")
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_task_params(self, task: str) -> dict:
        if task not in self.params:
            raise KeyError(f"配置文件里没有 task={task} 的参数")
        return self.params[task]


class YOLOInferencer:
    def __init__(self, cfg: InferConfig):
        self.cfg = cfg
        self.model = YOLO(self.cfg.model_path)
        self.task_loader = TaskParamLoader(self.cfg.task_param_file)

        self.save_dir = Path(self.cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.task = self._detect_task()
        self.task_params = self.task_loader.get_task_params(self.task)

    def _detect_task(self) -> str:
        task = getattr(self.model, "task", None)
        if not task:
            raise ValueError("无法从模型中识别 task")
        return task

    def _build_predict_kwargs(self) -> dict:
        kwargs = {
            "source": self.cfg.source,
            "imgsz": self.cfg.imgsz,
            "conf": self.cfg.conf,
            "save": False
        }
        task_predict_kwargs = self.task_params.get("predict", {})
        kwargs.update(task_predict_kwargs)
        return kwargs

    def _build_plot_kwargs(self) -> dict:
        return self.task_params.get("plot", {})

    def run(self):
        print(f"模型: {self.cfg.model_path}")
        print(f"自动识别任务: {self.task}")
        print(f"输入源: {self.cfg.source}")
        print(f"输出目录: {self.save_dir}")

        predict_kwargs = self._build_predict_kwargs()
        plot_kwargs = self._build_plot_kwargs()

        results = self.model.predict(**predict_kwargs)

        for i, r in enumerate(results):
            if getattr(r, "path", None):
                stem = Path(r.path).stem
            else:
                stem = f"result_{i:05d}"

            out_path = self.save_dir / f"{stem}{self.cfg.out_suffix}"
            r.save(filename=str(out_path), **plot_kwargs)

        print(f"推理完成，共保存 {len(results)} 张结果到: {self.save_dir}")


if __name__ == "__main__":
    cfg = InferConfig(
        model_path=BEST_SEG_MODEL,
        source=TEST_IMAGES_DIR,
        save_dir=str(Path(PREDICT_DIR) / "seg_dataset_all_pro_random__aug_e150_b16_mask_overlay2"),
        conf=0.406,
        imgsz=640,
        task_param_file="infer_task_params.json",
        out_suffix="_overlay.jpg"
    )

    inferencer = YOLOInferencer(cfg)
    inferencer.run()
```

Note: Only changed line 6 — `from paths import ...` → `from config import ...`

- [ ] **Step 3: Verify predict.py import**

```bash
cd /home/ljz/vibe_coding/YOLO/YOLO-LAB-CLI && python3 -c "from predict import InferConfig; print('predict.py imports OK')"
```

- [ ] **Step 4: Delete scripts/paths.py**

```bash
rm scripts/paths.py
```

- [ ] **Step 5: Commit**

```bash
git add train_logger.py predict.py
git rm scripts/paths.py
git commit -m "feat: move train_logger.py and predict.py to root, fix predict import"
```

---

### Task 5: Delete `scripts/` directory and old root files

**Files:**
- Delete: `scripts/` (entire directory)
- Delete: old root `config.py` if it was auto-created

- [ ] **Step 1: Remove scripts/ directory**

```bash
cd /home/ljz/vibe_coding/YOLO/YOLO-LAB-CLI && rm -rf scripts/
```

- [ ] **Step 2: Verify final structure**

```bash
cd /home/ljz/vibe_coding/YOLO/YOLO-LAB-CLI && ls -la *.py
```

Expected output:
```
config.py  main.py  predict.py  train.py  train_logger.py
```

- [ ] **Step 3: Full import verification**

```bash
cd /home/ljz/vibe_coding/YOLO/YOLO-LAB-CLI && python3 -c "
from config import TrainConfig, PROJECT_ROOT, DATA_YAML
from train import set_locale, start_new_training, resume_training, train_from_previous_best
from train_logger import append_train_log, append_full_val_log
from predict import InferConfig, YOLOInferencer
from main import main, parse_args, _detect_lang
print('All imports OK')
print('PROJECT_ROOT:', PROJECT_ROOT)
"
```

Expected: `All imports OK` + PROJECT_ROOT path

- [ ] **Step 4: Quick dry-run of main (import + parse --help)**

```bash
cd /home/ljz/vibe_coding/YOLO/YOLO-LAB-CLI && python3 main.py --help
```

Expected: argparse help output with `--epochs`, `--imgsz`, `--batch`, `--device`, `--name`, `--lang`

- [ ] **Step 5: Commit**

```bash
git rm -r scripts/
git add -A
git commit -m "feat: remove scripts/ directory, project fully flattened"
```

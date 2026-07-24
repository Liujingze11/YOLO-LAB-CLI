import os
import yaml
os.environ["MPLBACKEND"] = "Agg"

from ultralytics import YOLO
from config import TrainConfig
from core.train_logger import append_train_log, append_full_val_log
from core.training import (
    build_train_kwargs,
    list_experiments,
    find_latest_experiment_dir,
    get_class_names_from_data_yaml,
    get_val_labels_dir,
    count_val_label_stats,
    get_val_metrics,
)
from core.lr_schedulers import build_lr_callback
from core.i18n import t as _t

# ── i18n (injected by main.py) ─────────────────────────────

_loc = {}

def set_locale(loc):
    """Called by main.py after loading locale JSON."""
    global _loc
    _loc = loc

# === 类别过滤：不修改原始标注txt文件，训练时自动过滤+重映射 ===
_CLASSES_FILTER = [1, 2, 3]  # mix7_cls3: button, switch, DDBC

# ── 确认流程（分步：YAML → 超参数 → 增强 → mixup）──────────

def _parse_data_yaml(data_yaml_path: str) -> dict:
    """Parse data YAML and return key info."""
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {
        "path": data.get("path", ""),
        "train": data.get("train", ""),
        "val": data.get("val", ""),
        "test": data.get("test", ""),
        "names": data.get("names", {}),
    }


def _confirm(prompt: str, show_n: bool = False) -> bool | None:
    """确认步骤。Enter/y → True，q → None（退出），n → False（取消）。
    其他任意键忽略，防止误触。
    """
    while True:
        choice = input(prompt).strip().lower()
        if choice == "q":
            print(f"\n{_t(_loc, 'confirm.quit')}")
            return None
        if choice in ("", "y"):
            return True
        if choice == "n":
            return False
        # 其他按键忽略，重新提示


def _format_class_names(names) -> str:
    """Format class names dict/list into display string."""
    if isinstance(names, dict):
        items = [(int(k), v) for k, v in names.items() if int(k) != 0]
    elif isinstance(names, list):
        items = [(i, name) for i, name in enumerate(names) if i != 0]
    else:
        return ""
    return ", ".join(f"{k}={v}" for k, v in sorted(items))


def confirm_data_yaml(config: TrainConfig) -> bool:
    """Step: Show parsed data YAML info, Enter=yes, q=quit."""
    info = _parse_data_yaml(config.data_yaml)
    classes_str = _format_class_names(info["names"])

    print(f"\n{'='*55}")
    print(_t(_loc, "yaml.title"))
    print(f"{'='*55}")
    print(f"  {_t(_loc, 'yaml.file')}: {config.data_yaml}")
    print(f"  {_t(_loc, 'yaml.dataset_path')}: {info['path']}")
    print(f"  {_t(_loc, 'yaml.train')}: {info['train']}")
    print(f"  {_t(_loc, 'yaml.val')}: {info['val']}")
    if info["test"]:
        print(f"  {_t(_loc, 'yaml.test')}: {info['test']}")
    print(f"  {_t(_loc, 'yaml.classes')}: {classes_str}")
    print(f"{'='*55}")

    return _confirm(_t(_loc, "yaml.prompt")) is True


def confirm_hyperparams(config: TrainConfig, pt_path: str, is_resume: bool = False) -> bool:
    """Step: Show hyperparameters, Enter=yes, q=quit."""
    freeze_val = getattr(config, "freeze", 0)
    cos_lr_val = getattr(config, "cos_lr", True)
    warmup_val = getattr(config, "warmup_epochs", 1.0)

    lr_label = f"{config.lr0}"
    if cos_lr_val:
        lr_label += " (cosine)"
    if is_resume:
        lr_label += " [" + _t(_loc, "hyper.lr_resume_note") + "]"

    print(f"\n{'='*55}")
    print(_t(_loc, "hyper.title"))
    print(f"{'='*55}")
    print(f"  {_t(_loc, 'hyper.model')}: {os.path.basename(pt_path)}")
    print(f"  {_t(_loc, 'hyper.exp_name')}: {config.experiment_name}")
    print(f"  {_t(_loc, 'hyper.epochs')}: {config.epochs}")
    print(f"  {_t(_loc, 'hyper.lr0')}: {lr_label}")
    print(f"  {_t(_loc, 'hyper.batch')}: {config.batch}")
    print(f"  {_t(_loc, 'hyper.imgsz')}: {config.imgsz}")
    print(f"  {_t(_loc, 'hyper.freeze')}: {freeze_val}")
    print(f"  {_t(_loc, 'hyper.mosaic')}: {config.mosaic}")
    print(f"  {_t(_loc, 'hyper.warmup')}: {warmup_val}")
    print(f"{'='*55}")

    return _confirm(_t(_loc, "hyper.prompt")) is True


def ask_confirm_train(mode, pt_path, config):
    """Step 1: 基本确认 — Enter=yes, q=quit."""
    print(f"\n------------------------------")
    print(_t(_loc, "confirm.title", mode=mode))
    print(_t(_loc, "confirm.pt_file", path=pt_path))
    print(_t(_loc, "confirm.data_yaml", path=config.data_yaml))
    print(_t(_loc, "confirm.exp_name", name=config.experiment_name))
    print(_t(_loc, "confirm.epochs", epochs=config.epochs))
    print("------------------------------")

    return _confirm(_t(_loc, "confirm.prompt"), show_n=True) is True


def confirm_augment_params(config: TrainConfig) -> bool:
    """Step: 增强参数 — Enter=yes, q=quit."""
    print(f"\n{'='*55}")
    print(_t(_loc, "aug_params.title"))
    print(f"{'='*55}")
    print(f"  {_t(_loc, 'aug_params.hsv_h')}: {config.hsv_h}")
    print(f"  {_t(_loc, 'aug_params.hsv_s')}: {config.hsv_s}")
    print(f"  {_t(_loc, 'aug_params.hsv_v')}: {config.hsv_v}")
    print(f"  {_t(_loc, 'aug_params.degrees')}: {config.degrees}")
    print(f"  {_t(_loc, 'aug_params.translate')}: {config.translate}")
    print(f"  {_t(_loc, 'aug_params.scale')}: {config.scale}")
    print(f"  {_t(_loc, 'aug_params.shear')}: {config.shear}")
    print(f"  {_t(_loc, 'aug_params.perspective')}: {config.perspective}")
    print(f"  {_t(_loc, 'aug_params.flipud')}: {config.flipud}")
    print(f"  {_t(_loc, 'aug_params.fliplr')}: {config.fliplr}")
    print(f"  {_t(_loc, 'aug_params.mosaic')}: {config.mosaic}")
    print(f"  {_t(_loc, 'aug_params.copy_paste')}: {config.copy_paste}")
    print(f"{'='*55}")

    return _confirm(_t(_loc, "aug_params.prompt")) is True


def ask_lr_scheduler(config):
    """Step: Select LR scheduler — 1/2/3, Enter=default, q=quit."""
    current = config.lr_scheduler
    labels = {
        "adaptive": _t(_loc, "lr_scheduler.adaptive"),
        "restart": _t(_loc, "lr_scheduler.restart"),
        "cosine": _t(_loc, "lr_scheduler.cosine"),
    }
    current_label = labels.get(current, current)

    print(f"\n------------------------------")
    print(_t(_loc, "lr_scheduler.title"))
    print(_t(_loc, "lr_scheduler.current", current=current_label))
    print(f"  1 - {_t(_loc, 'lr_scheduler.adaptive')}")
    print(f"  2 - {_t(_loc, 'lr_scheduler.restart')}")
    print(f"  3 - {_t(_loc, 'lr_scheduler.cosine')}")
    print("------------------------------")

    while True:
        choice = input(_t(_loc, "lr_scheduler.prompt")).strip().lower()
        if choice == "q":
            print(f"\n{_t(_loc, 'confirm.quit')}")
            return False
        if choice == "1":
            config.lr_scheduler = "adaptive"; break
        elif choice == "2":
            config.lr_scheduler = "restart"; break
        elif choice == "3":
            config.lr_scheduler = "cosine"; break
        elif choice == "":
            break  # Enter = use default
        # 其他按键忽略

    selected = labels.get(config.lr_scheduler, config.lr_scheduler)
    print(_t(_loc, "lr_scheduler.selected", selected=selected))
    return True


def _run_confirmation_flow(config: TrainConfig, pt_path: str, mode_label: str, is_resume: bool = False) -> tuple:
    """Full confirmation flow (保持用户习惯，第一步不变):
    Step 1: ask_confirm_train (existing)
    Step 2: confirm_data_yaml (NEW)
    Step 3: confirm_hyperparams (NEW)
    Step 3.5: ask_lr_scheduler (NEW - LR策略选择)
    Step 4: ask_use_augment (existing)
    Step 4.5: confirm_augment_params (NEW, only if augment enabled)
    Step 5: ask_mixup (existing)
    Returns (confirmed: bool, use_augment: bool, mixup_value: float).
    """
    # Step 1: basic confirm (keeping user habit)
    if not ask_confirm_train(mode_label, pt_path, config):
        return False, False, 0.0

    # Step 2: data YAML details (NEW)
    if not confirm_data_yaml(config):
        return False, False, 0.0

    # Step 3: hyperparameters (NEW)
    if not confirm_hyperparams(config, pt_path, is_resume):
        return False, False, 0.0

    # Step 3.5: LR scheduler (NEW)
    if not is_resume:
        if not ask_lr_scheduler(config):
            return False, False, 0.0

    # Step 4: augment
    use_augment = ask_use_augment(config)
    if use_augment is None:
        return False, False, 0.0

    # Step 4.5: if augment enabled, show augment params for confirmation (NEW)
    if use_augment:
        if not confirm_augment_params(config):
            return False, False, 0.0

    # Step 5: mixup
    mixup_value = ask_mixup(config)
    if mixup_value is None:
        return False, False, 0.0

    return True, use_augment, mixup_value


def ask_use_augment(config):
    status = _t(_loc, "augment.status_on") if config.use_augment else _t(_loc, "augment.status_off")
    print(f"\n------------------------------")
    print(_t(_loc, "augment.title"))
    print(_t(_loc, "augment.current", status=status))
    print("------------------------------")

    while True:
        choice = input(_t(_loc, "augment.prompt")).strip().lower()
        if choice == "q":
            print(f"\n{_t(_loc, 'confirm.quit')}")
            return None
        if choice == "y":
            return True
        if choice == "n":
            return False
        if choice == "":
            return config.use_augment  # Enter = default
        # 其他按键忽略


def ask_mixup(config):
    status = _t(_loc, "mixup.status_on", value=config.mixup) if config.mixup > 0 else _t(_loc, "mixup.status_off")
    print(f"\n------------------------------")
    print(_t(_loc, "mixup.title"))
    print(_t(_loc, "mixup.current", status=status))
    print("------------------------------")

    while True:
        choice = input(_t(_loc, "mixup.prompt")).strip()
        if choice == "":
            return config.mixup  # Enter = default
        low = choice.lower()
        if low == "q":
            print(f"\n{_t(_loc, 'confirm.quit')}")
            return None
        if low == "n":
            return 0.0
        if low == "y":
            return config.mixup if config.mixup > 0 else 0.2
        try:
            val = float(choice)
            return max(0.0, min(1.0, val))
        except ValueError:
            pass  # 无效输入，重新提示


# ── 验证 (i18n wrapper) ──────────────────────────────────

def log_validation_result(config, mode, notes=""):
    if not os.path.exists(config.best_pt):
        print(f"\n{_t(_loc, 'val.no_best_pt', path=config.best_pt)}")
        return
    try:
        metrics = get_val_metrics(config.best_pt, config, classes=_CLASSES_FILTER)
        class_image_counts, class_instance_counts = count_val_label_stats(config)
        append_full_val_log(
            config=config, mode=mode, metrics=metrics,
            class_image_counts=class_image_counts,
            class_instance_counts=class_instance_counts, notes=notes,
            classes_filter=_CLASSES_FILTER,
        )
        print(f"\n{_t(_loc, 'val.logged')}")
    except Exception as e:
        print(f"\n{_t(_loc, 'val.failed', err=e)}")


# ── 训练流程 ──────────────────────────────────────────────

def start_new_training(config):
    mode_label = _t(_loc, "train.new_mode_label")

    confirmed, use_augment, mixup_value = _run_confirmation_flow(
        config, config.model_file, mode_label, is_resume=False,
    )
    if not confirmed:
        return

    original_mixup = config.mixup
    config.mixup = mixup_value
    aug_label = _t(_loc, "augment.status_on") if use_augment else _t(_loc, "augment.status_off")
    mixup_label = _t(_loc, "mixup.status_on", value=mixup_value) if mixup_value > 0 else _t(_loc, "mixup.status_off")

    # 构建 LR scheduler callback
    config._lr_callback = build_lr_callback(config.lr_scheduler, lr0=config.lr0)

    print(f"\n>>> {_t(_loc, 'all_settings')}: epochs={config.epochs}, imgsz={config.imgsz}, batch={config.batch}, lr={config.lr0}, lr_scheduler={config.lr_scheduler}, mixup={mixup_value}, aug={aug_label}")

    notes = _t(_loc, "log.new_started", aug=aug_label) + f", mixup={mixup_label}, lr_scheduler={config.lr_scheduler}"
    append_train_log(config, mode="new_train", status="started", notes=notes)

    try:
        model = YOLO(config.model_file)
        if config._lr_callback is not None:
            model.add_callback("on_fit_epoch_end", config._lr_callback.on_fit_epoch_end)
        train_kwargs = build_train_kwargs(config, use_augment, _CLASSES_FILTER)
        model.train(**train_kwargs)
        config.mixup = original_mixup
        append_train_log(config, mode="new_train", status="finished",
                         notes=_t(_loc, "log.new_finished", aug=aug_label))
        log_validation_result(config, mode="new_train", notes=_t(_loc, "log.new_val"))
    except Exception as e:
        config.mixup = original_mixup
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
        if _confirm(_t(_loc, "resume.fallback_prompt")) is True:
            start_new_training(config)
        return

    mode_label = _t(_loc, "train.resume_mode_label")
    confirmed, _, _ = _run_confirmation_flow(
        config, config.last_pt, mode_label, is_resume=True,
    )
    if not confirmed:
        return

    append_train_log(config, mode="resume_train", status="started",
                     notes=_t(_loc, "log.resume_started"))
    try:
        model = YOLO(config.last_pt)
        model.train(resume=True, project=config.results_dir, name=config.experiment_name, exist_ok=True)
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

    confirmed, use_augment, mixup_value = _run_confirmation_flow(
        config, selected_best_pt, mode_label, is_resume=False,
    )
    if not confirmed:
        return

    aug_label = _t(_loc, "augment.status_on") if use_augment else _t(_loc, "augment.status_off")
    mixup_label = _t(_loc, "mixup.status_on", value=mixup_value) if mixup_value > 0 else _t(_loc, "mixup.status_off")
    original_mixup = config.mixup
    config.mixup = mixup_value

    # 构建 LR scheduler callback
    config._lr_callback = build_lr_callback(config.lr_scheduler, lr0=config.lr0)

    print(f"\n>>> {_t(_loc, 'all_settings')}: epochs={config.epochs}, imgsz={config.imgsz}, batch={config.batch}, lr={config.lr0}, lr_scheduler={config.lr_scheduler}, mixup={mixup_value}, aug={aug_label}")

    notes = _t(_loc, "log.finetune_started", exp=selected_exp, aug=aug_label) + f", mixup={mixup_label}, lr_scheduler={config.lr_scheduler}"
    append_train_log(config, mode="train_from_best", status="started", notes=notes)
    try:
        model = YOLO(selected_best_pt)
        if config._lr_callback is not None:
            model.add_callback("on_fit_epoch_end", config._lr_callback.on_fit_epoch_end)
        train_kwargs = build_train_kwargs(config, use_augment, _CLASSES_FILTER)
        model.train(**train_kwargs)
        config.mixup = original_mixup
        append_train_log(config, mode="train_from_best", status="finished",
                         notes=_t(_loc, "log.finetune_finished", exp=selected_exp, aug=aug_label))
        log_validation_result(config, mode="train_from_best",
                              notes=_t(_loc, "log.finetune_val", exp=selected_exp))
    except Exception as e:
        config.mixup = original_mixup
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

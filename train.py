import os
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
from core.i18n import t as _t

# ── i18n (injected by main.py) ─────────────────────────────

_loc = {}

def set_locale(loc):
    """Called by main.py after loading locale JSON."""
    global _loc
    _loc = loc

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


# ── 验证 (i18n wrapper) ──────────────────────────────────

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

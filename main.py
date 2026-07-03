#!/usr/bin/env python3
"""YOLO-LAB-CLI entry point — i18n, arg parsing, mode menu."""

import locale
import argparse
from pathlib import Path

from config import TrainConfig, DATA_YAML, MODEL_FILE, RESULTS_DIR, LOG_DIR
from train import set_locale, start_new_training, resume_training, train_from_previous_best
from core.i18n import t as _t, load_locale

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
    _loc = load_locale(LOCALE_DIR, lang)

    # Inject locale into train module so all training functions can use it
    set_locale(_loc)

    config = TrainConfig(
        data_yaml=DATA_YAML,
        model_file=MODEL_FILE,
        results_dir=RESULTS_DIR,
        log_dir=LOG_DIR,
    )
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

import locale
import argparse
from pathlib import Path

from cli_config import TrainConfig, DATA_YAML, MODEL_FILE, RESULTS_DIR, LOG_DIR
from training_flows import set_locale, start_new_training, resume_training, train_from_previous_best
from core.i18n import t as _t, load_locale

# ── i18n ──────────────────────────────────────────────────

LOCALE_DIR = Path(__file__).resolve().parent / "locales"


def _detect_lang():
    # 读取系统语言前两位,支持 zh/en/fr/es,其余情况兜底返回 en
    try:
        system_lang, _ = locale.getdefaultlocale()
        if system_lang:
            code = system_lang[:2].lower()
            if code in ("zh", "en", "fr", "es"):
                return code
    except Exception:
        pass
    return "en"


# ── CLI args ─────────────────────────────────────────────

def parse_args():
    # 声明可接收的命令行开关,用户未传入的一律为 None
    parser = argparse.ArgumentParser(description="YOLO training script")
    parser.add_argument("--epochs", type=int, default=None, help="training epochs")
    parser.add_argument("--imgsz", type=int, default=None, help="input image size")
    parser.add_argument("--batch", type=int, default=None, help="batch size")
    parser.add_argument("--device", type=str, default=None, help="device: 0 / 0,1 / cpu")
    parser.add_argument("--name", type=str, default=None, help="experiment name")
    return parser.parse_args()


def apply_cli_overrides(train_config, args):
    # 只覆盖用户显式传入的开关,未传入的保持默认值
    if args.epochs is not None:
        train_config.epochs = args.epochs
    if args.imgsz is not None:
        train_config.imgsz = args.imgsz
    if args.batch is not None:
        train_config.batch = args.batch
    if args.device is not None:
        train_config.device = args.device
    if args.name is not None:
        train_config.experiment_name = args.name
    return train_config


# ── main ─────────────────────────────────────────────────

def main():
    args = parse_args()
    lang = _detect_lang()
    _loc = load_locale(LOCALE_DIR, lang)

    # 把翻译表注入训练模块,后续所有训练函数都能使用
    set_locale(_loc)

    train_config = TrainConfig(
        data_yaml=DATA_YAML,
        model_file=MODEL_FILE,
        results_dir=RESULTS_DIR,
        log_dir=LOG_DIR,
    )
    train_config = apply_cli_overrides(train_config, args)

    print(_t(_loc, "mode.select"))
    print(_t(_loc, "mode.1"))
    print(_t(_loc, "mode.2"))
    print(_t(_loc, "mode.3"))
    choice = input(_t(_loc, "mode.prompt") + "\n").strip()

    if choice == "1":
        start_new_training(train_config)
    elif choice == "2":
        resume_training(train_config)
    elif choice == "3":
        train_from_previous_best(train_config)
    else:
        print(_t(_loc, "mode.invalid"))


if __name__ == "__main__":
    main()

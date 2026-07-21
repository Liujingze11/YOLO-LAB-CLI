# YOLO Lab CLI

[中文](docs/README_zh.md) | [Français](docs/README_fr.md) | [Español](docs/README_es.md)

Command-line YOLO segmentation training tool built on Ultralytics.

## Features

- Three training modes: New / Resume / Fine-tune
- Toggleable data augmentation
- Automatic validation with CSV logging (overall + per-class metrics)
- Experiment isolation: each run creates independent result directories and logs
- CLI parameter overrides (`--epochs`, `--imgsz`, `--batch`, `--device`, `--name`)
- Auto-detects system language (zh/en/fr/es), with `--lang` override

## Quick Start

```bash
git clone https://github.com/Liujingze11/YOLO-LAB-CLI.git
cd YOLO-LAB-CLI
pip install -r requirements.txt
python main.py
```

## Requirements

- Python 3.8+
- ultralytics, PyYAML

```bash
pip install ultralytics pyyaml
```

## Project Structure

```
YOLO-LAB-CLI/
├── main.py                 # CLI entry point (i18n, arg parsing, mode menu)
├── work_flows.py                # Training flows (new / resume / fine-tune)
├── config.py               # Path defaults + core re-exports
├── yaml/                   # Dataset configuration files
│   └── data.yaml.example   # Example dataset config
├── core/                   # Shared library (identical across CLI/GUI/LAB)
│   ├── config.py           # TrainConfig dataclass
│   ├── training.py         # Training utilities
│   ├── train_logger.py     # CSV logging
│   ├── device.py           # GPU detection
│   ├── i18n.py             # i18n helpers
│   └── paths.py            # Model registry
├── tools/                  # Utility scripts
│   ├── predict_tools/      # Inference (predict.py + task params)
│   └── dataset_tools/      # Dataset splitting & label utilities
├── outputs/                # Training outputs (git-ignored)
│   ├── result/             # Model weights & plots
│   └── logs/               # CSV training logs
├── locales/                # i18n translation files
└── pretrained_models/      # Pretrained model weights
```

## Training Modes

Run `python main.py` and choose:

- **1** — New training from initial weights
- **2** — Resume from last.pt
- **3** — Fine-tune from historical best.pt

## CLI Options

```bash
python main.py --epochs 200 --imgsz 1280 --batch 8 --device 0 --name my_experiment
```

Language is auto-detected from the system locale. Override with `--lang`:

```bash
python main.py --lang en   # English
python main.py --lang fr   # Français
python main.py --lang es   # Español
python main.py --lang zh   # 中文
```

## Dataset Config Format

Place your dataset YAML files in the `yaml/` directory. See `yaml/data.yaml.example`:

```yaml
path: ./data/datasets
train: images/train
val: images/val
test: images/test
names:
  0: background
  1: class_a
  2: class_b
```

## Outputs

- Weights & plots: `outputs/result/<experiment_name>/`
- CSV logs: `outputs/logs/`

## License

MIT

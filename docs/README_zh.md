# YOLO Lab CLI

[English](README.md) | [Français](README_fr.md) | [Español](README_es.md)

YOLO 分割模型命令行训练工具，基于 Ultralytics。

## 功能

- 三种训练模式：新训练 / 续训 / 微调
- 数据增强开关控制
- 自动验证并记录 CSV 日志（整体 + 每类指标）
- 实验隔离：每次训练生成独立的结果目录和日志
- 命令行参数覆盖配置（`--epochs`, `--imgsz`, `--batch`, `--device`, `--name`）
- 自动检测系统语言（zh/en/fr/es），支持 `--lang` 手动指定

## 快速开始

```bash
git clone https://github.com/Liujingze11/YOLO-LAB-CLI.git
cd YOLO-LAB-CLI
pip install -r requirements.txt
python main.py
```

## 依赖

- Python 3.8+
- ultralytics, PyYAML

```bash
pip install ultralytics pyyaml
```

## 项目结构

```
YOLO-LAB-CLI/
├── main.py                 # 主入口
├── train.py                # 训练模块
├── predict.py              # 推理模块
├── config.py               # 配置类
├── train_logger.py         # CSV 日志
├── infer_task_params.json  # 任务推理参数
├── data.yaml               # 数据集配置
├── requirements.txt        # Python 依赖
├── dataset_tools/          # 数据集分割 & 标签工具
│   ├── create_empty_labels.py
│   ├── split_train_val/
│   ├── split_train_val_test/
│   └── split_images_only/
├── locales/                # i18n 翻译文件
└── pretrained_models/      # 预训练模型
```

## 训练模式

运行 `python main.py` 后选择：

- **1** — 新训练，从初始权重开始
- **2** — 续训，从上次 `last.pt` 继续
- **3** — 微调，基于历史实验 `best.pt`

## 命令行参数

```bash
python main.py --epochs 200 --imgsz 1280 --batch 8 --device 0 --name my_experiment
```

语言默认根据系统自动检测，也可通过 `--lang` 指定：

```bash
python main.py --lang zh   # 中文
python main.py --lang en   # English
python main.py --lang fr   # Français
python main.py --lang es   # Español
```

## data.yaml 格式

```yaml
path: ./data/datasets
train: images/train
val: images/val
names:
  0: class_a
  1: class_b
```

## 输出

- 实验结果：`result/<experiment_name>/weights/` (best.pt, last.pt)
- CSV 日志：`train_logs/`

## License

MIT

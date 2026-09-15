# YOLO Lab CLI

[English](../README.md) | [Français](README_fr.md) | [Español](README_es.md)

YOLO 分割模型命令行训练工具，基于 Ultralytics。

## 功能

- 三种训练模式：新训练 / 续训 / 微调
- 分步确认流程：数据集 YAML → 超参数 → 数据增强 → Mixup → 保存所有 epoch 权重
- 数据增强开关控制（mosaic、mixup、copy-paste、随机擦除、翻转、HSV 等）
- 冻结 backbone、余弦学习率、可插拔学习率调度器（adaptive / restart / cosine）
- 可选"保存所有 epoch 权重"，自动整理到 `weights/epochs/`
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
- ultralytics、PyYAML、numpy

```bash
pip install ultralytics pyyaml numpy
```

## 项目结构

```
YOLO-LAB-CLI/
├── main.py                 # CLI 入口（i18n、参数解析、模式菜单）
├── training_flows.py       # 训练流程（新训练 / 续训 / 微调）
├── cli_config.py           # CLI 路径默认值 + TrainConfig 再导出
├── yaml/                   # 数据集配置文件
│   └── data.yaml.example   # 数据集配置示例
├── core/                   # 共享库（CLI/GUI/LAB 三仓库一致）
│   ├── train_config.py     # TrainConfig 数据类 + 用户配置持久化
│   ├── training.py         # 训练工具函数
│   ├── train_logger.py     # CSV 日志记录
│   ├── lr_schedulers.py    # 学习率调度器回调
│   ├── device.py           # GPU 检测
│   ├── i18n.py             # i18n 辅助函数
│   └── paths.py            # 模型注册表
├── scripts/                # 推理与复评辅助脚本
├── tools/                  # 工具脚本
│   ├── predict_tools/      # 推理（predict.py + 任务参数）
│   └── dataset_tools/      # 数据集划分与标签工具
├── docs/                   # 多语言 README（zh / fr / es）
├── outputs/                # 训练输出（git 忽略）
│   ├── result/             # 模型权重与图表
│   └── logs/               # CSV 训练日志
├── locales/                # i18n 翻译文件
└── pretrained_models/      # 预训练模型权重
```

## 训练模式

运行 `python main.py` 后选择：

- **1** — 使用初始权重新训练
- **2** — 从 last.pt 续训
- **3** — 从历史 best.pt 微调

每种模式都会依次经过确认流程——数据集 YAML → 超参数 → 数据增强 → Mixup → 保存所有 epoch 权重——训练开始前可逐项确认和调整。

## 命令行参数

```bash
python main.py --epochs 200 --imgsz 1280 --batch 8 --device 0 --name my_experiment
```

语言默认根据系统区域自动检测，也可用 `--lang` 手动指定：

```bash
python main.py --lang en   # English
python main.py --lang fr   # Français
python main.py --lang es   # Español
python main.py --lang zh   # 中文
```

## 数据集配置格式

将数据集 YAML 文件放在 `yaml/` 目录中，参考 `yaml/data.yaml.example`：

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

## 输出

- 权重与图表：`outputs/result/<experiment_name>/`
- 各 epoch 权重（可选）：`outputs/result/<experiment_name>/weights/epochs/`
- CSV 日志：`outputs/logs/`

## 许可证

MIT

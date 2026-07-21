# YOLO Lab CLI

[English](../README.md) | [Français](README_fr.md) | [Español](README_es.md)

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
├── main.py                 # CLI 入口（国际化、参数解析、模式菜单）
├── work_flows.py                # 训练流程（新训练 / 续训 / 微调）
├── config.py               # 路径默认值 + core 重导出
├── yaml/                   # 数据集配置文件
│   └── data.yaml.example   # 示例数据集配置
├── core/                   # 共享库（CLI/GUI/LAB 完全一致）
│   ├── config.py           # TrainConfig 数据类
│   ├── training.py         # 训练工具函数
│   ├── train_logger.py     # CSV 日志
│   ├── device.py           # GPU 检测
│   ├── i18n.py             # 国际化辅助
│   └── paths.py            # 模型注册表
├── tools/                  # 工具脚本
│   ├── predict_tools/      # 推理（predict.py + 任务参数）
│   └── dataset_tools/      # 数据集切分 & 标签工具
├── outputs/                # 训练输出（git-ignored）
│   ├── result/             # 模型权重 & 曲线图
│   └── logs/               # CSV 训练日志
├── locales/                # i18n 翻译文件
└── pretrained_models/      # 预训练模型权重
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

## 数据集配置格式

将数据集 YAML 文件放入 `yaml/` 目录。参考 `yaml/data.yaml.example`：

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

- 权重与曲线图：`outputs/result/<experiment_name>/`
- CSV 日志：`outputs/logs/`

## License

MIT

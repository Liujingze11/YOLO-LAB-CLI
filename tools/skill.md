---
name: yolo-tools
description: YOLO 数据集与推理工具集。当用户需要准备 YOLO 数据集——为缺失标注的图片补齐空标签、把数据集划分成 train/val/test——或对图片/文件夹做模型推理出图时使用。工具位于 tools/ 目录,均为命令行脚本,完整参数用 --help 查看。
---

# YOLO 工具集

所有工具都支持相对路径和绝对路径;相对路径按运行命令时所在的目录解释。

## yolo_create_empty_labels.py — 补齐缺失标签

**功能**:为训练数据中每张图片创建同名空标签 txt,已存在的标签**绝不覆盖**。

**什么时候用**:照片有的标注了、有的没标注,需要把标签补齐时。

**用法**:

```bash
python tools/yolo_create_empty_labels.py                                # 默认: images/ → labels/
python tools/yolo_create_empty_labels.py --images <训练图片文件夹> --labels <标签文件夹>
python tools/yolo_create_empty_labels.py --images /path/to/images --labels /path/to/labels  # 绝对路径也行
```

**注意**:只识别 .jpg / .jpeg / .png / .bmp / .webp;在划分数据集之前执行。

## yolo_split_dataset.py — 划分数据集

**功能**:把 YOLO 训练数据划分成 train/val(/test)(随机按比例,或每隔 N 张取 1 张),输出官方格式目录;图片与同名标签同步移动。

**什么时候用**:标签补齐后,把 data/images、data/labels 里的图片分成 train/val(/test)。

**用法**:

```bash
python tools/yolo_split_dataset.py --images data/images --labels data/labels --ratio 0.2
python tools/yolo_split_dataset.py --images /path/to/data/images --labels /path/to/data/labels --ratio 0.2
python tools/yolo_split_dataset.py --images data/images --split train_val_test --test-ratio 0.1
python tools/yolo_split_dataset.py --images data/images --mode every --every 5
```

运行前 data/images、data/labels 里直接放着图片与标签;运行后两个目录下各生成 train/ 和 val/(可选 test/),抽中的进 val/test,其余全部进 train,源目录不再有图片文件。

**主要参数**:

- `--labels` 也可不传:`data/images` 自动对应 `data/labels`;图片文件夹不叫 images 时,必须用 `--labels` 指定
- `--no-labels` 只分图片;`--dash` 支持带 `-` 或 `_` 的文件名;`--seed` 默认 42,结果可复现

**注意**:默认只处理纯数字文件名(如 1681.jpg),其余跳过并警告;运行后源目录不再有图片文件,重复运行会直接报错——想重新划分,先把文件挪回源目录再跑。

## yolo_task_predict.py — 单源推理,输出叠加图

**功能**:加载模型后自动识别任务类型(detect / segment / classify / pose / obb),按 `yolo_task_params.json` 中该任务的参数推理,保存 `_overlay.jpg` 叠加图。

**什么时候用**:对单个图片或单个数据源推理,输出 overlay 叠加图。

**配置**:脚本底部 `InferConfig`(注释"只改这里")——模型路径、输入源、输出目录等都在那里改。

**注意**:无命令行参数,直接运行脚本。

## yolo_batch_predict.py — 文件夹批量推理,断点续跑

**功能**:对文件夹逐张推理并画标注,保持原目录结构输出;损坏的图片复制原图;已处理的跳过。

**什么时候用**:整个文件夹批量推理出图。

**用法**:

```bash
python tools/yolo_batch_predict.py --model <best.pt> --source <输入文件夹> --output <输出文件夹>
```

**可调参数**:`--conf 0.7`(置信度)、`--font-size 9`(文字字号)、`--line-width 1`(框线宽)。

**注意**:路径必须传参(默认值是占位符);每处理 500 张打印一次进度。

## yolo_task_params.json — 任务参数(数据文件)

`yolo_task_predict.py` 读取的配置文件:五种任务各自 predict / plot 参数(字体大小、线宽、是否画 mask 等),按任务名自动匹配。

## 典型工作流(串起来用)

1. **补齐标签** — `yolo_create_empty_labels.py`:照片有的标注了、有的没标注,先用它把标签补齐,让每张图都有标签(已存在的绝不覆盖)
2. **划分数据集** — `yolo_split_dataset.py`:标签补齐后,再划分 train/val(/test)
3. **训练** — `python main.py`(主程序,不在 tools/)
4. **推理出图** — `yolo_task_predict.py`(单源,输出叠加图)或 `yolo_batch_predict.py`(文件夹批量)

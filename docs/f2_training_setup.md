# F2 auto 对照训练配置（2026-10-11）

当前默认从官方 `pretrained_models/yolov8n-seg.pt` 加载权重，在第二版 F2 上
新建auto对照实验训练120轮。运行菜单选择 **1（新训练）**，会新建优化器和学习率调度。
旧的F1起点微调实验及其权重保留；当前入口使用官方预训练权重。

## 数据

- 原始数据：`/home/ljz/Train_logging/YOLO/3cls_data/3 - YOLO训练数据/第二版/F2`。
- 已准备划分：`/home/ljz/Train_logging/YOLO/3cls_data/3 - YOLO训练数据/第二版/F2_split`。
- 数据YAML：`yaml/data_f2_split.yaml`，沿用当前随机8:2划分、种子42。
- 总计3129张：训练2503张，验证626张，两份名单没有交叉。
- 划分目录通过符号链接读取原图和标签，原始文件保留。
- 类别过滤为 `[1, 2, 3]`：button、switch、DDBC。保留0号占位类别，Ultralytics只过滤、不重映射，因此YAML类别数为4。
- 训练56张、验证19张空标签图片作为背景样本保留。

## 参数

| 参数 | 设置 |
| --- | --- |
| epochs / imgsz / batch | 120 / 640 / 16 |
| optimizer / lr0 | auto / 0.001（配置占位，auto会覆盖） |
| cos_lr / lrf | true / 0.01 |
| warmup_epochs / warmup_bias_lr | 3 / 0 |
| freeze / multi_scale | 0 / 0 |
| close_mosaic | 10 |
| HSV h / s / v | 0.015 / 0.7 / 0.5 |
| translate / scale | 0.1 / 0.3 |
| degrees / shear / perspective | 0 / 0 / 0 |
| flipud / fliplr | 0 / 0 |
| mosaic | 1 |
| mixup / copy_paste | 0 / 0 |
| erasing | 0.2 |
| save_period | -1（保存best.pt和last.pt） |

已核对当前Ultralytics 8.4.82的实际代码：auto按
`ceil(2503 / max(batch16, nbs64)) * 120 = 4800`次优化器迭代选择AdamW，
YAML保留4类输出空间，因此初始学习率预计为`0.002 * 5 / (4 + 4) = 0.00125`，
动量（AdamW beta1）为0.9。auto会忽略传入的lr0与momentum，
并将warmup_bias_lr设为0；前3轮预热，随后余弦衰减。
CLI超参数页面若显示0.001，是配置值，不是auto最终选定值；启动日志中的
`optimizer: AdamW(lr=0.00125, momentum=0.9)`才是实际依据。
auto不保证曲线更平滑，需保持同一划分及评估精度比较结果。
增强沿用之前指定旧实验的参数。`erasing`属于分类任务增强，在此分割任务中不触发。

## 启动

```bash
/home/ljz/miniconda3/envs/yolo/bin/python /home/ljz/vibe_coding/YOLO/YOLO-LAB-CLI/main.py
```

选 **1**，之后回车确认默认设置：余弦调度、开启增强、Mixup为0，
不保存每一轮权重。Mixup提示输入`y`会启用0.2，保持当前设置应直接回车。

独立实验名：`f2_t8v2_n3129_yolov8n-seg_pretrained_auto_e120_freeze0_V1`。
用户自行启动训练，本次只修改配置和进行运行前检查。

上一版AdamW lr0=0.001的120轮实验已完成并保留：
`outputs/result/f2_t8v2_n3129_yolov8n-seg_pretrained_adamw_e120_lr1e-3_freeze0_V1`。
本次仅改变optimizer及实验名，其余训练与增强参数沿用上一版；
不从上一版best/last.pt续训，仍从官方yolov8n-seg.pt开始。
自动监控保持暂停；训练完成后告诉助手，再分析并交付本次版本。

## 检查与后续评估

重新训练的配置及CLI确认检查记录在 `outputs/f2_retrain_setup_check.json`。
此前已通过F2实际数据加载和三类标注完整性检查。Ultralytics在训练集的
276.png和3368.png各识别并去除一条重复标注，仅影响加载后的数据，原标签保留。

上次同一微调best.pt在相同验证集上，FP16复核的Mask mAP50-95为55.78%，
FP32复核为51.89%；两次复核仅改变quantize精度设置。比较新旧模型时应固定
评估精度、验证名单和推理参数，避免将设置差异当作训练效果变化。

A16的64张图片已核实有50张属于F2训练集、14张属于验证集。对A16的可视化
推理适合人工核对，不能当作独立测试集结果。

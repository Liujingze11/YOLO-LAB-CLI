#!/usr/bin/env python3
"""e60 模型推理脚本 — 输出标签 txt 到指定目录"""

from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("source", help="图片目录路径")
parser.add_argument("--name", default="predict_output", help="输出文件夹名")
args = parser.parse_args()

model = YOLO("<path_to_best.pt>")

results = model.predict(
    source=args.source,
    imgsz=640, conf=0.25, iou=0.7,
    save_txt=True, save_conf=True, save=True,
    project="<output_project_dir>", name=args.name,
    exist_ok=True, device='0',
    verbose=True,
)
print(f"Done: {len(results)} images -> <output_project_dir>/{args.name}")

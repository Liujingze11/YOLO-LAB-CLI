#!/usr/bin/env python3
"""用 V2-1 best.pt 推理所有 train+val 图片，结果保存到桌面。"""
import os
import cv2
from ultralytics import YOLO

MODEL = "<path_to_best.pt>"
DATASET = "<path_to_dataset>"
OUTPUT = os.path.expanduser("<path_to_output>")

model = YOLO(MODEL)

for split in ["train", "val"]:
    src_dir = os.path.join(DATASET, "images", split)
    out_dir = os.path.join(OUTPUT, split)
    images = sorted([f for f in os.listdir(src_dir) if f.endswith(".png")])
    total = len(images)
    print(f"[{split}] {total} 张图片")

    for i, img in enumerate(images, 1):
        src = os.path.join(src_dir, img)
        results = model(src, conf=0.25, verbose=False)
        # plot 用更小的字号
        plotted = results[0].plot(font_size=10, line_width=1)
        save_path = os.path.join(out_dir, img)
        cv2.imwrite(save_path, plotted)
        if i % 100 == 0:
            print(f"  [{split}] {i}/{total}")

    print(f"  [{split}] 完成: {total}/{total}")

print(f"\n全部完成 → {OUTPUT}")

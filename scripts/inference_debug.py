#!/usr/bin/env python3
"""Run YOLO inference on debug_image folder with 3 confidence thresholds."""

import os
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "/home/ljz/vibe_coding/YOLO/YOLO-LAB-CLI/outputs/result/6cls_mix4_finetune_e100_lr0001_from_e150/weights/best.pt"
SOURCE_DIR = "/home/ljz/桌面/debug_image"
OUTPUT_BASE = "/home/ljz/桌面/debug_inference"

CONFIDENCES = [0.7, 0.8, 0.9]
# Format folder name: conf_0.7 etc.
FOLDER_NAMES = {c: f"conf_{str(c).replace('.', '_')}" for c in CONFIDENCES}

def main():
    model = YOLO(MODEL_PATH)

    image_files = sorted(Path(SOURCE_DIR).glob("*.png"))
    total = len(image_files)
    print(f"Found {total} images to process.\n")

    for idx, img_path in enumerate(image_files, 1):
        stem = img_path.stem  # e.g. "000001_pose"
        print(f"[{idx}/{total}] Processing {img_path.name}...", end=" ", flush=True)

        for conf in CONFIDENCES:
            out_name = f"{stem}.jpg"
            out_path = Path(OUTPUT_BASE) / FOLDER_NAMES[conf] / out_name

            # Run inference with specific confidence threshold
            results = model(img_path, conf=conf, verbose=False)
            results[0].save(filename=str(out_path))

        print("done")

    print("\n=== Done ===")
    for conf in CONFIDENCES:
        folder = Path(OUTPUT_BASE) / FOLDER_NAMES[conf]
        count = len(list(folder.glob("*")))
        print(f"  {FOLDER_NAMES[conf]}: {count} images")


if __name__ == "__main__":
    main()

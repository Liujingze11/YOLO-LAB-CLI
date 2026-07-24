#!/usr/bin/env python3
"""Run YOLO inference on a debug image folder with several confidence thresholds."""

import argparse
from pathlib import Path
from ultralytics import YOLO

DEFAULT_MODEL_PATH = "<path_to_best.pt>"
DEFAULT_SOURCE_DIR = "<path_to_debug_images>"
DEFAULT_OUTPUT_DIR = "<path_to_debug_output>"

CONFIDENCES = [0.7, 0.8, 0.9]
# Format folder name: conf_0.7 etc.
FOLDER_NAMES = {c: f"conf_{str(c).replace('.', '_')}" for c in CONFIDENCES}


def parse_args():
    parser = argparse.ArgumentParser(description="Run debug inference at several confidence thresholds")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to best.pt")
    parser.add_argument("--source", default=DEFAULT_SOURCE_DIR, help="Folder containing debug images")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Base folder for debug outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)

    source_dir = Path(args.source)
    output_dir = Path(args.output)
    image_files = sorted(source_dir.glob("*.png"))
    total = len(image_files)
    print(f"Found {total} images to process.\n")

    for idx, img_path in enumerate(image_files, 1):
        stem = img_path.stem  # e.g. "000001_pose"
        print(f"[{idx}/{total}] Processing {img_path.name}...", end=" ", flush=True)

        for conf in CONFIDENCES:
            out_name = f"{stem}.jpg"
            out_path = output_dir / FOLDER_NAMES[conf] / out_name

            # Run inference with specific confidence threshold
            results = model(img_path, conf=conf, verbose=False)
            results[0].save(filename=str(out_path))

        print("done")

    print("\n=== Done ===")
    for conf in CONFIDENCES:
        folder = output_dir / FOLDER_NAMES[conf]
        count = len(list(folder.glob("*")))
        print(f"  {FOLDER_NAMES[conf]}: {count} images")


if __name__ == "__main__":
    main()

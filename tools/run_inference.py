"""YOLO 分割推理脚本:对图片文件夹逐张推理,输出带小字标注的图片(保持原目录结构)。

用法:
    python run_inference.py --model <path_to_best.pt> --source <输入图片文件夹> --output <输出文件夹>
    python run_inference.py --model ... --source ... --output ... --conf 0.5 --font-size 9 --line-width 1

说明:
    置信度默认 0.7,文字字号默认 9(小字),框线宽默认 1(细线)。
    已存在于输出目录的图片会跳过,损坏的图片直接复制原图保留。
"""
import argparse
import os
import shutil
import cv2
from ultralytics import YOLO

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")


def parse_args():
    # 声明全部开关;路径默认值用占位符,防止个人路径泄露到仓库,使用时传参
    parser = argparse.ArgumentParser(description="对图片文件夹做 YOLO 分割推理,输出带标注的图片")
    parser.add_argument("--model", default="<path_to_best.pt>", help="模型权重路径,如 best.pt")
    parser.add_argument("--source", default="<path_to_images>", help="输入图片文件夹")
    parser.add_argument("--output", default="<path_to_output>", help="输出文件夹")
    parser.add_argument("--conf", type=float, default=0.7, help="置信度阈值(默认: 0.7)")
    parser.add_argument("--font-size", type=float, default=9, help="标注文字字号(默认: 9)")
    parser.add_argument("--line-width", type=int, default=1, help="标注框线宽(默认: 1)")
    return parser.parse_args()


def run_inference(args) -> None:
    model_path = args.model
    source_dir = args.source
    output_dir = args.output

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    os.makedirs(output_dir, exist_ok=True)

    # 收集所有图片(保持原有目录结构)
    all_images = []
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                all_images.append(os.path.join(root, f))

    print(f"共找到 {len(all_images)} 张图片")

    processed = 0
    skipped = 0

    for img_path in all_images:
        processed += 1

        # 跳过已处理的
        rel_path = os.path.relpath(img_path, source_dir)
        out_path = os.path.join(output_dir, rel_path)
        if os.path.exists(out_path):
            continue

        try:
            # 推理 (不保存,自己控制输出)
            results = model.predict(img_path, conf=args.conf, verbose=False, save=False)
            result = results[0]

            # 用 result.plot() 绘制,它内部处理 mask 缩放等问题
            annotated_im = result.plot(
                conf=True,
                line_width=args.line_width,
                font_size=args.font_size,
                labels=True,
                boxes=True,
                masks=True,
            )

            # 保持原有目录结构保存
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, annotated_im)
        except Exception as e:
            # 损坏/无法读取的文件 -> 直接复制原图保留
            print(f"[{processed}/{len(all_images)}] 损坏文件,复制原图: {os.path.basename(img_path)}")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            shutil.copy2(img_path, out_path)
            skipped += 1
            continue

        if processed % 500 == 0:
            print(f"[{processed}/{len(all_images)}] 已处理...")

    print(f"\n===== 推理完成 =====")
    print(f"处理图片: {processed}")
    print(f"其中成功推理: {processed - skipped}")
    print(f"损坏文件(复制原图): {skipped}")
    print(f"输出目录: {output_dir}")


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()

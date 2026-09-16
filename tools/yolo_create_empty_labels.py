"""为训练数据中的每张图片创建同名空标签 txt(已存在的标签绝不覆盖)。

用法(相对路径按运行命令时所在的目录解释,绝对路径同样可以):
    python yolo_create_empty_labels.py                                  # 默认: images/ → labels/
    python yolo_create_empty_labels.py --images images --labels labels  # 自定义文件夹名
    python yolo_create_empty_labels.py --images data/train/images --labels data/train/labels
    python yolo_create_empty_labels.py --images /path/to/images --labels /path/to/labels  # 绝对路径也行
"""
import argparse
import os

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    # 声明 --images / --labels 两个开关,默认值采用官方格式的顶层文件夹(images 与 labels)
    parser = argparse.ArgumentParser(description="为图片创建同名空标签 txt,已存在的标签绝不覆盖")
    parser.add_argument("--images", default="images", help="训练图片文件夹(默认: images)")
    parser.add_argument("--labels", default="labels", help="标签文件夹(默认: labels)")
    return parser.parse_args()


def create_empty_labels(image_dir: str, label_dir: str) -> tuple:
    """为 image_dir 中每张图片创建同名空标签 txt,返回 (新建数, 跳过数)。

    跳过 = 标签已存在,绝不覆盖其内容。
    """
    os.makedirs(label_dir, exist_ok=True)

    created = 0
    skipped = 0
    for filename in os.listdir(image_dir):
        name, ext = os.path.splitext(filename)
        if ext.lower() in IMAGE_EXTS:
            label_path = os.path.join(label_dir, f"{name}.txt")
            if not os.path.exists(label_path):
                with open(label_path, "w", encoding="utf-8") as f:
                    pass
                created += 1
            else:
                skipped += 1
    return created, skipped


def main():
    args = parse_args()
    created, skipped = create_empty_labels(args.images, args.labels)
    print(f"完成: 新建 {created} 个空标签,跳过 {skipped} 个已存在的标签 → {args.labels}")


if __name__ == "__main__":
    main()

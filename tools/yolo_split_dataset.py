"""把 YOLO 训练数据划分成 train/val(可选 test),图片与同名标签同步移动,输出官方格式目录。

用法(相对路径按运行命令时所在的目录解释,绝对路径同样可以):
    python yolo_split_dataset.py --images data/images --labels data/labels --ratio 0.2
    python yolo_split_dataset.py --images /path/to/data/images --labels /path/to/data/labels --ratio 0.2
    python yolo_split_dataset.py --images data/images --split train_val_test --test-ratio 0.1
    python yolo_split_dataset.py --images data/images --mode every --every 5
    python yolo_split_dataset.py --no-labels --images data/images --ratio 0.2
    python yolo_split_dataset.py --dash --images data/images --ratio 0.2      # 支持 1681-2 这类文件名

运行前:data/images 与 data/labels 里直接放着图片与同名标签
运行后:两个目录下各生成 train/、val/(可选 test/),抽中的图片进 val/test,其余全部进 train;
       源目录不再有图片文件,重复运行会直接报错。

说明:
    --labels 也可不传:data/images 自动对应 data/labels;图片文件夹不叫 images 时,必须用 --labels 自己指定。
    文件名默认只认纯数字(如 1681.jpg),其余跳过并警告;--dash 支持 1681-2、100_5 这类名字。
"""
import argparse
import os
import random
import shutil

VALID_EXT = (".jpg", ".jpeg", ".png")


def _list_images(image_dir: str) -> list:
    """返回 image_dir 下所有图片文件名(按支持的后缀过滤)。"""
    return [f for f in os.listdir(image_dir) if f.lower().endswith(VALID_EXT)]


def _sort_key_dash(filename: str):
    """按文件名(不含后缀)中的数字段排序,支持 1681、1681-2、100_5 这类格式。"""
    name = os.path.splitext(filename)[0]
    parts = name.replace("-", " ").replace("_", " ").split()
    return tuple(int(p) for p in parts)


def _strict_numeric_images(images: list) -> list:
    """只保留"去掉后缀后是纯数字"的图片,其余跳过并警告。"""
    valid = []
    for f in images:
        if os.path.splitext(f)[0].isdigit():
            valid.append(f)
        else:
            print(f"跳过非纯数字文件名图片: {f}")
    return valid


def _derive_labels_dir(images_dir: str, labels_override) -> str:
    """推导标签根目录:优先用 --labels,否则把 images 路径中的 images 替换为 labels。"""
    if labels_override:
        return labels_override
    if "images" not in images_dir:
        raise ValueError("--images 路径中不含 images,无法自动推导标签路径,请显式传 --labels。")
    return images_dir.replace("images", "labels")


def _move_batch(image_list, images_src, images_dst, labels_src, labels_dst, no_labels):
    """移动一批图片(及同名标签),返回 (移动图片数, 移动标签数, 缺失标签数)。"""
    os.makedirs(images_dst, exist_ok=True)
    if not no_labels:
        os.makedirs(labels_dst, exist_ok=True)

    moved_img = moved_label = missing = 0
    for img in image_list:
        name = os.path.splitext(img)[0]
        shutil.move(os.path.join(images_src, img), os.path.join(images_dst, img))
        moved_img += 1
        if no_labels:
            continue
        label_src = os.path.join(labels_src, f"{name}.txt")
        if os.path.exists(label_src):
            shutil.move(label_src, os.path.join(labels_dst, f"{name}.txt"))
            moved_label += 1
        else:
            print(f"警告:未找到对应标签文件 {label_src}")
            missing += 1
    return moved_img, moved_label, missing


def run_split(cfg) -> dict:
    """按 cfg 划分数据集,移动图片与同名标签,返回统计信息字典。"""
    labels_root = _derive_labels_dir(cfg.images, cfg.labels)

    # ---- 扫描与排序 ----
    images = _list_images(cfg.images)
    if cfg.dash and cfg.mode == "random":
        valid_images = sorted(images, key=_sort_key_dash)
    else:
        valid_images = _strict_numeric_images(images)
        valid_images.sort(key=lambda x: int(os.path.splitext(x)[0]))

    total_count = len(valid_images)
    if total_count == 0:
        raise ValueError("训练数据中没有可用图片(已划分过的数据源里不会再有图片文件)。")

    # ---- 抽取 val/test ----
    if cfg.mode == "every":
        if cfg.split == "train_val_test":
            raise ValueError("every 模式只支持 train_val 两份划分。")
        val_images = [img for img in valid_images
                      if int(os.path.splitext(img)[0]) % cfg.every == 0]
        if not val_images:
            raise ValueError(f"每隔 {cfg.every} 张取 1 张,但没有图片符合条件,请调小 --every。")
        if len(val_images) == total_count:
            raise ValueError(f"每隔 {cfg.every} 张取 1 张,全部图片都被抽中,请调大 --every。")
        test_images = []
    else:
        random.seed(cfg.seed)
        val_count = round(total_count * cfg.ratio)
        if val_count < 1:
            raise ValueError(f"按照当前比例 {cfg.ratio} 计算,验证集数量小于 1,请调大比例。")
        if val_count >= total_count:
            raise ValueError(f"按照当前比例 {cfg.ratio} 计算,验证集数量达到或超过训练集总数,请调小比例。")
        val_images = random.sample(valid_images, val_count)

        test_images = []
        if cfg.split == "train_val_test":
            test_count = round(total_count * cfg.test_ratio)
            if test_count < 1:
                raise ValueError(f"test_ratio={cfg.test_ratio} 计算后测试集数量小于 1,请调大比例。")
            if val_count + test_count >= total_count:
                raise ValueError(
                    f"val({val_count}) + test({test_count}) >= 总数({total_count}),请调小比例。")
            remaining = [img for img in valid_images if img not in val_images]
            test_images = random.sample(remaining, test_count)

    train_images = [img for img in valid_images
                    if img not in val_images and img not in test_images]

    # ---- 移动 ----
    moved = {}
    for part, image_list in (("val", val_images), ("test", test_images), ("train", train_images)):
        if not image_list:
            continue
        moved[part] = _move_batch(
            image_list, cfg.images, os.path.join(cfg.images, part),
            labels_root, os.path.join(labels_root, part), cfg.no_labels)

    stats = {"total_count": total_count, "val_count": len(val_images),
             "test_count": len(test_images), "train_count": len(train_images)}
    for part, (img_n, label_n, missing_n) in moved.items():
        stats[f"{part}_moved_img"] = img_n
        stats[f"{part}_moved_label"] = label_n
        stats[f"{part}_missing_label"] = missing_n

    # ---- 输出 ----
    print(f"原图片总数: {total_count}")
    if cfg.mode == "every":
        print(f"每隔 {cfg.every} 张取 1 张 → val {len(val_images)} 张 / train {len(train_images)} 张")
    else:
        print(f"按比例划分 → val {len(val_images)} 张({cfg.ratio * 100:.1f}%) / train {len(train_images)} 张")
    if test_images:
        print(f"test {len(test_images)} 张({cfg.test_ratio * 100:.1f}%)")
    missing_total = sum(v[2] for v in moved.values())
    if missing_total:
        print(f"缺失标签数量: {missing_total}")
    print(f"图片已就绪: {cfg.images}/train、{cfg.images}/val"
          + (f"、{cfg.images}/test" if test_images else ""))
    if not cfg.no_labels:
        print(f"标签已就绪: {labels_root}/train、{labels_root}/val"
              + (f"、{labels_root}/test" if test_images else ""))

    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="把 YOLO 训练数据划分成 train/val(/test),图片与同名标签同步移动")
    parser.add_argument("--images", required=True, help="训练图片文件夹(源)")
    parser.add_argument("--labels", default=None, help="标签文件夹(默认: 由 images 路径推导)")
    parser.add_argument("--split", choices=["train_val", "train_val_test"], default="train_val",
                        help="划分份数(默认: train_val)")
    parser.add_argument("--mode", choices=["random", "every"], default="random",
                        help="划分方式(默认: random 按比例随机)")
    parser.add_argument("--ratio", type=float, default=0.2, help="验证集比例(默认: 0.2)")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例(默认: 0.1)")
    parser.add_argument("--every", type=int, default=5, help="每隔 N 张取 1 张(默认: 5)")
    parser.add_argument("--no-labels", action="store_true", help="只移动图片,不处理标签")
    parser.add_argument("--dash", action="store_true", help="支持 1681-2 这类带 - 或 _ 分隔符的文件名")
    parser.add_argument("--seed", type=int, default=42, help="随机种子(默认: 42)")
    return parser.parse_args()


def main():
    args = parse_args()
    run_split(args)


if __name__ == "__main__":
    main()

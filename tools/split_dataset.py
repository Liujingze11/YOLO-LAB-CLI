"""把 train 按比例随机或按编号每隔 N 张抽出 1 张到 val(可选 test),图片与同名标签同步移动。

用法:
    python split_dataset.py --images-train data/train/images --images-val data/val/images
    python split_dataset.py --images-train ... --labels-train ... --images-val ... --labels-val ... --ratio 0.2
    python split_dataset.py --split train_val_test --images-train ... --images-val ... --images-test ... --labels-test ... --test-ratio 0.1
    python split_dataset.py --mode every --every 5 --images-train ... --images-val ...
    python split_dataset.py --no-labels --images-train ... --images-val ...
    python split_dataset.py --dash --images-train ... --images-val ...     # 支持 1681-2 这类文件名

说明:
    --labels-* 不传时,自动把对应 --images-* 路径中的 "images" 替换为 "labels" 推导。
    默认只处理纯数字文件名(如 1681.jpg),其余跳过并警告;--dash 可支持带 - 或 _ 的文件名。
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
    """只保留"去掉后缀后是纯数字"的图片,其余跳过并警告(与旧脚本一致)。"""
    valid = []
    for f in images:
        if os.path.splitext(f)[0].isdigit():
            valid.append(f)
        else:
            print(f"跳过非纯数字文件名图片: {f}")
    return valid


def _move_one(img, images_train_dir, images_dir, labels_train_dir, labels_dir):
    """移动一张图片和它的同名标签,返回 (移动图片数, 移动标签数, 缺失标签数)。"""
    name = os.path.splitext(img)[0]
    shutil.move(os.path.join(images_train_dir, img), os.path.join(images_dir, img))
    moved_img = 1

    label_src = os.path.join(labels_train_dir, f"{name}.txt")
    if os.path.exists(label_src):
        shutil.move(label_src, os.path.join(labels_dir, f"{name}.txt"))
        return 1, 1, 0
    print(f"警告:未找到对应标签文件 {label_src}")
    return 1, 0, 1


def _move_batch(image_list, images_train_dir, images_dir,
                labels_train_dir, labels_dir, no_labels):
    """移动一批图片(及同名标签),返回 (移动图片数, 移动标签数, 缺失标签数)。"""
    os.makedirs(images_dir, exist_ok=True)
    if not no_labels:
        os.makedirs(labels_dir, exist_ok=True)

    moved_img = moved_label = missing = 0
    for img in image_list:
        if no_labels:
            shutil.move(os.path.join(images_train_dir, img),
                        os.path.join(images_dir, img))
            moved_img += 1
        else:
            img_n, label_n, missing_n = _move_one(
                img, images_train_dir, images_dir, labels_train_dir, labels_dir)
            moved_img += img_n
            moved_label += label_n
            missing += missing_n
    return moved_img, moved_label, missing


def run_split(cfg) -> dict:
    """按 cfg 划分数据集,移动图片与同名标签,返回统计信息字典。"""
    labels_train_dir = cfg.labels_train or cfg.images_train.replace("images", "labels")

    images = _list_images(cfg.images_train)

    # ---- 筛选与排序 ----
    if cfg.dash and cfg.mode == "random":
        valid_images = sorted(images, key=_sort_key_dash)
    else:
        valid_images = _strict_numeric_images(images)
        valid_images.sort(key=lambda x: int(os.path.splitext(x)[0]))

    total_count = len(valid_images)
    if total_count == 0:
        raise ValueError("train 文件夹中没有可用图片。")

    stats = {"total_count": total_count, "val_count": 0, "test_count": 0}

    # ---- 抽取 val(以及可选 test)----
    if cfg.mode == "every":
        if cfg.split == "train_val_test":
            raise ValueError("every 模式只支持 train_val 两份划分。")
        val_images = [img for img in valid_images
                      if int(os.path.splitext(img)[0]) % cfg.every == 0]
        stats["val_count"] = len(val_images)
        test_images = []
    else:
        random.seed(cfg.seed)
        val_count = round(total_count * cfg.ratio)
        if val_count < 1:
            raise ValueError(f"按照当前比例 {cfg.ratio} 计算,验证集数量小于 1,请调大比例。")
        if val_count >= total_count:
            raise ValueError(f"按照当前比例 {cfg.ratio} 计算,验证集数量达到或超过训练集总数,请调小比例。")
        val_images = random.sample(valid_images, val_count)
        stats["val_count"] = val_count

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
            stats["test_count"] = test_count

    # ---- 移动图片与标签 ----
    labels_val_dir = cfg.labels_val or cfg.images_val.replace("images", "labels")
    moved_val_img, moved_val_label, missing_val_label = _move_batch(
        val_images, cfg.images_train, cfg.images_val,
        labels_train_dir, labels_val_dir, cfg.no_labels)

    moved_test_img = moved_test_label = missing_test_label = 0
    if test_images:
        labels_test_dir = cfg.labels_test or cfg.images_test.replace("images", "labels")
        moved_test_img, moved_test_label, missing_test_label = _move_batch(
            test_images, cfg.images_train, cfg.images_test,
            labels_train_dir, labels_test_dir, cfg.no_labels)

    stats.update({
        "val_moved_img": moved_val_img,
        "val_moved_label": moved_val_label,
        "val_missing_label": missing_val_label,
        "test_moved_img": moved_test_img,
        "test_moved_label": moved_test_label,
        "test_missing_label": missing_test_label,
    })

    # ---- 输出结果(与旧脚本格式一致)----
    print(f"原 train 图片总数: {total_count}")
    if cfg.mode == "every":
        print(f"每隔 {cfg.every} 张取 1 张,移动到 val: {moved_val_img} 张图片,{moved_val_label} 个标签")
    else:
        print(f"设置比例: {cfg.ratio * 100:.1f}%")
        print(f"计划移动到 val 的图片数量: {stats['val_count']}")
        print(f"实际移动图片数量: {moved_val_img}")
        print(f"实际移动标签数量: {moved_val_label}")
    print(f"缺失标签数量: {missing_val_label}")
    print(f"已完成,图片已移动到: {cfg.images_val}")
    if not cfg.no_labels:
        print(f"已完成,标签已移动到: {labels_val_dir}")

    if test_images:
        print(f"\n[测试集 test]")
        print(f"设置比例: {cfg.test_ratio * 100:.1f}%")
        print(f"实际移动图片数量: {moved_test_img}")
        print(f"实际移动标签数量: {moved_test_label}")
        print(f"缺失标签数量: {missing_test_label}")
        print(f"\n[剩余 train]")
        print(f"剩余图片数量: {total_count - moved_val_img - moved_test_img}")

    return stats


def parse_args():
    # 声明全部开关;--labels-* 不传时由对应 --images-* 路径自动推导(images→labels)
    parser = argparse.ArgumentParser(description="把 train 按比例或编号分出 val(可选 test),图片与同名标签同步移动")
    parser.add_argument("--images-train", required=True, help="训练图片文件夹(源)")
    parser.add_argument("--images-val", required=True, help="验证图片文件夹(目标)")
    parser.add_argument("--images-test", default=None, help="测试图片文件夹(目标,--split train_val_test 时必填)")
    parser.add_argument("--labels-train", default=None, help="训练标签文件夹(默认: 由 images 路径推导)")
    parser.add_argument("--labels-val", default=None, help="验证标签文件夹(默认: 由 images 路径推导)")
    parser.add_argument("--labels-test", default=None, help="测试标签文件夹(默认: 由 images 路径推导)")
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
    if args.split == "train_val_test" and not args.images_test:
        raise ValueError("--split train_val_test 需要同时传入 --images-test。")
    run_split(args)


if __name__ == "__main__":
    main()

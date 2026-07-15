import os
import random
import shutil

# ===== 路径配置 =====
base_dir = "/home/ljz/Train_logging/Data/YOLO训练/mix6"   # 目标数据集文件夹

images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

images_train_dir = os.path.join(base_dir, "images", "train")
images_val_dir   = os.path.join(base_dir, "images", "val")
images_test_dir  = os.path.join(base_dir, "images", "test")

labels_train_dir = os.path.join(base_dir, "labels", "train")
labels_val_dir   = os.path.join(base_dir, "labels", "val")
labels_test_dir  = os.path.join(base_dir, "labels", "test")

# ===== 比例配置 =====
val_ratio = 0.20   # 验证集比例
test_ratio = 0.10  # 测试集比例

# 固定随机种子，保证每次结果一致
random.seed(42)

# 支持的图片格式
valid_ext = (".jpg", ".jpeg", ".png")

# =========================
# 步骤1: 将当前扁平结构变为 train 目录（所有文件先归入 train）
# =========================
os.makedirs(images_train_dir, exist_ok=True)
os.makedirs(labels_train_dir, exist_ok=True)
os.makedirs(images_val_dir, exist_ok=True)
os.makedirs(images_test_dir, exist_ok=True)
os.makedirs(labels_val_dir, exist_ok=True)
os.makedirs(labels_test_dir, exist_ok=True)

# 移动所有图片到 images/train/
print("===== 步骤1: 将所有文件移入 train 目录 =====")
images = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_ext)]
moved_img = 0
moved_lbl = 0
for f in images:
    name_no_ext = os.path.splitext(f)[0]
    # 移动图片
    img_src = os.path.join(images_dir, f)
    img_dst = os.path.join(images_train_dir, f)
    if os.path.isfile(img_src):
        shutil.move(img_src, img_dst)
        moved_img += 1
    # 移动标签
    label_name = name_no_ext + ".txt"
    label_src = os.path.join(labels_dir, label_name)
    label_dst = os.path.join(labels_train_dir, label_name)
    if os.path.isfile(label_src):
        shutil.move(label_src, label_dst)
        moved_lbl += 1

print(f"已移入 train: {moved_img} 张图片, {moved_lbl} 个标签")

# =========================
# 步骤2: 从 train 中随机抽取 val 和 test
# =========================
print("\n===== 步骤2: 随机划分 val / test =====")

images_in_train = [f for f in os.listdir(images_train_dir) if f.lower().endswith(valid_ext)]

# 只保留"纯数字文件名"的图片（与原脚本逻辑一致）
valid_images = []
for f in images_in_train:
    name_without_ext = os.path.splitext(f)[0]
    if name_without_ext.isdigit():
        valid_images.append(f)
    else:
        print(f"跳过非纯数字文件名图片: {f}")

# 按数字排序
valid_images.sort(key=lambda x: int(os.path.splitext(x)[0]))

total_count = len(valid_images)
val_count = round(total_count * val_ratio)
test_count = round(total_count * test_ratio)

print(f"有效图片总数: {total_count}")
print(f"验证集数量: {val_count} ({val_ratio*100:.0f}%)")
print(f"测试集数量: {test_count} ({test_ratio*100:.0f}%)")

if total_count == 0:
    raise ValueError("train 文件夹中没有可用图片。")

if val_count < 1:
    raise ValueError(f"val_ratio={val_ratio} 计算后验证集数量小于 1，请调大比例。")

if test_count < 1:
    raise ValueError(f"test_ratio={test_ratio} 计算后测试集数量小于 1，请调大比例。")

if val_count + test_count >= total_count:
    raise ValueError(
        f"val({val_count}) + test({test_count}) >= 总数({total_count})，请调小比例。"
    )

# 随机抽取 val
val_images = random.sample(valid_images, val_count)

# 剩余图片中抽 test
remaining_images = [img for img in valid_images if img not in val_images]
test_images = random.sample(remaining_images, test_count)


def move_files(image_list, target_images_dir, target_labels_dir):
    moved_img_count = 0
    moved_label_count = 0
    missing_label_count = 0

    for img in image_list:
        name_without_ext = os.path.splitext(img)[0]

        # 移动图片
        img_src = os.path.join(images_train_dir, img)
        img_dst = os.path.join(target_images_dir, img)
        if os.path.exists(img_src):
            shutil.move(img_src, img_dst)
            moved_img_count += 1
        else:
            print(f"警告：未找到图片文件 {img_src}")
            continue

        # 移动标签
        label_name = name_without_ext + ".txt"
        label_src = os.path.join(labels_train_dir, label_name)
        label_dst = os.path.join(target_labels_dir, label_name)
        if os.path.exists(label_src):
            shutil.move(label_src, label_dst)
            moved_label_count += 1
        else:
            print(f"警告：未找到对应标签文件 {label_src}")
            missing_label_count += 1

    return moved_img_count, moved_label_count, missing_label_count


# 执行移动
moved_val_img, moved_val_label, missing_val_label = move_files(
    val_images, images_val_dir, labels_val_dir
)
moved_test_img, moved_test_label, missing_test_label = move_files(
    test_images, images_test_dir, labels_test_dir
)

# ===== 输出结果 =====
print("\n===== 数据集划分完成 =====")
print(f"原 train 图片总数: {total_count}")

print(f"\n[验证集 val]")
print(f"设置比例: {val_ratio * 100:.1f}%")
print(f"计划移动图片数量: {val_count}")
print(f"实际移动图片数量: {moved_val_img}")
print(f"实际移动标签数量: {moved_val_label}")
print(f"缺失标签数量: {missing_val_label}")
print(f"图片目标路径: {images_val_dir}")
print(f"标签目标路径: {labels_val_dir}")

print(f"\n[测试集 test]")
print(f"设置比例: {test_ratio * 100:.1f}%")
print(f"计划移动图片数量: {test_count}")
print(f"实际移动图片数量: {moved_test_img}")
print(f"实际移动标签数量: {moved_test_label}")
print(f"缺失标签数量: {missing_test_label}")
print(f"图片目标路径: {images_test_dir}")
print(f"标签目标路径: {labels_test_dir}")

print(f"\n[剩余 train]")
remaining_in_train = [f for f in os.listdir(images_train_dir) if f.lower().endswith(valid_ext)]
remaining_lbl = [f for f in os.listdir(labels_train_dir) if f.lower().endswith(".txt")]
print(f"剩余图片数量: {len(remaining_in_train)}")
print(f"剩余标签数量: {len(remaining_lbl)}")
print(f"图片路径: {images_train_dir}")
print(f"标签路径: {labels_train_dir}")

"""
YOLOv8n-seg 推理脚本
- 使用置信度 0.7 进行推理
- 输出小文字标注的图片到桌面
"""
import os
import shutil
import cv2
from ultralytics import YOLO

# ============ 配置 ============
MODEL_PATH = '<path_to_model.pt>'
SOURCE_DIR  = '<path_to_images>'
OUTPUT_DIR  = '<path_to_output>'
CONF_THRESHOLD = 0.7
FONT_SIZE   = 9         # 小字体（默认约20+，这里设为9）
LINE_WIDTH  = 1         # 细线框
# =============================

print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 收集所有图片
extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
all_images = []
for root, dirs, files in os.walk(SOURCE_DIR):
    for f in files:
        if f.lower().endswith(extensions):
            all_images.append(os.path.join(root, f))

print(f"共找到 {len(all_images)} 张图片")

processed = 0
skipped = 0

for img_path in all_images:
    processed += 1

    # 跳过已处理的
    rel_path = os.path.relpath(img_path, SOURCE_DIR)
    out_path = os.path.join(OUTPUT_DIR, rel_path)
    if os.path.exists(out_path):
        continue

    try:
        # 推理 (不保存，自己控制输出)
        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False, save=False)
        result = results[0]

        # 用 result.plot() 绘制，它内部处理 mask 缩放等问题
        annotated_im = result.plot(
            conf=True,
            line_width=LINE_WIDTH,
            font_size=FONT_SIZE,
            labels=True,
            boxes=True,
            masks=True,
        )

        # 保持原有目录结构保存
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, annotated_im)
    except Exception as e:
        # 损坏/无法读取的文件 -> 直接复制原图保留
        print(f"[{processed}/{len(all_images)}] 损坏文件，复制原图: {os.path.basename(img_path)}")
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
print(f"输出目录: {OUTPUT_DIR}")

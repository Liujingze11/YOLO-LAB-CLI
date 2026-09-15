"""
YOLOv8n-seg 推理脚本 (final1 val)
- 用 cls3_all_final1_split 训练的 best.pt 对全部 val 图片推理
- 输出小文字标注的图片到桌面
"""
import os
import cv2
from ultralytics import YOLO

# ============ 配置 ============
MODEL_PATH = '<path_to_best.pt>'
SOURCE_DIR = '<path_to_val_images>'
OUTPUT_DIR = '<path_to_output>'
CONF_THRESHOLD = 0.504
FONT_SIZE = 9          # 小字体（默认约20+，这里设为9）
LINE_WIDTH = 1         # 细线框
# =============================

print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 目录批量推理（GPU 加速）
results = model.predict(SOURCE_DIR, conf=CONF_THRESHOLD, verbose=False, save=False)

saved = 0
for result in results:
    # 用 result.plot() 绘制，它内部处理 mask 缩放等问题
    annotated_im = result.plot(
        conf=True,
        line_width=LINE_WIDTH,
        font_size=FONT_SIZE,
        labels=True,
        boxes=True,
        masks=True,
    )
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(result.path))
    cv2.imwrite(out_path, annotated_im)
    saved += 1
    if saved % 200 == 0:
        print(f"[{saved}/{len(results)}] 已处理...")

print(f"\n===== 推理完成 =====")
print(f"处理图片: {saved}")
print(f"输出目录: {OUTPUT_DIR}")

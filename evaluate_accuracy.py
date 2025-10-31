import os
import pandas as pd
import torch
from ultralytics import YOLO

MODEL_PATHS = [
    "D:\\Codefield\\MyPython\\DeepLearning\\Projects\\FinalProject\\FP_test\\models\\best_4th.pt",
    "D:\\Codefield\\MyPython\\DeepLearning\\Projects\\FinalProject\\FP_test\\models\\best_liu.pt"
]
# 输出目录，以萨摩耶为例
OUT_DIR = "D:\\Codefield\\MyPython\\DeepLearning\\Projects\\FinalProject\\FP_test\\runs\\class_result\\萨摩耶"

# ==============================
# 模型与配置
# ==============================
model = YOLO(MODEL_PATHS[0])
names = model.names

# CSV 路径
csv_path = os.path.join(OUT_DIR, "fusion_results.csv")

# ==============================
# 输入
# ==============================
labels = [
    "cat_bo", "cat_mm", "cat_nn", "cat_hw", "cat_jm", "cat_teq",
    "dog_bige", "dog_guibing", "dog_habar", "dog_alsj", "dog_dbx", "dog_hsq", "dog_smy"
]

print("\n 可选类别：", ", ".join(labels))
label = input("请输入要计算正确率的类别：").strip()

if label not in labels:
    print(f"输入无效！'{label}' 不在已知类别中。")
    exit()

target_id = labels.index(label)

# ==============================
# CSV 文件读取与统计
# ==============================
if not os.path.exists(csv_path):
    print(f"未找到结果文件：{csv_path}")
    exit()

df = pd.read_csv(csv_path)
if "cls" not in df.columns:
    print("CSV 文件中未找到 'cls' 列，请检查文件结构。")
    exit()

# 统计
total = len(df)
flag = (df["cls"] == target_id).sum()
acc = flag / total if total > 0 else 0

# ==============================
# 结果输出
# ==============================
print(f"\n类别：{label}")
print(f"预测正确数量：{flag}/{total}")
print(f"正确率：{acc*100:.2f}%")

if acc < 0.8:
    print("模型对该类别识别效果较弱，建议检查样本质量或类别平衡。")
else:
    print("该类别预测表现良好！")

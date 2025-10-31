import os
import shutil
from collections import Counter

# 可选图像增强
try:
    import cv2
    import albumentations as A
    AUGMENT = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=10, p=0.2),
    ])
except ImportError:
    AUGMENT = None

# 自动增强少数类样本（复制图像与标签），可用于 train 或 valid 集。
# Args:
#     images_path (str): 图像目录路径，例如 "data/train/images"
#     augment_factor (int): 每个少数类样本复制次数（默认2）
#     use_aug (bool): 是否使用 Albumentations 图像增强（默认False）
def balance_dataset(images_path: str, augment_factor: int = 2, use_aug: bool = False):

    labels_path = images_path.replace("images", "labels")
    if not os.path.exists(labels_path):
        print(f" 标签目录不存在: {labels_path}")
        return

    # 统计类别分布
    counter = Counter()
    for lbl_file in os.listdir(labels_path):
        with open(os.path.join(labels_path, lbl_file)) as f:
            for line in f:
                if line.strip():
                    cls = int(line.split()[0])
                    counter[cls] += 1
    if not counter:
        print(f" 未检测到标签: {labels_path}")
        return

    avg = sum(counter.values()) / len(counter)
    minor_classes = [c for c, n in counter.items() if n < 0.7 * avg]

    print(f"\n 检测到 {len(counter)} 类，平均 {avg:.1f}，少数类: {minor_classes}")
    if not minor_classes:
        print(" 数据集平衡，无需增强。")
        return

    # 增强过程
    for lbl_file in os.listdir(labels_path):
        lbl_path = os.path.join(labels_path, lbl_file)
        with open(lbl_path) as f:
            lines = f.readlines()
        cls_set = {int(l.split()[0]) for l in lines}

        if cls_set & set(minor_classes):
            img_name = lbl_file.replace(".txt", ".jpg")
            img_path = os.path.join(images_path, img_name)
            if not os.path.exists(img_path):
                continue
            for i in range(augment_factor):
                new_lbl = os.path.join(labels_path, f"aug_{i}_{lbl_file}")
                new_img = os.path.join(images_path, f"aug_{i}_{img_name}")
                shutil.copy2(lbl_path, new_lbl)
                if use_aug and AUGMENT is not None:
                    import cv2
                    img = cv2.imread(img_path)
                    aug = AUGMENT(image=img)["image"]
                    cv2.imwrite(new_img, aug)
                else:
                    shutil.copy2(img_path, new_img)

    print(f" 已增强 {images_path} 中的少数类样本。\n")

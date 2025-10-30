import os
import random
import datetime
import argparse
from ultralytics import YOLO
import torch
import numpy as np
import yaml
import json

from balance_dataset import balance_dataset

# ------------------------
# 配置区域
# ------------------------
DEFAULT_DATA_YAML = "/root/autodl-tmp/yolo11/alldata/merged_dataset/data.yaml"
MODEL_WEIGHTS = "/root/autodl-tmp/yolo11/models/yolo11l.pt"
IMG_SIZE = 800
EPOCHS = 120
BATCH = 32
WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT = "/root/autodl-tmp/yolo11/runs/train"
RUN_NAME_PREFIX = "yolo11l_alldata"
CLOSE_MOSAIC_EPOCHS = int(EPOCHS * 0.1)
SEED = 42

# ------------------------
# 设定随机种子
# ------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False
    except Exception:
        pass

# ------------------------
# 生成唯一 run name
# ------------------------
def make_run_name(prefix=RUN_NAME_PREFIX):
    t = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{t}"

# ------------------------
# 主训练函数
# ------------------------
def main(args):
    set_seed(args.seed)
    run_name = make_run_name()
    print(f"Run name: {run_name}")
    print(f"Using device: {args.device}, model: {args.weights}")

    # 加载模型
    model = YOLO(args.weights)

    # 读取 data.yaml
    with open(args.data, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    base_dir = os.path.dirname(args.data)
    train_path = os.path.join(base_dir, data_cfg["train"])
    valid_path = os.path.join(base_dir, data_cfg["val"])

    # 平衡数据集
    balance_dataset(train_path, augment_factor=1, use_aug=True)
    balance_dataset(valid_path, augment_factor=1, use_aug=False)

    # 保存训练参数
    save_dir = os.path.join(args.project, run_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # 自动 resume 检查
    last_ckpt = os.path.join(save_dir, "weights", "last.pt")
    resume_flag = os.path.exists(last_ckpt)
    if resume_flag:
        print(f"Resuming from checkpoint: {last_ckpt}")

    # 开始训练
    print("Start training with improved configuration...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=run_name,
        # 数据增强参数
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        perspective=0.0005,  # 新增：轻微透视变换
        translate=0.1,       # 新增轻微平移，提升泛化
        # 优化器 & 学习率策略
        optimizer="AdamW",
        lr0=0.0008,
        lrf=0.05,
        cos_lr=True,         # 余弦退火调度
        warmup_epochs=3.0,
        weight_decay=0.0005,
        # 正则化
        label_smoothing=0.05,
        dropout=0.05,
        # 保存设置
        save=True,
        save_period=-1,      # 仅保存 best.pt 与 last.pt
        resume=resume_flag,
        patience=20,
        amp=True,
        close_mosaic=args.close_mosaic,
        val=True,
        # 可选日志
        exist_ok=True,
    )

    print("Training finished.")
    
    out_dir = os.path.join(args.project, run_name)
    best_path = os.path.join(out_dir, "weights", "best.pt")
    last_path = os.path.join(out_dir, "weights", "last.pt")

    print(f"Output directory: {out_dir}")
    print(f"Best checkpoint: {best_path if os.path.exists(best_path) else 'N/A'}")
    print(f"Last checkpoint: {last_path if os.path.exists(last_path) else 'N/A'}")

    # ------------------------
    # 最终验证与保存结果
    # ------------------------
    try:
        print("Running final validation (no-train)...")
        val_res = model.val(data=args.data, imgsz=args.imgsz)
        print("Validation summary:", val_res)
        with open(os.path.join(out_dir, "val_results.txt"), "w") as f:
            f.write(str(val_res))
    except Exception as e:
        print("Validation step failed:", e)

    # ------------------------
    # 预测可视化
    # ------------------------
    try:
        print("Generating visual predictions...")
        model.predict(
            source=valid_path,
            save=True,
            imgsz=args.imgsz,
            conf=0.25,
            project=args.project,
            name=f"{run_name}_val_vis"
        )
    except Exception as e:
        print("Prediction visualization skipped:", e)

    print("All done ✅")

# ------------------------
# CLI 参数
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced YOLO11-l training script with auto resume and better configs.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_YAML)
    parser.add_argument("--weights", type=str, default=MODEL_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--project", type=str, default=PROJECT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--close_mosaic", type=int, default=CLOSE_MOSAIC_EPOCHS)
    args = parser.parse_args()
    main(args)

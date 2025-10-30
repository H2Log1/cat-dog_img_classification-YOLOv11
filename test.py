import os
import time
import csv
import cv2
import torch
import numpy as np
import colorsys
from pathlib import Path
from ultralytics import YOLO
from torchvision.ops import nms as tv_nms
import matplotlib.pyplot as plt
import matplotlib

# ==============================
# 🔧 用户配置区
# ==============================

MODEL_PATHS = [
    "D:\\Codefield\\MyPython\\DeepLearning\\Projects\\FinalProject\\FP_test\\models\\best_4th.pt",
    "D:\\Codefield\\MyPython\\DeepLearning\\Projects\\FinalProject\\FP_test\\models\\best_liu.pt"
]
SOURCE_DIR = "D:\\Codefield\\MyPython\\DeepLearning\\Projects\\FinalProject\\FP_test\\merged_dataset\\test\\images"
OUT_DIR = "D:\\Codefield\\MyPython\\DeepLearning\\Projects\\FinalProject\\FP_test\\runs\\fusion_TTA"
DATA_YAML = "D:\\Codefield\\MyPython\\DeepLearning\\Projects\\FinalProject\\FP_test\\merged_dataset\\data.yaml"

IMGSZ = 800
CONF_THRES = 0.45
NMS_IOU_FINAL = 0.45
FUSE_IOU = 0.55
DEVICE = "cuda"
SAVE_TXT = True

USE_TTA = True
FUSION_MODE = "wbf"  # "avg" 或 "wbf"

# ==============================
# 工具函数
# ==============================

def ensure_dir(path): Path(path).mkdir(parents=True, exist_ok=True)

def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0

def weighted_boxes_fusion(boxes_list, scores_list, weights, iou_thr=0.4, power=2.0):
    """同类别框的加权融合（安全版本）"""
    dets = []
    for m, (b, s) in enumerate(zip(boxes_list, scores_list)):
        if b.size == 0:
            continue
        w = weights[m] if m < len(weights) else 1.0
        for i in range(len(b)):
            dets.append((b[i], s[i], w))
    dets.sort(key=lambda x: x[1], reverse=True)

    clusters = []
    for box, score, mw in dets:
        merged = False
        for c in clusters:
            if iou_xyxy(box, c["avg_box"]) >= iou_thr:
                w = mw * (score ** power)
                c["sum_w"] += w
                c["sum_score_w"] += w * score
                c["sum_box_w"] += w * box
                c["avg_box"] = c["sum_box_w"] / c["sum_w"]
                merged = True
                break
        if not merged:
            w = mw * (score ** power)
            clusters.append({
                "sum_w": w,
                "sum_score_w": w * score,
                "sum_box_w": w * box,
                "avg_box": box.copy()
            })
    if not clusters:
        return np.zeros((0, 4)), np.zeros((0,))
    fused_boxes = np.array([c["avg_box"] for c in clusters])
    fused_scores = np.array([c["sum_score_w"]/c["sum_w"] for c in clusters])
    return fused_boxes, fused_scores

def nms_torch(boxes, scores, iou_thr=0.5):
    if boxes.size == 0: return np.array([], dtype=int)
    b, s = torch.tensor(boxes), torch.tensor(scores)
    return tv_nms(b, s, iou_thr).cpu().numpy()

def suppress_multi_class_conflicts(boxes, scores, clses, iou_thr=0.6):
    """抑制不同类别之间的高重叠检测（只保留置信度高的）"""
    keep = []
    for i, bi in enumerate(boxes):
        conflict = False
        for j, bj in enumerate(boxes):
            if i == j or clses[i] == clses[j]:
                continue
            if iou_xyxy(bi, bj) > iou_thr and scores[i] < scores[j]:
                conflict = True
                break
        if not conflict:
            keep.append(i)
    if len(keep) == 0:
        return boxes, scores, clses
    return boxes[keep], scores[keep], clses[keep]

# ==============================
# 🎨 可视化（含颜色图例）
# ==============================

def class_color(cls_id):
    hue = (cls_id * 37) % 360 / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
    return int(255 * b), int(255 * g), int(255 * r)

def draw_rounded_rectangle(img, rect, color, radius=4):
    x1, y1, x2, y2 = rect
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
    cv2.addWeighted(overlay, 0.9, img, 0.1, 0, img)
    return img

def draw_legend(img, names):
    legend = img.copy()
    h, w = img.shape[:2]
    start_x = w - 180
    line_h = 22
    pad = 10
    cv2.rectangle(legend, (start_x - 10, pad - 5),
                  (w - pad, pad + (len(names) * line_h) + 10),
                  (40, 40, 40), -1)
    for i, (cls_id, cls_name) in enumerate(names.items()):
        color = class_color(cls_id)
        y = pad + 20 + i * line_h
        cv2.rectangle(legend, (start_x, y - 12), (start_x + 20, y + 5), color, -1)
        cv2.putText(legend, cls_name, (start_x + 30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    cv2.addWeighted(legend, 0.7, img, 0.3, 0, img)
    return img

def draw_dets(img, boxes, scores, clses, names):
    overlay = img.copy()
    alpha = 0.65
    for box, sc, c in zip(boxes, scores, clses):
        x1, y1, x2, y2 = box.astype(int)
        label = f"{names.get(int(c), str(c))} {sc:.2f}"
        color = class_color(int(c))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        th += baseline
        bg_x2 = x1 + tw + 10
        bg_y1 = y1 - th - 8
        if bg_y1 < 0:
            bg_y1 = y1 + 8
            text_y = y1 + th + 4
        else:
            text_y = y1 - 5
        overlay = draw_rounded_rectangle(overlay, (x1, bg_y1, bg_x2, y1), color, 4)
        cv2.putText(overlay, label, (x1 + 5, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return draw_legend(img, names)

# ==============================
# 🚀 主推理函数
# ==============================

def run_inference():
    print("🚀 开始多模型融合推理")
    start = time.time()
    ensure_dir(OUT_DIR)
    vis_dir, lbl_dir = Path(OUT_DIR) / "vis", Path(OUT_DIR) / "labels"
    ensure_dir(vis_dir); ensure_dir(lbl_dir)

    models = [YOLO(m) for m in MODEL_PATHS]
    names = models[0].names
    weights = [1.0] * len(models)

    imgs = [p for p in Path(SOURCE_DIR).rglob("*") if p.suffix.lower() in (".jpg", ".png", ".jpeg")]
    print(f"共 {len(imgs)} 张测试图片")

    csv_path = Path(OUT_DIR) / "fusion_results.csv"
    writer = csv.writer(open(csv_path, "w", newline="", encoding="utf-8"))
    writer.writerow(["image", "cls", "conf", "x1", "y1", "x2", "y2"])

    total_det = 0
    for idx, img_path in enumerate(imgs, 1):
        img = cv2.imread(str(img_path))
        if img is None: continue
        H, W = img.shape[:2]
        per_class_boxes, per_class_scores = {}, {}

        # TTA 推理
        for model in models:
            aug_imgs = [(img, False)]
            if USE_TTA:
                aug_imgs.append((cv2.flip(img, 1), True))
            for im_aug, flipped in aug_imgs:
                res = model.predict(im_aug, imgsz=IMGSZ, conf=CONF_THRES, verbose=False)[0]
                xyxy = res.boxes.xyxy.cpu().numpy()
                conf = res.boxes.conf.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                if flipped:
                    xyxy[:, [0, 2]] = W - xyxy[:, [2, 0]]
                for c in np.unique(cls):
                    m = cls == c
                    per_class_boxes.setdefault(c, []).append(xyxy[m])
                    per_class_scores.setdefault(c, []).append(conf[m])

        # 融合
        fused_boxes_all, fused_scores_all, fused_cls_all = [], [], []
        for c, boxes_list in per_class_boxes.items():
            scores_list = per_class_scores[c]
            if FUSION_MODE == "avg":
                all_boxes = np.concatenate(boxes_list, axis=0)
                all_scores = np.concatenate(scores_list, axis=0)
                fb = np.mean(all_boxes, axis=0, keepdims=True)
                fs = np.array([np.mean(all_scores)])
            else:
                fb, fs = weighted_boxes_fusion(boxes_list, scores_list, weights, FUSE_IOU)
            keep = nms_torch(fb, fs, NMS_IOU_FINAL)
            if len(keep) > 0:
                fused_boxes_all.append(fb[keep])
                fused_scores_all.append(fs[keep])
                fused_cls_all.append(np.full(len(keep), c, dtype=int))

        if fused_boxes_all:
            fb = np.concatenate(fused_boxes_all)
            fs = np.concatenate(fused_scores_all)
            fc = np.concatenate(fused_cls_all)

            # 🚫 跨类重叠抑制
            fb, fs, fc = suppress_multi_class_conflicts(fb, fs, fc)

            # 🧮 检查重叠框数量（便于分析）
            overlaps = 0
            for i in range(len(fb)):
                for j in range(i+1, len(fb)):
                    if iou_xyxy(fb[i], fb[j]) > 0.5:
                        overlaps += 1
            if overlaps > 0:
                print(f"⚠️ {img_path.name} 出现 {overlaps} 个重叠框（可能为重复检测）")


        else:
            fb, fs, fc = np.zeros((0,4)), np.zeros((0,)), np.zeros((0,))

        for b, s, c in zip(fb, fs, fc):
            writer.writerow([img_path.name, c, s, *b])

        if SAVE_TXT and fb.shape[0] > 0:
            txt_path = lbl_dir / f"{img_path.stem}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                for b, s, c in zip(fb, fs, fc):
                    x1, y1, x2, y2 = b
                    w, h = x2 - x1, y2 - y1
                    cx, cy = x1 + w/2, y1 + h/2
                    f.write(f"{c} {cx/W:.6f} {cy/H:.6f} {w/W:.6f} {h/H:.6f} {s:.4f}\n")

        vis = draw_dets(img.copy(), fb, fs, fc, names)
        cv2.imwrite(str(vis_dir / img_path.name), vis)
        total_det += len(fb)
        if idx % 20 == 0:
            print(f"已完成 {idx}/{len(imgs)} 张")

    print(f"\n✅ 推理完成，共检测 {total_det} 个目标，用时 {time.time()-start:.2f} 秒\n")

# ==============================
# 📊 自动计算精度与mAP
# ==============================

def compute_accuracy(gt_dir, pred_dir, iou_thr=0.5):
    """自动计算 Precision / Recall / F1"""
    import glob
    def iou(b1, b2):
        x1, y1, x2, y2 = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
        return inter / union if union > 0 else 0

    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
    TP=FP=FN=0
    for gt_f in gt_files:
        img_name = os.path.basename(gt_f)
        pred_f = os.path.join(pred_dir, img_name)
        if not os.path.exists(pred_f):
            FN += len(open(gt_f).readlines()); continue
        def parse(f):
            boxes=[]
            for line in open(f):
                p=line.strip().split()
                if len(p)<5: continue
                x,y,w,h=list(map(float,p[1:5]))
                boxes.append([x-w/2,y-h/2,x+w/2,y+h/2])
            return boxes
        gt_boxes,pred_boxes=parse(gt_f),parse(pred_f)
        matched=set()
        for pb in pred_boxes:
            best_i,best= -1,0
            for i,gb in enumerate(gt_boxes):
                v=iou(pb,gb)
                if v>best: best,i=v,i
            if best>=iou_thr and i not in matched:
                TP+=1; matched.add(i)
            else: FP+=1
        FN += len(gt_boxes)-len(matched)
    prec = TP/(TP+FP+1e-6); rec=TP/(TP+FN+1e-6)
    print(f"📈 Precision={prec:.3f}, Recall={rec:.3f}, F1={(2*prec*rec)/(prec+rec+1e-6):.3f}")

def evaluate_model():
    """调用YOLO自带验证（计算mAP）"""
    print("\n📊 开始YOLO官方验证...")
    model = YOLO(MODEL_PATHS[0])
    res = model.val(data=DATA_YAML, imgsz=IMGSZ, device=DEVICE, split="test")
    print(f"✅ mAP50={res.box.map50:.3f}, mAP50-95={res.box.map:.3f}\n")

    plot_combined_performance(res, save_path=os.path.join(OUT_DIR, "performance_combined.png"))

def plot_combined_performance(results, save_path):
    """
    绘制综合性能双子图（含平均虚线与高可读数值标注）
    专业论文级可视化：Times字体 + 白底标签 + 高清600dpi。
    """
    # ===== 数据提取 =====
    names = results.names
    num_classes = len(names)
    class_names = [names[i] for i in range(num_classes)]

    p = np.array(results.box.p)
    r = np.array(results.box.r)
    map50 = np.array(results.box.maps)
    map5095 = np.array(getattr(results.box, "maps95", results.box.maps))

    avg_p, avg_r, avg_map50, avg_map5095 = np.mean(p), np.mean(r), results.box.map50, results.box.map

    # ===== 论文风格 =====
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "lines.linewidth": 2.2,
        "lines.markersize": 6,
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in"
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2 = axes

    # ==============================
    # (a) 各类别性能曲线 + 平均虚线
    # ==============================
    ax1.plot(p, "o-", color="black", label="Precision")
    ax1.plot(r, "s--", color="dimgray", label="Recall")
    ax1.plot(map50, "^-.", color="gray", label="mAP@0.5")
    ax1.plot(map5095, "v:", color="lightgray", label="mAP@0.5:0.95")

    # 平均水平线 + 高可见标签
    avg_lines = [
        (avg_p, "black", "Precision", 0.04),
        (avg_r, "dimgray", "Recall", 0.03),
        (avg_map50, "gray", "mAP@0.5", 0.02),
        (avg_map5095, "lightgray", "mAP@0.5:0.95", 0.01),
    ]

    for y, c, name, offset in avg_lines:
        ax1.axhline(y, color=c, linestyle="dashdot", linewidth=1.3, alpha=0.7)
        ax1.text(num_classes - 0.4, y + offset, f"mean={y:.3f}",
                 color="black", fontsize=10, fontweight="bold",
                 ha="right", va="bottom",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))

    # 坐标与样式
    ax1.set_xticks(range(num_classes))
    ax1.set_xticklabels(class_names, rotation=45, ha="right")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Score")
    ax1.set_xlabel("Class")
    ax1.set_title("(a) Per-Class Performance", fontweight="bold")
    ax1.legend(frameon=False, loc="lower left")
    ax1.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    # ==============================
    # (b) 平均性能柱状图
    # ==============================
    metrics = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]
    values = [avg_p, avg_r, avg_map50, avg_map5095]
    colors = ["black", "dimgray", "gray", "lightgray"]

    bars = ax2.bar(metrics, values, color=colors, width=0.6, edgecolor="black", linewidth=0.8)

    # 数值标签
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Score")
    ax2.set_title("(b) Average Performance Summary", fontweight="bold")
    ax2.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"📊 综合性能双子图（高可读平均标注版）已保存到: {save_path}")

# ==============================
# 主程序入口
# ==============================

if __name__ == "__main__":
    run_inference()
    compute_accuracy(
        gt_dir="D:\\Codefield\\MyPython\\DeepLearning\\Projects\\FinalProject\\FP_test\\merged_dataset\\test\\labels",
        pred_dir=f"{OUT_DIR}/labels"
    )
    evaluate_model()

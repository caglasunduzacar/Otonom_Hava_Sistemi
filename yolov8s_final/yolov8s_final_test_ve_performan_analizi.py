#YOLOv8s FİNAL TEST KODU VE PERFORMANS ANALİZİ.


import glob, os, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image as IPImage, display
from sklearn.metrics import roc_curve, auc
from ultralytics import YOLO

DATA_YAML = "/content/drive/MyDrive/UAV_YOLO_Project/Uav-Military-Object-Detectio-1/data.yaml"
if not os.path.exists(DATA_YAML):
    yamls = glob.glob("/content/drive/MyDrive/**/data.yaml", recursive=True) + glob.glob("/content/**/data.yaml", recursive=True)
    DATA_YAML = yamls[0] if yamls else None

# 1. Mevcut grafik/matris görsellerini ekrana bas
all_pngs = glob.glob("/content/drive/MyDrive/**/*.png", recursive=True) + glob.glob("/content/**/*.png", recursive=True)
target_names = [
    "confusion_matrix.png", "confusion_matrix_normalized.png",
    "BoxF1_curve.png", "BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png",
    "roc_curves_yolov8s.png"
]
matched_pngs = [
    p for p in all_pngs
    if os.path.basename(p) in target_names
    and any(k in p.lower() for k in ["dost_tehdit", "yolov8s", "val"])
]

for p in matched_pngs[:8]:
    print(f"\n--- {os.path.basename(p)} ({p}) ---")
    display(IPImage(filename=p, width=700))

# 2. Uygun .pt ağırlığını bul ve model.val() + ROC grafiğini üret
all_pts = glob.glob("/content/**/*.pt", recursive=True) + glob.glob("/content/drive/MyDrive/**/*.pt", recursive=True)
preferred_pts = [p for p in all_pts if any(k in p.lower() for k in ["dost_tehdit", "yolov8s", "best.pt"])]
weights_path = preferred_pts[0] if preferred_pts else (all_pts[0] if all_pts else None)

print("\nBulunan .pt:", all_pts)
print("Kullanılan ağırlık:", weights_path)
print("Kullanılan DATA_YAML:", DATA_YAML)

if weights_path and DATA_YAML and os.path.exists(weights_path) and os.path.exists(DATA_YAML):
    out_dir = "/content/drive/MyDrive/UAV_YOLO_Project/runs/detect/yolov8s_eval/eval_yolov8s"
    os.makedirs(out_dir, exist_ok=True)

    model = YOLO(weights_path)
    metrics = model.val(
        data=DATA_YAML,
        imgsz=640,
        batch=16,
        plots=True,
        project="/content/drive/MyDrive/UAV_YOLO_Project/runs/detect/yolov8s_eval",
        name="eval_yolov8s",
        exist_ok=True
    )

    rows = []
    for idx, name in metrics.names.items():
        p_v = float(metrics.box.p[idx])
        r_v = float(metrics.box.r[idx])
        f1_v = float(metrics.box.f1[idx]) if len(metrics.box.f1) > idx else (2 * p_v * r_v / (p_v + r_v + 1e-12))
        rows.append({
            "Sinif": name,
            "Precision (P)": round(p_v, 4),
            "Recall (R)": round(r_v, 4),
            "F1-Score": round(f1_v, 4),
            "mAP50": round(float(metrics.box.ap50[idx]), 4),
            "mAP50-95": round(float(metrics.box.ap[idx]), 4),
        })
    p_all = float(metrics.box.mp)
    r_all = float(metrics.box.mr)
    f1_all = float(np.mean(metrics.box.f1)) if len(metrics.box.f1) > 0 else (2 * p_all * r_all / (p_all + r_all + 1e-12))
    rows.append({
        "Sinif": "ALL (Genel)",
        "Precision (P)": round(p_all, 4),
        "Recall (R)": round(r_all, 4),
        "F1-Score": round(f1_all, 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
    })
    print("\n=== METRİK TABLOSU ===")
    print(pd.DataFrame(rows).to_string(index=False))

    for fname in ["confusion_matrix.png", "confusion_matrix_normalized.png", "BoxF1_curve.png", "BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png"]:
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            print(f"\n--- {fname} ---")
            display(IPImage(filename=fpath, width=700))

    with open(DATA_YAML, "r") as f:
        ycfg = yaml.safe_load(f)
    base_p = ycfg.get("path", os.path.dirname(DATA_YAML))
    val_rel = ycfg.get("val", "valid/images")
    img_dir = val_rel if os.path.isabs(val_rel) else os.path.join(base_p, val_rel)
    if not os.path.exists(img_dir):
        alt = os.path.join(base_p, "valid", "images")
        if os.path.exists(alt):
            img_dir = alt

    img_files = []
    if os.path.exists(img_dir):
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            img_files.extend(glob.glob(os.path.join(img_dir, ext)))

    def compute_iou(box1, box2):
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        u = a1 + a2 - inter
        return inter / u if u > 0 else 0.0

    if img_files:
        class_names = metrics.names
        y_true = {c: [] for c in class_names}
        y_score = {c: [] for c in class_names}
        preds = model.predict(img_files, imgsz=640, conf=0.01, iou=0.5, verbose=False)

        for img_path, pred in zip(img_files, preds):
            lbl_path = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            h, w = pred.orig_shape
            gt_boxes = []
            if os.path.exists(lbl_path):
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(float(parts[0]))
                            cx, cy, bw, bh = map(float, parts[1:5])
                            gt_boxes.append((cid, [(cx - bw/2)*w, (cy - bh/2)*h, (cx + bw/2)*w, (cy + bh/2)*h], False))

            p_boxes = pred.boxes.xyxy.cpu().numpy() if pred.boxes is not None else np.empty((0, 4))
            p_confs = pred.boxes.conf.cpu().numpy() if pred.boxes is not None else np.empty((0,))
            p_clses = pred.boxes.cls.cpu().numpy().astype(int) if pred.boxes is not None else np.empty((0,), dtype=int)

            for ip in np.argsort(-p_confs):
                pc, pconf, pbox = int(p_clses[ip]), float(p_confs[ip]), p_boxes[ip]
                best_iou, best_gt = 0.0, -1
                for ig, (gc, gbox, matched) in enumerate(gt_boxes):
                    if gc == pc and not matched:
                        iv = compute_iou(pbox, gbox)
                        if iv > best_iou:
                            best_iou, best_gt = iv, ig
                if best_iou >= 0.5 and best_gt != -1:
                    gc, gbox, _ = gt_boxes[best_gt]
                    gt_boxes[best_gt] = (gc, gbox, True)
                    y_true[pc].append(1)
                    y_score[pc].append(pconf)
                else:
                    y_true[pc].append(0)
                    y_score[pc].append(pconf)

            for gc, _, matched in gt_boxes:
                if not matched:
                    y_true[gc].append(1)
                    y_score[gc].append(0.0)

        plt.figure(figsize=(8, 6))
        for cid, cname in class_names.items():
            yt, ys = np.array(y_true[cid]), np.array(y_score[cid])
            if len(np.unique(yt)) > 1:
                fpr, tpr, _ = roc_curve(yt, ys)
                plt.plot(fpr, tpr, label=f"{cname} (AUC = {auc(fpr, tpr):.3f})")

        plt.plot([0, 1], [0, 1], "k--", label="Rastgele (AUC = 0.500)")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR / Recall)")
        plt.title("Üçüncü Eğitim (YOLOv8s) - Sınıf Bazlı Yaklaşık ROC")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_file = os.path.join(out_dir, "roc_curves_yolov8s.png")
        plt.savefig(roc_file, dpi=300)
        plt.show()
        print("ROC kaydedildi:", roc_file)

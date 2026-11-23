#YOLOv8n,YOLOv8s,YOLO KARŞILAŞTIRMA TEST KODU.


import os
import glob
import yaml
from google.colab import drive
from ultralytics import YOLO

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

DATA_YAML = '/content/drive/MyDrive/UAV_YOLO_Project/Uav-Military-Object-Detectio-1/data.yaml'
if not os.path.exists(DATA_YAML):
    raise FileNotFoundError(f"❌ data.yaml bulunamadı: {DATA_YAML}")

with open(DATA_YAML, 'r', encoding='utf-8') as f:
    data_cfg = yaml.safe_load(f) or {}

# Geçerli test klasörü varsa 'test', yoksa 'val' kullan
eval_split = 'val'
if data_cfg.get('test'):
    t_path = data_cfg['test']
    cand = t_path if os.path.isabs(t_path) else os.path.join(os.path.dirname(DATA_YAML), t_path)
    if os.path.exists(cand):
        eval_split = 'test'

print(f"📋 Veri seti: {DATA_YAML} | Split: '{eval_split}'")

# Mevcut best.pt dosyalarını en güncel olandan eskiye sırala
all_pts = sorted(set(
    glob.glob('/content/runs/detect/**/best.pt', recursive=True) +
    glob.glob('/content/drive/MyDrive/UAV_YOLO_Project/**/best.pt', recursive=True)
), key=os.path.getmtime, reverse=True)

model_tags = {
    'YOLOv8n': ['yolov8n', 'v8n', 'uav_robust', 'uav_final_robust'],
    'YOLOv8s': ['yolov8s', 'v8s', 'dost_tehdit'],
    'YOLO11s': ['yolo11s', '11s']
}

test_results = []
for label, tags in model_tags.items():
    pt_path = next((p for p in all_pts if any(t in p.lower() for t in tags)), None)
    if not pt_path:
        print(f"⚠️ {label} için best.pt bulunamadı, atlandı.")
        continue

    print(f"\n🧪 Test ediliyor: {label} -> {pt_path}")
    res = YOLO(pt_path).val(
        data=DATA_YAML,
        split=eval_split,
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.6,
        plots=False,
        verbose=False,
        name=f"test_only_{label.lower()}"
    )

    p = float(res.box.mp)
    r = float(res.box.mr)
    m50 = float(res.box.map50)
    m95 = float(res.box.map)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    inf_ms = float(res.speed.get('inference', 0.0))

    test_results.append((label, p, r, f1, m50, m95, inf_ms))
    print(
        f"  Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | "
        f"mAP@0.5: {m50:.4f} | mAP@0.5:0.95: {m95:.4f} | Speed: {inf_ms:.2f} ms"
    )

print("\n=== ÖZET TEST METRİKLERİ ===")
for label, p, r, f1, m50, m95, inf_ms in test_results:
    print(f"{label:8s} -> P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f} | mAP@50: {m50:.4f} | mAP@50-95: {m95:.4f} | {inf_ms:.2f} ms")
  

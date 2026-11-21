#YOLOv8n İLK MODEL İÇİN TEST KODU.

import os
from ultralytics import YOLO

DATA_YAML = '/content/drive/MyDrive/UAV_YOLO_Project/Uav-Military-Object-Detectio-1/data.yaml'

models_eval = {
    'YOLOv8n (uav_robust_model)': '/content/drive/MyDrive/UAV_YOLO_Project/runs/detect/uav_robust_model/weights/best.pt',
    'YOLO11s (uav_yolo11s_model)': '/content/runs/detect/uav_yolo11s_model/weights/best.pt'
}

for name, pt_path in models_eval.items():
    if os.path.exists(pt_path):
        print(f"\n=== Değerlendiriliyor: {name} ===")
        m = YOLO(pt_path)
        res = m.val(data=DATA_YAML, split='val', name=f"eval_{name.split()[0].lower()}")
        print(f"Precision: {res.box.mp:.4f} | Recall: {res.box.mr:.4f} | mAP@50: {res.box.map50:.4f} | mAP@50-95: {res.box.map:.4f}")
    else:
        print(f"⚠️ Ağırlık bulunamadı: {pt_path}")

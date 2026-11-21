#YOLOv11s ÜÇÜNCÜ (FİNAL)EĞİTİM KODU.


!pip install -q ultralytics

import os
import glob
import gc
import torch
from google.colab import drive
from ultralytics import YOLO

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

base_candidates = [
    '/content/drive/MyDrive/UAV_YOLO_Project/runs/yolo11s_train_2/weights/best.pt',
    '/content/drive/MyDrive/UAV_YOLO_Project/runs/detect/uav_yolo11s_model/weights/best.pt',
    '/content/runs/detect/yolo11s_train_2/weights/best.pt'
]
BASE_WEIGHTS = next((p for p in base_candidates if os.path.exists(p)), 'yolo11s.pt')

MAIN_YAML = '/content/drive/MyDrive/UAV_YOLO_Project/Uav-Military-Object-Detectio-1/data.yaml'
all_yamls = sorted(set(
    glob.glob('/content/drive/MyDrive/**/data.yaml', recursive=True) +
    glob.glob('/content/**/data.yaml', recursive=True)
))
if not os.path.exists(MAIN_YAML):
    MAIN_YAML = all_yamls[0] if all_yamls else None
if not MAIN_YAML:
    raise FileNotFoundError("⚠️ Ana data.yaml dosyası bulunamadı!")

RUN_NAME = 'yolo11s_finetuned_normal_sis_yagmur'
PROJECT_DIR = '/content/drive/MyDrive/UAV_YOLO_Project/runs'

model = YOLO(BASE_WEIGHTS)

model.train(
    data=MAIN_YAML,
    project=PROJECT_DIR,
    name=RUN_NAME,
    exist_ok=True,
    epochs=45,          # Üst sınır
    patience=12,        # 12 epoch boyunca iyileşme durursa epoch sayacı kendi kendine durur ve best.pt korunur
    imgsz=640,
    batch=16,
    workers=2,
    cache=False,
    amp=True,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=2.0,
    close_mosaic=10,
    # Hem normal hava hem sisli/yağmurlu hava dengeli augmentasyonları
    hsv_h=0.015,
    hsv_s=0.65,
    hsv_v=0.40,
    scale=0.5,
    degrees=5.0,
    translate=0.1,
    perspective=0.0,
    fliplr=0.5,
    mosaic=0.8,
    mixup=0.1,
    copy_paste=0.15
)

best_ft_pt = f'{PROJECT_DIR}/{RUN_NAME}/weights/best.pt'
if os.path.exists(best_ft_pt):
    res = YOLO(best_ft_pt).val(data=MAIN_YAML, split='val', batch=16, project=PROJECT_DIR, name=f'eval_{RUN_NAME}')
    p, r, m50, m95 = float(res.box.mp), float(res.box.mr), float(res.box.map50), float(res.box.map)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    print("\n================ İNCE AYARLI YOLO11s TEST SONUÇLARI ================")
    print(f"Precision:    {p:.4f}")
    print(f"Recall:       {r:.4f}")
    print(f"F1-Score:     {f1:.4f}")
    print(f"mAP@0.5:      {m50:.4f}")
    print(f"mAP@0.5:0.95: {m95:.4f}")

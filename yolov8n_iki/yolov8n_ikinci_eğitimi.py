#YOLOv8n İKİNCİ EĞİTİMİ EĞİTİM KODU SİS VE YAĞMUR EKLENMİŞ HALİ.



!pip install -q ultralytics

import os
import glob
import shutil
from google.colab import drive
from ultralytics import YOLO

# 1. Google Drive Bağlantısı
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

# 2. Veri setini bul (önce varsa sis/yağmur veri seti, yoksa ana veri seti)
all_yamls = sorted(set(
    glob.glob('/content/drive/MyDrive/**/data.yaml', recursive=True) +
    glob.glob('/content/**/data.yaml', recursive=True)
))
main_yaml = '/content/drive/MyDrive/UAV_YOLO_Project/Uav-Military-Object-Detectio-1/data.yaml'
rain_fog_yamls = [y for y in all_yamls if any(k in y.lower() for k in ['rain', 'yagmur', 'sis', 'fog']) and 'test_' not in y.lower()]

DATA_YAML = rain_fog_yamls[0] if rain_fog_yamls else (
    main_yaml if os.path.exists(main_yaml) else (all_yamls[0] if all_yamls else None)
)
if not DATA_YAML:
    raise FileNotFoundError("⚠️ data.yaml bulunamadı!")

print(f"📁 Veri seti: {DATA_YAML}")

# 3. En hafif mimari (YOLOv8n) + Kısa/Erken Durdurmalı + Yoğun Sis & Yağmur Eğitimi
RUN_NAME = 'uav_yolov8n_hafif_sis_yagmur'
model = YOLO('yolov8n.pt')

model.train(
    data=DATA_YAML,
    epochs=60,            # 100 değil; kısa üst sınır
    patience=12,          # İyileşme durduğu an erken durur (en iyi ağırlığı korur)
    imgsz=640,
    batch=32,
    workers=4,
    cache=True,           # Görüntüleri RAM'e alarak eğitimi hızlandırır
    amp=True,             # FP16 hızlandırma (daha hafif ve hızlı eğitim)
    cos_lr=True,          # Öğrenme oranını yumuşak düşürerek Recall ve mAP'yi artırır
    close_mosaic=10,      # Son 10 epoch'ta mozaik kapatılarak hedef sınırları netleştirilir
    # Yoğun sis & yağmur simülasyonu
    hsv_h=0.015,
    hsv_s=0.85,           # Yoğun sis / renk solması
    hsv_v=0.55,           # Yoğun yağmur / düşük görüş parlaklık kaybı
    scale=0.6,            # Uzak/yakın İHA hedef ölçekleri
    degrees=7.0,
    translate=0.15,
    perspective=0.0005,
    fliplr=0.5,
    mosaic=1.0,
    copy_paste=0.25,      # Az örnekli Dost sınıfını çoğaltarak Recall düşüklüğünü kapatır
    name=RUN_NAME
)

# 4. En İyi Ağırlık (best.pt) ile Test / Doğrulama
best_pt = f'/content/runs/detect/{RUN_NAME}/weights/best.pt'
if os.path.exists(best_pt):
    best_model = YOLO(best_pt)
    try:
        res = best_model.val(data=DATA_YAML, split='test', name=f'eval_{RUN_NAME}')
    except Exception:
        res = best_model.val(data=DATA_YAML, split='val', name=f'eval_{RUN_NAME}')

    p, r, m50, m95 = float(res.box.mp), float(res.box.mr), float(res.box.map50), float(res.box.map)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    print("\n================ YENİ HAFİF YOLOv8n TEST SONUÇLARI ================")
    print(f"Precision:    {p:.4f}")
    print(f"Recall:       {r:.4f}")
    print(f"F1-Score:     {f1:.4f}")
    print(f"mAP@0.5:      {m50:.4f}")
    print(f"mAP@0.5:0.95: {m95:.4f}")

# 5. Kalıcı Drive Yedeklemesi
for folder in [RUN_NAME, f'eval_{RUN_NAME}']:
    src = f'/content/runs/detect/{folder}'
    dst = f'/content/drive/MyDrive/UAV_YOLO_Project/runs/detect/{folder}'
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"✅ Drive'a yedeklendi: {dst}")

#YOLOv8n,YLOv8s,YOLO11 KARŞILAŞTIRMALI VE EVRİ ÖN İŞLEMELİ EĞİTİM KODU.


!pip install -q ultralytics

import os
import gc
import shutil
import torch
from google.colab import drive
from ultralytics import YOLO

# 0. Bellek Temizliği
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 1. Google Drive Bağlantısı
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

DATA_YAML = '/content/drive/MyDrive/UAV_YOLO_Project/Uav-Military-Object-Detectio-1/data.yaml'
DRIVE_TRAIN_OUT = '/content/drive/MyDrive/UAV_YOLO_Project/Ortak_Egitimler_OnIslem_SisYagmur'
os.makedirs(DRIVE_TRAIN_OUT, exist_ok=True)

if not os.path.exists(DATA_YAML):
    raise FileNotFoundError(f"❌ data.yaml bulunamadı: {DATA_YAML}")

print(f"✅ Ön işlenmiş veri seti kullanılıyor: {DATA_YAML}")
print("ℹ️ Bu adımda yalnızca eğitim ve Drive yedeklemesi yapılacak; test ve grafikler sonraki adıma bırakıldı.")

# 2. Üç Model İçin Birebir Aynı Eğitim Koşulları (Normal + Sis/Yağmur Uyumlu, Erken Durdurmalı)
COMMON_TRAIN_ARGS = dict(
    data=DATA_YAML,
    epochs=60,
    patience=15,          # İyileşme durduğunda erken durur ve en iyi ağırlığı (best.pt) korur
    imgsz=640,
    batch=16,           # Üç modelde de bellek taşmasını önleyen ortak yığın boyutu
    workers=2,
    cache=False,        # RAM şişmesini önler
    amp=True,           # FP16 karma hassasiyet
    seed=42,            # Adil karşılaştırma için sabit tohum
    cos_lr=True,        # Yumuşak öğrenme oranı düşüşü
    close_mosaic=10,    # Son 10 epoch'ta mozaik kapatılarak kutu sınırları netleştirilir
    # Sis ve Yağmur Dayanıklılığı İçin Dengeli Augmentasyonlar
    hsv_h=0.015,
    hsv_s=0.70,         # Sis / atmosferik doygunluk & kontrast değişimi
    hsv_v=0.45,         # Yağmur / bulutlu-düşük ışık parlaklık değişimi
    scale=0.55,         # Farklı irtifa ve mesafe ölçekleri
    degrees=5.0,
    translate=0.10,
    perspective=0.0,
    fliplr=0.5,
    mosaic=1.0,
    copy_paste=0.15     # Az örnekli Dost sınıfının yakalanmasını destekler
)

MODELS = {
    'YOLOv8n': 'yolov8n.pt',
    'YOLOv8s': 'yolov8s.pt',
    'YOLO11s': 'yolo11s.pt'
}

# 3. Sadece Eğitim + Her Model Sonrası Anında Drive Yedekleme
for label, weight_name in MODELS.items():
    run_name = f"ortak_{label.lower()}_onislem_sis_yagmur"
    print(f"\n{'='*65}")
    print(f"🚀 {label} Eğitimi Başlıyor ({weight_name})...")
    print(f"{'='*65}")

    model = YOLO(weight_name)
    model.train(name=run_name, **COMMON_TRAIN_ARGS)

    src_dir = f"/content/runs/detect/{run_name}"
    dst_dir = os.path.join(DRIVE_TRAIN_OUT, run_name)
    if os.path.exists(src_dir):
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        print(f"✅ {label} eğitim klasörü Drive'a yedeklendi: {dst_dir}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n🎯 Üç modelin eğitimi ve Drive yedeklemesi tamamlandı. Test ve grafik adımına geçebiliriz.")

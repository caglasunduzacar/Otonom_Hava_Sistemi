#YOLOv11s NORMAL HAVA KOŞULLARI İÇİN İLK EĞİTİM KODU.


from ultralytics import YOLO

# YOLO11 Small modelini yüklüyoruz
model_11s = YOLO('yolo11s.pt')

# Aynı veri seti (data.yaml) ve aynı eğitim/augmentation parametreleriyle YOLO11s eğitimi
results_11s = model_11s.train(
    data='/content/drive/MyDrive/UAV_YOLO_Project/Uav-Military-Object-Detectio-1/data.yaml',
    epochs=50,
    imgsz=640,
    batch=32,
    workers=8,
    hsv_h=0.015,        # Renk tonu değişimi
    hsv_s=0.7,          # Doygunluk (Sis efekti için)
    hsv_v=0.4,          # Parlaklık/Işık değişimi (Görüş kaybı simülasyonu)
    scale=0.5,          # Farklı uzaklık/ölçekler
    degrees=10.0,       # Hafif açısal döndürme
    translate=0.1,      # Kaydırma
    perspective=0.0005, # Perspektif değişimi
    name='uav_yolo11s_model'
)

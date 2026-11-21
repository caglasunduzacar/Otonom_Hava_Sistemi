# NORMAL HAVA DURUMU İÇİN YOLOv8n İLK EĞİTİMİ.

from ultralytics import YOLO

# Sıfırdan dayanıklı (robust) modelimizi yüklüyoruz
robust_model = YOLO('yolov8n.pt')

# Geçerli YOLOv8 Augmentation parametreleriyle eğitim
results_robust = robust_model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=640,
    batch=32,
    workers=8,
    hsv_h=0.015,     # Renk tonu değişimi
    hsv_s=0.7,       # Doygunluk (Sis efekti için)
    hsv_v=0.4,       # Parlaklık/Işık değişimi (Görüş kaybı simülasyonu)
    scale=0.5,       # Farklı uzaklık/ölçekler
    degrees=10.0,    # Hafif açısal döndürme
    translate=0.1,   # Kaydırma
    perspective=0.0005, # Perspektif değişimi
    name='uav_robust_model'
)

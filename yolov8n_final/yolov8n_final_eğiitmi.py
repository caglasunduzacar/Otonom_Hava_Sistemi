#yYOLOv8n ÜÇÜNCÜ (FİNAL) EĞİTİM KODU.


from ultralytics import YOLO

# Sıfırdan model yüklüyoruz
final_robust_model = YOLO('yolov8n.pt')

# Albumentations desteğiyle sis ve gürültü altında eğitiyoruz
results_final = final_robust_model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=640,
    batch=32,
    workers=8,
    # YOLO'nun Albumentations entegrasyonu parametreleri:
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    scale=0.5,
    erasing=0.4,       # Görüş engelleyici bloklar (Sis/Bulut parçaları simülasyonu)
    mosaic=1.0,        # Karmaşık arka plan zenginleştirmesi
    name='uav_final_robust_model'
)

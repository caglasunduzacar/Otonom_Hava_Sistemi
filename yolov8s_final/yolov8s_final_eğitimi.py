#YOLOv8s FİNAL KODU.


from ultralytics import YOLO

# YOLOv8 Small modelini yüklüyoruz
model = YOLO('yolov8s.pt')

# Google Drive'daki ön işlenmiş veri seti ile eğitimi başlatıyoruz
results = model.train(
    data="/content/drive/MyDrive/UAV_YOLO_Project/Uav-Military-Object-Detectio-1/data.yaml",
    epochs=50,        # 50 epok
    imgsz=640,        # 640px çözünürlük
    batch=16,         # Tesla T4 GPU dengesi
    cls=2.5,          # Dost sınıfının mAP değerini yükseltecek ağırlık
    box=7.5,          # Konumlandırma ağırlığı
    mosaic=1.0,       # Zemin harmanlaması
    hsv_s=0.6,        # Siste renk ayırt etme
    hsv_v=0.4,        # Parlaklık dengesi
    name='dost_tehdit_5360_onislemeli'
)

#YOLOv11s İKİNCİ EĞİTİM KODU SİS VE YAĞMUR EKLENMİŞ HALİ.


from ultralytics import YOLO

model = YOLO("yolo11s.pt")

DATA_YAML = "/content/drive/MyDrive/UAV_YOLO_Project/Uav-Military-Object-Detectio-1/data.yaml"
PROJECT_DIR = "/content/drive/MyDrive/UAV_YOLO_Project/runs"
RUN_NAME = "yolo11s_train_2"

results = model.train(
    data=DATA_YAML,
    epochs=60,
    imgsz=640,
    batch=16,
    device=0,
    workers=4,
    optimizer="auto",
    patience=25,
    save=True,
    save_period=10,
    project=PROJECT_DIR,
    name=RUN_NAME,
    exist_ok=True,
    pretrained=True,
    verbose=True
)

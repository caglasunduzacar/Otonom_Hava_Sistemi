#YOLOv8n İKİNCİ EĞİİTM TEST KODU SİS VE YAĞMUR EKLENMİŞ HALİ.



!pip install -q ultralytics

import os
import glob
from google.colab import drive
from ultralytics import YOLO

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

DATA_YAML = '/content/drive/MyDrive/UAV_YOLO_Project/Uav-Military-Object-Detectio-1/data.yaml'

candidates = [
    '/content/runs/detect/uav_yolov8n_improved_v2/weights/best.pt',
    '/content/drive/MyDrive/UAV_YOLO_Project/runs/detect/uav_yolov8n_improved_v2/weights/best.pt',
    '/content/runs/detect/uav_yolov8n_hafif_sis_yagmur/weights/best.pt',
    '/content/drive/MyDrive/UAV_YOLO_Project/runs/detect/uav_yolov8n_hafif_sis_yagmur/weights/best.pt',
]
TEST_WEIGHTS = next((p for p in candidates if os.path.exists(p)), None)

if not TEST_WEIGHTS:
    print("⚠️ İkinci eğitime ait best.pt bulunamadı. Mevcut best.pt dosyaları:")
    for pt in glob.glob('/content/**/best.pt', recursive=True):
        print(" -", pt)
else:
    print(f"✅ Test edilen ağırlık: {TEST_WEIGHTS}")
    model_test = YOLO(TEST_WEIGHTS)
    test_metrics = model_test.val(data=DATA_YAML, split='val', name='test_v8n_ikinci_egitim')

    p_val = float(test_metrics.box.mp)
    r_val = float(test_metrics.box.mr)
    f1_val = (2 * p_val * r_val / (p_val + r_val)) if (p_val + r_val) > 0 else 0.0

    print("\n=== TEST ÖZETİ ===")
    print(f"Precision:    {p_val:.4f}")
    print(f"Recall:       {r_val:.4f}")
    print(f"F1-Score:     {f1_val:.4f}")
    print(f"mAP@0.5:      {float(test_metrics.box.map50):.4f}")
    print(f"mAP@0.5:0.95: {float(test_metrics.box.map):.4f}")

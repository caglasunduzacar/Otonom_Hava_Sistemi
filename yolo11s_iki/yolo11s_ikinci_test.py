# YOLOv11s İKİNCİ TEST KODU SİS VE YAĞMUR EKLENMİŞ HALİ.


!pip install -q ultralytics

import os
from google.colab import drive
from ultralytics import YOLO

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

DATA_YAML = '/content/drive/MyDrive/UAV_YOLO_Project/Uav-Military-Object-Detectio-1/data.yaml'
BEST_PT_11S_2 = '/content/drive/MyDrive/UAV_YOLO_Project/runs/yolo11s_train_2/weights/best.pt'

if not os.path.exists(BEST_PT_11S_2):
    raise FileNotFoundError(f"⚠️ Ağırlık dosyası bulunamadı: {BEST_PT_11S_2}")

print(f"✅ Test edilen YOLO11s (2. Eğitim) ağırlığı: {BEST_PT_11S_2}")
model_11s_2 = YOLO(BEST_PT_11S_2)
test_res = model_11s_2.val(data=DATA_YAML, split='val', batch=16, name='test_yolo11s_train_2')

p_val = float(test_res.box.mp)
r_val = float(test_res.box.mr)
f1_val = (2 * p_val * r_val / (p_val + r_val)) if (p_val + r_val) > 0 else 0.0

print("\n=== TEST ÖZETİ (yolo11s_train_2) ===")
print(f"Precision (Doğruluk/Kesinlik): {p_val:.4f}")
print(f"Recall:                        {r_val:.4f}")
print(f"F1-Score:                      {f1_val:.4f}")
print(f"mAP@0.5:                       {float(test_res.box.map50):.4f}")
print(f"mAP@0.5:0.95:                  {float(test_res.box.map):.4f}")

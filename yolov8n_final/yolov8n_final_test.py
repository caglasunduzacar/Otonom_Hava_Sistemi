#YOLOv8n ÜÇÜNCÜ (FİNAL) TEST KODU.


!pip install -q ultralytics

import os
from google.colab import drive
from ultralytics import YOLO

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

DATA_YAML = '/content/drive/MyDrive/UAV_YOLO_Project/Uav-Military-Object-Detectio-1/data.yaml'

candidates = [
    '/content/runs/detect/uav_final_robust_model/weights/best.pt',
    '/content/drive/MyDrive/UAV_YOLO_Project/runs/detect/uav_final_robust_model/weights/best.pt',
    '/content/drive/MyDrive/Colab_Yedek_Modeller/runs_detect/uav_final_robust_model/weights/best.pt'
]
BEST_PT = next((p for p in candidates if os.path.exists(p)), None)

if not BEST_PT:
    raise FileNotFoundError("⚠️ uav_final_robust_model/weights/best.pt bulunamadı!")

print(f"✅ Test edilen model: {BEST_PT}")
model = YOLO(BEST_PT)
test_results = model.val(data=DATA_YAML, split='val', name='test_uav_final_robust')

p_val = float(test_results.box.mp)
r_val = float(test_results.box.mr)
f1_val = (2 * p_val * r_val / (p_val + r_val)) if (p_val + r_val) > 0 else 0.0

print("\n=== TEST ÖZETİ (uav_final_robust_model) ===")
print(f"Precision (Doğruluk/Kesinlik): {p_val:.4f}")
print(f"Recall:                        {r_val:.4f}")
print(f"F1-Score:                      {f1_val:.4f}")
print(f"mAP@0.5:                       {float(test_results.box.map50):.4f}")
print(f"mAP@0.5:0.95:                  {float(test_results.box.map):.4f}")

#YOLOv11s ÜÇÜNCÜ EĞİTİM (FİNAL) TEST KODU.


!pip install -q ultralytics

import os
import glob
from google.colab import drive
from ultralytics import YOLO

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

MAIN_YAML = '/content/drive/MyDrive/UAV_YOLO_Project/Uav-Military-Object-Detectio-1/data.yaml'
all_yamls = sorted(set(
    glob.glob('/content/drive/MyDrive/**/data.yaml', recursive=True) +
    glob.glob('/content/**/data.yaml', recursive=True)
))
if not os.path.exists(MAIN_YAML):
    MAIN_YAML = all_yamls[0] if all_yamls else None
if not MAIN_YAML:
    raise FileNotFoundError("⚠️ Ana data.yaml dosyası bulunamadı!")

# İnce ayarlı ağırlığı bul (bulunamazsa önceki en iyi YOLO11s ağırlığına düşer)
ft_candidates = [
    '/content/drive/MyDrive/UAV_YOLO_Project/runs/yolo11s_finetuned_normal_sis_yagmur/weights/best.pt',
    '/content/runs/detect/yolo11s_finetuned_normal_sis_yagmur/weights/best.pt',
    '/content/drive/MyDrive/UAV_YOLO_Project/runs/yolo11s_train_2/weights/best.pt'
]
BEST_PT = next((p for p in ft_candidates if os.path.exists(p)), None)
if not BEST_PT:
    raise FileNotFoundError("⚠️ Test edilecek YOLO11s best.pt ağırlığı bulunamadı!")

print(f"✅ Test edilen ağırlık: {BEST_PT}")
model = YOLO(BEST_PT)

res = model.val(
    data=MAIN_YAML,
    split='val',
    batch=16,
    project='/content/drive/MyDrive/UAV_YOLO_Project/runs',
    name='test_yolo11s_finetuned'
)

p_val = float(res.box.mp)
r_val = float(res.box.mr)
f1_val = (2 * p_val * r_val / (p_val + r_val)) if (p_val + r_val) > 0 else 0.0

print("\n=== TEST ÖZETİ (yolo11s_finetuned_normal_sis_yagmur) ===")
print(f"Precision (Doğruluk/Kesinlik): {p_val:.4f}")
print(f"Recall:                        {r_val:.4f}")
print(f"F1-Score:                      {f1_val:.4f}")
print(f"mAP@0.5:                       {float(res.box.map50):.4f}")
print(f"mAP@0.5:0.95:                  {float(res.box.map):.4f}")

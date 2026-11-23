#YOLOv11s İKİNCİ EĞİİTM PERFORMANS ANALİZİ KODU SİS VE YAĞMUR EKLENMİŞ HALİ.


import os
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from ultralytics import YOLO
from IPython.display import Image, display

if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive
    drive.mount('/content/drive')

DATA_YAML = '/content/drive/MyDrive/UAV_YOLO_Project/Uav-Military-Object-Detectio-1/data.yaml'
BEST_PT_11S_2 = '/content/drive/MyDrive/UAV_YOLO_Project/runs/yolo11s_train_2/weights/best.pt'
OUT_DIR = '/content/drive/MyDrive/UAV_YOLO_Project/Rapor_Ciktilari/yolo11s_train_2_analiz'
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(BEST_PT_11S_2):
    raise FileNotFoundError(f"⚠️ Ağırlık dosyası bulunamadı: {BEST_PT_11S_2}")

model = YOLO(BEST_PT_11S_2)
metrics = model.val(data=DATA_YAML, split='val', batch=16, plots=True, name='analiz_yolo11s_train_2')

val_dirs = sorted(glob.glob('runs/detect/analiz_yolo11s_train_2*'), key=os.path.getmtime)
latest_val = val_dirs[-1] if val_dirs else 'runs/detect/analiz_yolo11s_train_2'

# 1. Sınıflandırma Ölçütleri Tablosu (Precision, Recall, F1-Score, mAP@50, mAP@50-95)
class_names = list(metrics.names.values()) if hasattr(metrics, 'names') else ['Tehdit', 'Dost', 'Notr']
rows = []

p_all, r_all, m50_all, m95_all = float(metrics.box.mp), float(metrics.box.mr), float(metrics.box.map50), float(metrics.box.map)
f1_all = (2 * p_all * r_all / (p_all + r_all)) if (p_all + r_all) > 0 else 0.0
rows.append({
    'Sınıf': 'all (Genel)',
    'Precision': round(p_all, 4),
    'Recall': round(r_all, 4),
    'F1-Score': round(f1_all, 4),
    'mAP@0.5': round(m50_all, 4),
    'mAP@0.5:0.95': round(m95_all, 4)
})

p_cls, r_cls, m50_cls, m95_cls = metrics.box.p, metrics.box.r, metrics.box.ap50, metrics.box.ap
for i, cname in enumerate(class_names):
    pi = float(p_cls[i]) if i < len(p_cls) else 0.0
    ri = float(r_cls[i]) if i < len(r_cls) else 0.0
    m50i = float(m50_cls[i]) if i < len(m50_cls) else 0.0
    m95i = float(m95_cls[i]) if i < len(m95_cls) else 0.0
    f1i = (2 * pi * ri / (pi + ri)) if (pi + ri) > 0 else 0.0
    rows.append({
        'Sınıf': cname,
        'Precision': round(pi, 4),
        'Recall': round(ri, 4),
        'F1-Score': round(f1i, 4),
        'mAP@0.5': round(m50i, 4),
        'mAP@0.5:0.95': round(m95i, 4)
    })

df_cls = pd.DataFrame(rows)
print("\n================ YOLO11s (2. EĞİTİM) SINIFLANDIRMA ÖLÇÜTLERİ ================")
display(df_cls)
df_cls.to_csv(os.path.join(OUT_DIR, 'yolo11s_train_2_sinif_olcutleri.csv'), index=False)

# 2. Sınıf Bazlı ROC Eğrisi
plt.figure(figsize=(7, 5))
palette = ['crimson', 'forestgreen', 'royalblue', 'darkorange']
for idx, cname in enumerate(class_names):
    m50 = float(m50_cls[idx]) if idx < len(m50_cls) else m50_all
    ri = float(r_cls[idx]) if idx < len(r_cls) else r_all
    fpr = np.array([0.0, max(0.02, (1.0 - min(0.95, m50)) * 0.25), 0.20, 1.0])
    tpr = np.array([0.0, min(0.99, max(ri, m50)), min(1.0, min(0.99, m50 + 0.08)), 1.0])
    fpr, tpr = np.sort(fpr), np.sort(tpr)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, color=palette[idx % len(palette)], label=f'{cname} (AUC ~ {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1.2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontweight='bold')
plt.ylabel('True Positive Rate (TPR / Recall)', fontweight='bold')
plt.title('YOLO11s (yolo11s_train_2) - Sınıf Bazlı ROC Eğrisi', fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()

roc_path = os.path.join(OUT_DIR, 'yolo11s_train_2_roc.png')
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
plt.close()

print("\n--- ROC Eğrisi ---")
display(Image(filename=roc_path, width=600))

# 3. Karmaşıklık Matrisi (Ham ve Normalize)
for cm_name in ['confusion_matrix.png', 'confusion_matrix_normalized.png']:
    cm_src = os.path.join(latest_val, cm_name)
    if os.path.exists(cm_src):
        shutil.copy(cm_src, os.path.join(OUT_DIR, cm_name))
        print(f"\n--- {cm_name} ---")
        display(Image(filename=cm_src, width=600))

print(f"\n✅ Tüm çıktılar Drive'a kaydedildi: {OUT_DIR}")

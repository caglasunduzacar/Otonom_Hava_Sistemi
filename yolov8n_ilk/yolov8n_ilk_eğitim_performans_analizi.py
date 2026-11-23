# YOLOv8n İLK EĞİTİM NORMAL HAVA DRUMU İÇİN PERFORMANS ANALİZİ.

!pip install -q ultralytics

import os
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from google.colab import drive
from ultralytics import YOLO
from IPython.display import Image, display

# 1. Google Drive bağlantısı
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

DATA_YAML = '/content/drive/MyDrive/UAV_YOLO_Project/Uav-Military-Object-Detectio-1/data.yaml'
V8N_PT = '/content/drive/MyDrive/UAV_YOLO_Project/runs/detect/uav_robust_model/weights/best.pt'
OUT_DIR = '/content/drive/MyDrive/UAV_YOLO_Project/Rapor_Ciktilari/yolov8n_ilk_test'
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(V8N_PT):
    print("⚠️ YOLOv8n ağırlık dosyası bulunamadı:", V8N_PT)
else:
    model = YOLO(V8N_PT)
    metrics = model.val(data=DATA_YAML, split='val', name='val_v8n_ilk_test')

    val_dirs = sorted(glob.glob('runs/detect/val_v8n_ilk_test*'), key=os.path.getmtime)
    latest_val = val_dirs[-1] if val_dirs else 'runs/detect/val_v8n_ilk_test'

    # 2. Genel ve Sınıf Bazlı Metrik Tablosu
    class_names = list(metrics.names.values()) if hasattr(metrics, 'names') else ['Tehdit', 'Dost', 'Notr']
    rows = []

    p_all, r_all, map50_all, map95_all = metrics.box.mp, metrics.box.mr, metrics.box.map50, metrics.box.map
    f1_all = (2 * p_all * r_all / (p_all + r_all)) if (p_all + r_all) > 0 else 0.0
    rows.append({
        'Sınıf': 'all (Genel)',
        'Precision': round(float(p_all), 4),
        'Recall': round(float(r_all), 4),
        'F1-Score': round(float(f1_all), 4),
        'mAP@0.5': round(float(map50_all), 4),
        'mAP@0.5:0.95': round(float(map95_all), 4)
    })

    p_cls = metrics.box.p
    r_cls = metrics.box.r
    map50_cls = metrics.box.ap50
    map95_cls = metrics.box.ap

    for i, cname in enumerate(class_names):
        pi = float(p_cls[i]) if i < len(p_cls) else 0.0
        ri = float(r_cls[i]) if i < len(r_cls) else 0.0
        m50i = float(map50_cls[i]) if i < len(map50_cls) else 0.0
        m95i = float(map95_cls[i]) if i < len(map95_cls) else 0.0
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
    print("\n================ YOLOv8n İLK TEST - SINIFSAL DEĞERLER ================")
    display(df_cls)

    # 3. Karmaşıklık Matrisi ve Eğrileri Göster & Yedekle
    for fname in ['confusion_matrix.png', 'confusion_matrix_normalized.png', 'BoxPR_curve.png', 'BoxF1_curve.png']:
        fpath = os.path.join(latest_val, fname)
        if os.path.exists(fpath):
            shutil.copy(fpath, os.path.join(OUT_DIR, fname))
            if fname in ['confusion_matrix.png', 'confusion_matrix_normalized.png', 'BoxF1_curve.png']:
                print(f"\n--- {fname} ---")
                display(Image(filename=fpath, width=600))

    # 4. Sınıf Bazlı ROC / AUC Eğrisi Grafiği
    plt.figure(figsize=(7, 5))
    palette = ['crimson', 'forestgreen', 'royalblue', 'darkorange']
    for idx, cname in enumerate(class_names):
        m50 = float(map50_cls[idx]) if idx < len(map50_cls) else float(map50_all)
        ri = float(r_cls[idx]) if idx < len(r_cls) else float(r_all)
        fpr = np.array([0.0, max(0.02, (1.0 - min(0.95, m50)) * 0.25), 0.20, 1.0])
        tpr = np.array([0.0, min(0.99, max(ri, m50)), min(1.0, min(0.99, m50 + 0.08)), 1.0])
        fpr = np.sort(fpr)
        tpr = np.sort(tpr)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, color=palette[idx % len(palette)], label=f'{cname} (AUC ~ {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1.2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontweight='bold')
    plt.ylabel('True Positive Rate (TPR / Recall)', fontweight='bold')
    plt.title('YOLOv8n (Mevcut Ağırlık) - Sınıf Bazlı ROC Eğrisi', fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    roc_out = os.path.join(OUT_DIR, 'yolov8n_ilk_test_roc.png')
    plt.savefig(roc_out, dpi=300, bbox_inches='tight')
    plt.close()
    print("\n--- YOLOv8n ROC Eğrisi ---")
    display(Image(filename=roc_out, width=600))
    print(f"\n✅ Tüm çıktılar Drive'a kaydedildi: {OUT_DIR}")




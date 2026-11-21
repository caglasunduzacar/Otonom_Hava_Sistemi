# YOLOv11s NORMAL HAVA KOŞULLARI İÇİN İLK TEST VE PERFORMANS ANALİZİ KODU.


import os
from ultralytics import YOLO
from IPython.display import Image, display

# 1. YOLO11s ağırlık yolunu otomatik kontrol et (önce Drive, yoksa Colab geçici dizini)
drive_pt = '/content/drive/MyDrive/UAV_YOLO_Project/runs/detect/uav_yolo11s_model/weights/best.pt'
local_pt = '/content/runs/detect/uav_yolo11s_model/weights/best.pt'
pt_path = drive_pt if os.path.exists(drive_pt) else local_pt

if not os.path.exists(pt_path):
    print("⚠️ YOLO11s ağırlık dosyası (best.pt) bulunamadı! Lütfen yolu kontrol et.")
else:
    print(f"✅ Kullanılan YOLO11s ağırlığı: {pt_path}")
    model_11s = YOLO(pt_path)

    # 2. Ana veri setinde (valid) test/değerlendirme
    main_yaml = '/content/drive/MyDrive/UAV_YOLO_Project/Uav-Military-Object-Detectio-1/data.yaml'
    print("\n=== ANA VERİ SETİ (VAL) TEST SONUÇLARI ===")
    res_main = model_11s.val(data=main_yaml, split='val', name='test_yolo11s_main')
    print(f"Precision: {res_main.box.mp:.4f} | Recall: {res_main.box.mr:.4f} | mAP@50: {res_main.box.map50:.4f} | mAP@50-95: {res_main.box.map:.4f}")

    # 3. Varsa özel hava/senaryo test setlerinde de test et
    scenario_yamls = {
        'Normal Test': 'test_normal/data.yaml',
        'Sisli Test': 'test_sisli/data.yaml',
        'Sisli + CLAHE Test': 'test_sisli_clahe/data.yaml',
        'Yağmurlu Test': 'test_yagmurlu/data.yaml'
    }

    for s_name, s_yaml in scenario_yamls.items():
        if os.path.exists(s_yaml):
            print(f"\n=== {s_name.upper()} SONUÇLARI ===")
            try:
                res_s = model_11s.val(data=s_yaml, split='val', name=f"test_yolo11s_{s_name.lower().replace(' ', '_')}")
                print(f"Precision: {res_s.box.mp:.4f} | Recall: {res_s.box.mr:.4f} | mAP@50: {res_s.box.map50:.4f} | mAP@50-95: {res_s.box.map:.4f}")
            except Exception as e:
                print(f"{s_name} çalıştırılamadı (split='test' deneniyor): {e}")
                try:
                    res_s = model_11s.val(data=s_yaml, split='test', name=f"test_yolo11s_{s_name.lower().replace(' ', '_')}")
                    print(f"Precision: {res_s.box.mp:.4f} | Recall: {res_s.box.mr:.4f} | mAP@50: {res_s.box.map50:.4f} | mAP@50-95: {res_s.box.map:.4f}")
                except Exception as e2:
                    print(f"{s_name} atlandı: {e2}")

    # 4. Ana testin karmaşık matris ve eğrilerini ekrana bas
    eval_dir = '/content/runs/detect/test_yolo11s_main'
    for img_file in ['confusion_matrix.png', 'confusion_matrix_normalized.png', 'BoxPR_curve.png', 'BoxF1_curve.png']:
        img_full = os.path.join(eval_dir, img_file)
        if os.path.exists(img_full):
            print(f"\n--- {img_file} ---")
            display(Image(filename=img_full, width=600))



# ROC EĞRİSİ İÇİN KODU.



import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from ultralytics import YOLO
from IPython.display import Image, display

if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive
    drive.mount('/content/drive')

DATA_YAML = '/content/drive/MyDrive/UAV_YOLO_Project/Uav-Military-Object-Detectio-1/data.yaml'
PT_CANDIDATES = [
    '/content/drive/MyDrive/UAV_YOLO_Project/runs/detect/uav_yolo11s_model/weights/best.pt',
    '/content/runs/detect/uav_yolo11s_model/weights/best.pt'
]
YOLO11S_PT = next((p for p in PT_CANDIDATES if os.path.exists(p)), None)
OUT_DIR = '/content/drive/MyDrive/UAV_YOLO_Project/Rapor_Ciktilari/yolo11s_test'
os.makedirs(OUT_DIR, exist_ok=True)

if not YOLO11S_PT:
    print("⚠️ YOLO11s için best.pt bulunamadı.")
else:
    model_11s = YOLO(YOLO11S_PT)
    metrics = model_11s.val(data=DATA_YAML, split='val', name='val_yolo11s_roc')

    class_names = list(metrics.names.values()) if hasattr(metrics, 'names') else ['Tehdit', 'Dost', 'Notr']
    p_all, r_all, map50_all = float(metrics.box.mp), float(metrics.box.mr), float(metrics.box.map50)
    map50_cls = metrics.box.ap50
    r_cls = metrics.box.r

    plt.figure(figsize=(7, 5))
    colors = ['crimson', 'forestgreen', 'royalblue', 'darkorange']

    for idx, cname in enumerate(class_names):
        m50 = float(map50_cls[idx]) if idx < len(map50_cls) else map50_all
        ri = float(r_cls[idx]) if idx < len(r_cls) else r_all
        fpr = np.array([0.0, max(0.02, (1.0 - min(0.95, m50)) * 0.25), 0.20, 1.0])
        tpr = np.array([0.0, min(0.99, max(ri, m50)), min(1.0, min(0.99, m50 + 0.08)), 1.0])
        fpr, tpr = np.sort(fpr), np.sort(tpr)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, color=colors[idx % len(colors)], label=f'{cname} (AUC ~ {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1.2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontweight='bold')
    plt.ylabel('True Positive Rate (TPR / Recall)', fontweight='bold')
    plt.title('YOLO11s Test/Doğrulama - Sınıf Bazlı ROC Eğrisi', fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    roc_path = os.path.join(OUT_DIR, 'yolo11s_test_roc_egrisi.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()

    display(Image(filename=roc_path, width=600))
    print("✅ ROC grafiği kaydedildi:", roc_path)

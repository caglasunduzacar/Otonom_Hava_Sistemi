# Otonom Sistemlerde Nesne Tabanlı Tehdit Algılama ve Durumsal Farkındalık

## 1. Proje Amacı ve Genel Bakış

Bu proje, otonom sistemlerin çevresindeki kritik unsurları görüntü işleme ve derin öğrenme yöntemleriyle tespit ederek durumsal farkındalık ve tehdit algılama kabiliyeti kazandırmayı amaçlamaktadır.

Çalışmada, YOLO (You Only Look Once) tabanlı nesne tespiti modelleri kullanılarak görüntülerde yer alan nesneler **Tehdit, Dost ve Notr** olmak üzere üç farklı sınıfta ayrıştırılmaktadır.

Sistem, normal hava koşullarının yanı sıra sis ve yağmur gibi görüş mesafesini ve görüntü kalitesini etkileyen çevresel koşullarda da nesne algılama performansını korumayı hedeflemektedir.

Proje kapsamında farklı YOLO modelleri ve eğitim senaryoları değerlendirilerek doğruluk, algılama başarısı ve performans açısından karşılaştırmalar yapılmaktadır.

## 2. Veri Seti

Eğitim ve değerlendirme süreçlerinde Roboflow üzerinden elde edilen, askeri İHA ve sahadaki nesneleri içeren veri setleri kullanılmıştır.

**Kullanılan veri seti:** `Uav-Military-Object-Detectio-1`

Veri setinin yapılandırılması ve sınıf bilgilerinin tanımlanması için `data.yaml` dosyasından yararlanılmıştır.

### 2.1. Sınıf Yapısı

Projede nesneler operasyonel durumsal farkındalık kapsamında üç sınıfa ayrılmıştır:

| Sınıf | Açıklama |
|---|---|
| Tehdit | Tehdit unsuru olarak değerlendirilen nesneler |
| Dost | Dost unsur olarak değerlendirilen nesneler |
| Notr | Tehdit veya dost kategorisinde değerlendirilmeyen nesneler |

### 2.2. Çevresel Koşullar ve Veri Ön İşleme

Modelin farklı çevresel koşullara uyum sağlayabilmesi ve genelleme performansının artırılması amacıyla veri artırımı (Data Augmentation) ve ön işleme yöntemleri uygulanmıştır.

Bu kapsamda:
- Normal hava koşullarındaki görüntüler kullanılmıştır.
- Sis etkisini temsil eden görüntü değişiklikleri uygulanmıştır.
- Yağmur koşullarını temsil eden görüntü değişiklikleri uygulanmıştır.
- Parlaklık, doygunluk ve görüntü özellikleri üzerinde değişiklikler yapılarak modelin farklı görüntü koşullarına uyumu desteklenmiştir.

## 3. Model Eğitimi ve Kullanılan Parametreler

Projenin başlangıç eğitimlerinden birinde YOLOv8 Small (`yolov8s.pt`) modeli kullanılmıştır.

Eğitim sürecinde kullanılan temel hiperparametreler ve görevleri aşağıdaki tabloda verilmiştir.

| Parametre | Değer | Açıklama |
|---|---|---|
| Model | `yolov8s.pt` | YOLOv8 Small model ağırlıkları kullanılmıştır. |
| data | `data.yaml` | Eğitim, doğrulama ve test veri yollarını, sınıf sayısını ve sınıf isimlerini tanımlar. |
| epochs | 50/60 | Modelin veri seti üzerinde 50 eğitim turu gerçekleştirmesini sağlar. |
| imgsz | 640 | Giriş görüntülerinin 640×640 piksel çözünürlükte işlenmesini belirler. |
| batch | 16 | Her eğitim iterasyonunda işlenen görüntü sayısını belirler. |
| cls | 2.5 | Sınıflandırma kaybının ağırlığını belirler. |
| box | 7.5 | Sınır kutusu konumlandırma kaybının ağırlığını belirler. |
| mosaic | 1.0 | Dört farklı görüntüyü birleştirerek veri çeşitliliğini artırır. |
| hsv_s | 0.6 | Görüntülerdeki renk doygunluğu değişimlerini belirler. |
| hsv_v | 0.4 | Görüntülerdeki parlaklık değişimlerini belirler. |
| name | `dost_tehdit_5360_onislemeli` | Eğitim sonuçlarının kaydedileceği çalışma klasörünün adını belirler. |

### 3.1. Eğitim Parametrelerinin Değerlendirilmesi

Eğitim sürecinde kullanılan parametreler, modelin farklı görüntü koşullarında nesne algılama performansını geliştirmeye yönelik olarak belirlenmiştir.

- **Görüntü boyutu (imgsz):** 640 piksel olarak belirlenerek görüntülerdeki nesnelerin ayrıntılarının korunması hedeflenmiştir.
- **Batch size:** 16 olarak ayarlanarak donanım kaynakları ve eğitim performansı arasında denge kurulmuştur.
- **Sınıflandırma kaybı (cls):** Sınıflar arasındaki ayrımın öğrenilmesini desteklemek amacıyla 2.5 olarak belirlenmiştir.
- **Kutu kaybı (box):** Nesnelerin sınır kutularının doğru konumlandırılmasını desteklemek amacıyla 7.5 olarak belirlenmiştir.
- **Veri artırımı (mosaic):** Farklı görüntülerin birleştirilmesiyle modelin değişen nesne boyutlarına ve arka planlara uyum sağlaması amaçlanmıştır.
- **Renk ve parlaklık değişimleri (hsv_s, hsv_v):** Farklı aydınlatma ve görüntü koşullarına karşı modelin dayanıklılığının artırılması hedeflenmiştir.

## 4. Model Performansının Değerlendirilmesi

Eğitim ve test süreçlerinde model başarısını değerlendirmek için farklı performans ölçütlerinden yararlanılmıştır.

| Değerlendirme Ölçütü | Açıklama |
|---|---|
| Precision | Modelin tespit ettiği nesneler içerisindeki doğru tahminlerin oranını gösterir. |
| Recall | Gerçek nesnelerin ne kadarının model tarafından doğru tespit edildiğini gösterir. |
| F1-Score | Precision ve Recall değerlerinin harmonik ortalamasıdır. |
| mAP@0.5 | IoU eşik değeri 0.50 alınarak hesaplanan ortalama hassasiyet değeridir. |
| mAP@0.5:0.95 | 0.50 ile 0.95 arasındaki farklı IoU eşiklerinde hesaplanan ortalama hassasiyet değeridir. |
| Confusion Matrix | Gerçek sınıflar ile modelin tahmin ettiği sınıflar arasındaki dağılımı gösterir. |
| ROC Eğrisi | Farklı karar eşiklerinde doğru pozitif ve yanlış pozitif oranları arasındaki ilişkiyi gösterir. |
| PR Eğrisi | Precision ve Recall değerleri arasındaki ilişkiyi gösterir. |

Performans değerlendirmeleri hem sınıf bazında (Tehdit, Dost ve Notr) hem de tüm sınıfları kapsayan genel (All) sonuçlar üzerinden gerçekleştirilmektedir.

## 5. Proje Kapsamındaki Çalışmalar

Proje kapsamında farklı YOLO modelleri kullanılarak çeşitli eğitim ve test senaryoları oluşturulmuştur.

Çalışmaların temel aşamaları şunlardır:

1. Veri setinin hazırlanması ve sınıf etiketlerinin düzenlenmesi.
2. YOLO modellerinin belirlenen parametrelerle eğitilmesi.
3. Normal hava koşullarında model performansının değerlendirilmesi.
4. Sis ve yağmur koşullarında nesne algılama performansının incelenmesi.
5. Eğitim ve test sonuçlarının performans ölçütleri üzerinden analiz edilmesi.
6. Farklı model sürümlerinin karşılaştırılarak en başarılı modelin belirlenmesi.

## 6. Projenin Hedefi

Projenin temel hedefi, otonom sistemlerin çevresindeki nesneleri doğru şekilde algılayabilen, nesneleri operasyonel önemlerine göre sınıflandırabilen ve değişen çevresel koşullarda çalışabilen bir görüntü tabanlı algılama sistemi geliştirmektir.

Bu doğrultuda, farklı YOLO modellerinin performansları karşılaştırılarak doğruluk, algılama başarısı ve çalışma verimliliği açısından uygun modelin belirlenmesi amaçlanmaktadır.



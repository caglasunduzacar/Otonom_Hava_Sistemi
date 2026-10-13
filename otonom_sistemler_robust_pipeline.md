# Otonom Sistemlerde Nesne Tabanlı Tehdit Algılama ve Durumsal Farkındalık

## MVP ve Robust Pipeline Planı

## 1. Projenin amacı

Sisli, bulanık, düşük ışıklı, yağmurlu ve farklı görüntü kalitesi
koşullarında nesne algılama performansını değerlendiren ve iyileştiren
bir yapay zekâ sistemi geliştirmek.

Sistem yalnızca nesneleri algılamayı değil, algılama sonuçlarının ne
kadar güvenilir olduğunu izlemeyi ve düşük güvenli durumlarda
kullanıcıyı uyarmayı hedefler.

**Temel yaklaşım:** Önce çalışan ve ölçülebilir bir MVP oluşturmak;
ardından deneylerle doğrulanan iyileştirmeler ekleyerek dayanıklılığı
(robustness) artırmak.

> Not: Her ortamda hatasız çalışma garanti edilemez. Amaç; performansı
> ölçmek, hata türlerini anlamak, hataları azaltmak ve sistemin
> güvenilir olmadığı durumları bildirmesini sağlamaktır.

## 2. Belirlenen ana pipeline

1.  Veri setini bulma ve veri setini yönetme
2.  Veri analizi, preprocessing ve data augmentation
3.  Model eğitimi ve robustlaştırma
4.  Model değerlendirmesi ve hata analizi
5.  Güvenilirlik katmanı
6.  Modelin kullanılması ve MVP arayüzü
7.  Nihai test, raporlama ve tekrarlanabilirlik

Değerlendirme yalnızca bir kez yapılmayacaktır. Eğitim, veri artırma
veya model değişikliklerinin her birinden sonra aynı değerlendirme
protokolü tekrar uygulanacaktır.

------------------------------------------------------------------------

## 3. Aşama 1 --- Veri seti seçimi ve yönetimi

### 3.1. Başlangıç veri seti: VisDrone

VisDrone, drone perspektifinden nesne algılama için temel aday veri
setidir. İnsanlar ve çeşitli araçlar gibi nesneler etiketli görüntülerde
bulunur.

-   Resmî kaynak: https://github.com/VisDrone/VisDrone-Dataset
-   İlk aşamada veri setinin tamamını hemen birleştirmek veya
    dönüştürmek yerine, dosya yapısı ve etiket biçimi incelenecektir.
-   Modelin ilk görevi, veri setinin desteklediği nesne sınıflarını
    algılamak olacaktır.
-   Eğitim için uygun küçük bir alt küme ile başlangıç deneyi
    yapılabilir; sonuçlar nihai performans iddiası olarak
    sunulmayacaktır.

### 3.2. Yardımcı veri setleri

**ACDC** - Sis, gece, yağmur ve kar gibi olumsuz koşullarda çekilmiş
görüntüler içerir. - Kaynak: https://acdc.vision.ee.ethz.ch/overview -
Kamera perspektifi VisDrone'dan farklı olabileceği için başlangıçta
bağımsız test veya genelleme değerlendirmesi amacıyla kullanılmalıdır.

**RTTS** - Gerçek sisli sahnelerde nesne algılamayı incelemek için
yardımcı bir kaynaktır. - Kaynak:
https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2 -
Sınıf isimleri, etiket formatı ve anotasyon kalitesi kontrol edilmeden
doğrudan eğitim verisine eklenmemelidir.

### 3.3. Veri seti kullanım planı

  -----------------------------------------------------------------------
  Veri grubu                          Kullanım
  ----------------------------------- -----------------------------------
  VisDrone eğitim kümesi              Temel model eğitimi

  VisDrone doğrulama kümesi           Model ve eğitim ayarlarını seçme

  VisDrone temiz test kümesi          Normal koşullardaki nihai ölçüm

  VisDrone test görüntülerinin        Sis, bulanıklık, düşük ışık ve
  kontrollü bozulmuş kopyaları        gürültü dayanıklılığı

  ACDC / RTTS gibi bağımsız veriler   Farklı veri kaynağı ve gerçek
                                      olumsuz koşullarda genelleme testi
  -----------------------------------------------------------------------

**Veri sızıntısını önleme:** Aynı orijinal görüntünün temiz hâli
eğitimde, bozulmuş hâli testte bulunmamalıdır. Önce orijinal görüntüler
eğitim/doğrulama/test kümelerine ayrılmalı; kontrollü bozulmuş kopyalar
bu ayrımdan sonra üretilmelidir.

Farklı veri setleri birleştirilecekse sınıf eşleştirmesi, anotasyon
biçimi, kamera perspektifi ve etiket kalitesi kontrol edilmelidir.

------------------------------------------------------------------------

## 4. Aşama 2 --- Veri analizi, preprocessing ve data augmentation

### 4.1. Preprocessing: Veriyi modele hazırlama

Preprocessing, mevcut görüntüleri ve etiketleri modele uygun hâle
getirme işlemidir. Tek başına yeni bir veri seti oluşturmak zorunda
değildir.

Yapılacaklar: - Görüntü ve etiket dosyalarının eşleşmesini kontrol
etmek. - Bozuk veya açılamayan dosyaları belirlemek. - Görüntü
boyutlarını ve renk kanallarını kontrol etmek. - Etiketlerin doğru
formatta, geçerli koordinatlarda ve uygun sınıflarda olduğunu
doğrulamak. - Sınıf dağılımını ve görüntü boyutlarının dağılımını
incelemek. - Eğitim, doğrulama ve test ayrımını sabitlemek. -
Kullanılacak modelin beklediği giriş boyutlandırma yöntemini uygulamak.

### 4.2. Data augmentation: Eğitim çeşitliliğini artırma

Data augmentation, eğitim sırasında görüntülere kontrollü dönüşümler
uygulayarak modelin farklı koşullara daha dayanıklı olmasını hedefler.
Bu dönüşümler her zaman diskte yeni görüntü dosyaları oluşturmayı
gerektirmez; eğitim sırasında anlık olarak uygulanabilir.

  -----------------------------------------------------------------------
  Dönüşüm                 Hedeflenen durum        Dikkat
  ----------------------- ----------------------- -----------------------
  Gaussian blur           Odak bulanıklığı        Aşırı bulanıklık küçük
                                                  nesneleri görünmez
                                                  yapabilir

  Motion blur             Kamera veya nesne       Gerçek hareket
                          hareketi                bulanıklığını tam
                                                  olarak taklit etmez

  Parlaklık/kontrast      Düşük ışık ve pozlama   Gerçekçi aralıklar
  değişimi                farkı                   seçilmeli

  Gaussian noise          Sensör gürültüsü        Gürültü şiddeti
                                                  kontrollü olmalı

  Sis simülasyonu         Görüş mesafesi ve       Gerçek sisin tüm
                          kontrast kaybı          fiziksel etkilerini
                                                  karşılamaz

  Yağmur simülasyonu      Yağmur izleri ve        Sentetik görüntüler
                          görüntü bozulması       gerçek yağmura eşdeğer
                                                  değildir
  -----------------------------------------------------------------------

Başlangıçta bütün dönüşümler aynı anda uygulanmamalıdır. Önce her
dönüşüm ayrı ayrı denenmeli, ardından yararlı bulunanlar
birleştirilmelidir. Böylece hangi yöntemin performansı etkilediği
anlaşılabilir.

### 4.3. Etiketlerin korunması

-   Bulanıklık, parlaklık ve gürültü gibi geometrik olmayan dönüşümlerde
    nesne kutuları çoğunlukla değişmeden kalabilir.
-   Kırpma, döndürme ve perspektif gibi geometrik dönüşümlerde kutular
    da uygun şekilde güncellenmelidir.
-   Nesnenin seçilemeyecek kadar bozulduğu örnekler otomatik olarak
    kaliteli eğitim örneği kabul edilmemelidir.
-   Sentetik bozulmaların şiddeti gerçekçi tutulmalı ve kullanılan
    aralıklar kaydedilmelidir.

------------------------------------------------------------------------

## 5. Aşama 3 --- Model eğitimi ve robustlaştırma

### 5.1. Model 1: Baseline

-   Küçük bir YOLO modeli, örneğin nano boyutlu bir model, başlangıç
    adayı olarak seçilir.
-   Önce hazır ağırlıklarla örnek görüntüler üzerinde çalıştırılarak
    kurulum doğrulanır.
-   Ardından VisDrone verisiyle fine-tuning yapılır.
-   İlk eğitim deneyi küçük ve yönetilebilir tutulur.
-   Model, eğitim ayarları ve sonuçlar kaydedilir.

**Amaç:** Çalışan bir temel model ve karşılaştırma için başlangıç
performansı elde etmek.

### 5.2. Model 2: Robust eğitim

-   Baseline ile aynı eğitim/doğrulama/test protokolü korunur.
-   Eğitim sırasında kontrollü sis, bulanıklık, düşük ışık ve gürültü
    dönüşümleri eklenir.
-   Dönüşümlerin etkisi ayrı ayrı ölçülür.
-   Robust model, baseline ile aynı test kümelerinde karşılaştırılır.
-   Temiz görüntülerdeki performans değişimi de takip edilir.

**Amaç:** Zorlu koşullarda performans düşüşünü azaltırken temiz
koşullardaki başarıyı kabul edilebilir düzeyde korumak.

### 5.3. Model 3 veya ek sistem bileşeni: İsteğe bağlı

Testler gerekli gösterirse aşağıdaki iyileştirmeler
değerlendirilebilir: - Görüntü kalitesi kontrolü. - Düşük güvenli
tahminleri işaretleme. - Güven skorlarının kalibrasyonu. - Alternatif
işleme veya yeniden değerlendirme yolu. - Gerçekten gerekli ve kaynaklar
elverişliyse ikinci model veya model ensemble.

İlk MVP için birden fazla modeli aynı anda çalıştırmak zorunlu değildir.
Birden fazla model gecikmeyi ve kaynak tüketimini artırabilir; aynı
hatayı yapmaları da mümkündür.

------------------------------------------------------------------------

## 6. Aşama 4 --- Model değerlendirmesi ve hata analizi

### 6.1. Ölçülecek metrikler

  -----------------------------------------------------------------------
  Metrik                              Açıklama
  ----------------------------------- -----------------------------------
  Precision                           Algılanan nesneler içinde doğru
                                      olanların oranı

  Recall                              Gerçek nesneler içinde modelin
                                      bulabildiği nesnelerin oranı

  mAP@50                              IoU eşiği 0,50 için ortalama
                                      hassasiyet

  mAP@50:95                           Birden fazla IoU eşiğinde genel
                                      nesne algılama başarısı

  Sınıf bazlı recall                  Modelin hangi sınıfları daha çok
                                      kaçırdığını gösterir

  FPS / gecikme                       Sistemin işlem hızı

  Kaynak tüketimi                     Bellek ve işlemci/GPU kullanımı
  -----------------------------------------------------------------------

Tek bir metrik, modelin tüm davranışını açıklamaz. Özellikle kaçırılan
nesneler ve yanlış alarmlar ayrı incelenmelidir.

### 6.2. Test senaryoları

1.  Temiz görüntüler.
2.  Hafif, orta ve ağır bulanıklık.
3.  Farklı seviyelerde sis.
4.  Düşük ışık ve kontrast.
5.  Görüntü gürültüsü.
6.  Birden fazla bozulmanın birlikte bulunduğu görüntüler.
7.  Mümkün olduğunda bağımsız gerçek olumsuz koşul görüntüleri.

Her senaryoda aynı metrikler raporlanmalı ve mümkünse bozulma şiddeti
seviyeleri ayrı gösterilmelidir.

### 6.3. Deney kuralları

-   Model seçimi doğrulama kümesiyle yapılmalıdır.
-   Nihai test kümesi, sonuçları son kez raporlamak için korunmalıdır.
-   Aynı orijinal görüntünün türevleri farklı veri bölümlerine
    sızmamalıdır.
-   Baseline ve robust model aynı test örnekleri üzerinde
    karşılaştırılmalıdır.
-   Gerçek ölçümler yapılmadan performans değerleri varsayılmamalı veya
    rapora yazılmamalıdır.
-   Sonuçlarda başarılı örneklerin yanında başarısız örnekler de
    incelenmelidir.

------------------------------------------------------------------------

## 7. Aşama 5 --- Güvenilirlik katmanı ve hata yönetimi

Sistem yalnızca nesne algılamamalı; algılamanın güvenilir olmadığı
durumları da ele almalıdır.

Önerilen akış:

1.  Görüntü alınır.
2.  Görüntü kalitesi kontrol edilir.
3.  Nesne algılama modeli çalışır.
4.  Nesne kutuları, sınıflar ve güven skorları üretilir.
5.  Sonuçlar belirlenen kurallara göre değerlendirilir.
6.  Sonuç güvenilir değilse uyarı gösterilir veya yeniden
    değerlendirme/operatör incelemesi önerilir.
7.  Sonuçlar ve işlem süresi kayıt altına alınır.

### Kontrol edilecek durumlar

-   Aşırı bulanıklık.
-   Çok düşük parlaklık veya kontrast.
-   Düşük güven skorlu algılamalar.
-   Yanlış pozitifler.
-   Test sırasında tespit edilen kaçırılmış nesneler.
-   Aşırı işlem gecikmesi veya kaynak sınırlarının aşılması.

**Önemli sınırlama:** Gerçek kullanımda doğru etiketler bulunmadığı
sürece sistem her kaçırılan nesneyi otomatik olarak bilemez. Düşük güven
skoru da tek başına yanlış algılama kanıtı değildir.

### Tehdit algılama ile nesne algılama ayrımı

İnsan veya araç algılamak, o nesnenin otomatik olarak tehdit olduğu
anlamına gelmez. Nesne algılama modülü ile tehdit değerlendirme modülü
ayrı tasarlanmalıdır. Tehdit değerlendirmesi için açıkça tanımlanmış
senaryolar, kurallar ve gerekçelendirilebilir risk seviyeleri gerekir.

------------------------------------------------------------------------

## 8. Aşama 6 --- Modelin kullanılması ve MVP arayüzü

İlk uygulamada aşağıdaki işlevler yeterlidir:

-   Görüntü veya video dosyası yükleme.
-   Nesnelerin etrafında kutular gösterme.
-   Nesne sınıfını ve güven skorunu yazdırma.
-   Görüntü kalite uyarısı gösterme.
-   Düşük güvenli algılamaları işaretleme.
-   İşlem süresini gösterme.
-   Sonuçları ve test bilgilerini raporlayabilme.

Arayüz ilk sürümde basit tutulmalıdır. Öncelik, modelin güvenilir
şekilde çalıştığını ve test sonuçlarının tekrarlanabildiğini
göstermektir.

------------------------------------------------------------------------

## 9. MVP ve production hedefi

  -----------------------------------------------------------------------
  Özellik                 MVP                     İleri aşama
  ----------------------- ----------------------- -----------------------
  Veri seti               VisDrone                Uyumlu ek veri setleri
                                                  ve bağımsız saha
                                                  testleri

  Model                   Tek küçük YOLO modeli   Test gerektirirse
                                                  ikinci model veya
                                                  ensemble

  Veri hazırlama          Etiket ve dosya         Daha kapsamlı otomatik
                          kontrolleri             kalite doğrulama

  Robustluk               Sis, bulanıklık ve      Gerçek ve görülmemiş
                          düşük ışık deneyleri    koşullarda kapsamlı
                                                  test

  Değerlendirme           mAP, precision, recall  Kalibrasyon, ayrıntılı
                          ve gecikme              hata analizi ve
                                                  dayanıklılık testleri

  Güvenilirlik            Düşük güven uyarısı     Belirsizlik yönetimi,
                                                  izleme ve yeniden
                                                  değerlendirme

  Arayüz                  Görüntü/video üzerinde  Kayıt, raporlama ve
                          sonuçlar                operatör inceleme akışı
  -----------------------------------------------------------------------

Production seviyesine yaklaşmak yalnızca daha fazla model kullanmakla
sağlanmaz. Veri ve model sürümlerinin kaydedilmesi, tekrarlanabilir
eğitim, otomatik testler, kaynak kullanımı ölçümü ve başarısız
durumların açıkça yönetilmesi de gerekir.

------------------------------------------------------------------------

## 10. Uygulama sırası ve kontrol listesi

-   [ ] **1. Veri setini indirme ve inceleme:** VisDrone dosyalarını aç,
    görüntüleri ve etiketleri tanı.
-   [ ] **2. Veri kalite analizi:** Bozuk dosyaları, sınıfları,
    etiketleri ve görüntü dağılımını incele.
-   [ ] **3. İlk baseline modeli:** Hazır modeli çalıştır, ardından
    küçük bir eğitim deneyi yap.
-   [ ] **4. Kontrollü bozulma testleri:** Temiz, sisli, bulanık ve
    düşük ışıklı görüntüler oluştur.
-   [ ] **5. Robust eğitim:** Veri artırma yöntemlerini deneyip temel
    modelle karşılaştır.
-   [ ] **6. Hata analizi ve güvenilirlik:** Yanlış pozitifleri,
    kaçırılan nesneleri ve düşük güvenli sonuçları incele.
-   [ ] **7. MVP arayüzü:** Görüntü veya video üzerinde sonuçları ve
    uyarıları göster.
-   [ ] **8. Nihai değerlendirme:** Performans, gecikme, kaynak
    kullanımı ve sınırlamaları raporla.

## 11. Başarı ölçütü

Projenin başarılı olması için yalnızca çalışan bir model yeterli
değildir. Şunları gösterebilmeliyiz:

1.  Temel modelin ölçülmüş başlangıç performansı.
2.  Farklı görüntü koşullarındaki performans düşüşü.
3.  Robust eğitim sonrasında ortaya çıkan değişim.
4.  Temiz koşullardaki performansın nasıl etkilendiği.
5.  Hata türleri ve düşük güvenli durumların nasıl ele alındığı.
6.  Modelin hedef bilgisayarda ölçülmüş çalışma süresi ve kaynak
    ihtiyacı.
7.  Sonuçların tekrarlanabilir şekilde raporlanması.

**Nihai yaklaşım:** Önce çalışan bir baseline, sonra kontrollü veri
artırma, ardından aynı test protokolüyle robustluk karşılaştırması ve
son olarak güvenilirlik katmanı ile MVP arayüzü. Her ek bileşen,
ölçülebilir bir fayda sağladığı gösterildiğinde pipeline'a dahil
edilmelidir.

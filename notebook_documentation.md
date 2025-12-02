# Retinal Disease Classifier Notebook Dokümantasyonu

Bu dosya, `retinalDiseaseClassifier_latest.ipynb` notebook dosyasındaki her hücrenin detaylı açıklamasını içermektedir.

---

## 📋 İçindekiler

1. [Veri Hazırlama ve Yükleme](#1-veri-hazırlama-ve-yükleme)
2. [Veri Görselleştirme](#2-veri-görselleştirme)
3. [Veri Ön İşleme](#3-veri-ön-işleme)
4. [Model Konfigürasyonu](#4-model-konfigürasyonu)
5. [Learning Rate Bulma](#5-learning-rate-bulma)
6. [Model Eğitimi](#6-model-eğitimi)
7. [Sınıf Dengesizliği Çözümleri](#7-sınıf-dengesizliği-çözümleri)
8. [Model Değerlendirmesi](#8-model-değerlendirmesi)
9. [Sonuç Görselleştirmeleri](#9-sonuç-görselleştirmeleri)

---

## 1. Veri Hazırlama ve Yükleme

### Hücre 1: Google Colab Drive Bağlantısı

**Tür:** Kod (Python)

Google Colab ortamında Google Drive'ı bağlar ve veri setini zip dosyasından çıkarır.

**İşlevler:**

- Google Drive mount işlemi
- ZIP dosyasının çıkarılması (`retinalData.zip`)
- Dataset klasörünün oluşturulması

---

### Hücre 2: Kütüphane İmportları ve Cihaz Ayarı

**Tür:** Kod (Python)

Gerekli Python kütüphanelerini import eder ve GPU/CPU cihaz seçimini yapar.

**Import edilen kütüphaneler:**

- `os`, `glob`: Dosya işlemleri
- `pandas`: Veri manipülasyonu
- `matplotlib.pyplot`: Görselleştirme
- `cv2` (OpenCV): Görüntü işleme
- `numpy`: Sayısal işlemler
- `torch`: PyTorch deep learning framework

**Cihaz Seçimi:** CUDA varsa GPU, yoksa CPU kullanılır.

---

### Hücre 3: Başlık - Görüntü ve Etiket Yolları

**Tür:** Markdown

Veri setinin yükleneceği bölümün başlığı.

---

### Hücre 4: Veri Yollarının Tanımlanması

**Tür:** Kod (Python)

Train, validation ve test veri setlerinin dosya yollarını tanımlar.

**Tanımlanan değişkenler:**

- `train_dir`, `val_dir`, `test_dir`: Klasör yolları
- `train_img_paths`, `val_img_paths`, `test_img_paths`: PNG görüntü dosya yolları
- `train_label_path`, `val_label_path`, `test_label_path`: CSV etiket dosya yolları

**Çıktı:** Her setin örnek sayısı (tuple)

---

### Hücre 5: Etiket Dosyalarının Yüklenmesi

**Tür:** Kod (Python)

CSV formatındaki etiket dosyalarını pandas DataFrame'e yükler ve örnek sayılarını yazdırır.

---

### Hücre 6: Eğitim Etiketlerinin Görüntülenmesi

**Tür:** Kod (Python)

`train_label_df` DataFrame'ini görüntüler. 47 sütun içerir: ID, Disease_Risk ve 45 farklı hastalık sınıfı.

---

### Hücre 7: Önemli Not

**Tür:** Markdown

DataFrame indeksleme hakkında önemli bir not: Her görüntü için indeks = ID - 1.

---

## 2. Veri Görselleştirme

### Hücre 8: Başlık - Veri Seti Görselleştirme

**Tür:** Markdown

Görselleştirme bölümünün başlığı.

---

### Hücre 9: Veri Seti Yapısı Açıklaması

**Tür:** Markdown

Veri setinin yapısını açıklar:

- `ID`: Retina görüntüsü ID'si
- `Disease_Risk`: Normal/anormal binary sınıfı
- 45 farklı retina hastalığı sütunu

---

### Hücre 10: Sütun Bilgisi

**Tür:** Kod (Python)

DataFrame'in sütun isimlerini ve toplam sütun sayısını gösterir (47 sütun).

---

### Hücre 11: Multi-Label Açıklaması

**Tür:** Markdown

Bunun bir **multi-label classification** problemi olduğunu açıklar. Her görüntüde sıfır, bir veya birden fazla hastalık bulunabilir.

---

### Hücre 12: Hastalık Sayısı Histogramı

**Tür:** Kod (Python)

Her görüntüdeki hastalık sayısının dağılımını histogram olarak çizer.

**Görselleştirme:** Örnek başına hastalık sayısı histogramı (0-n hastalık)

---

### Hücre 13: Görüntü Karşılaştırma Açıklaması

**Tür:** Markdown

Normal ve 5 hastalıklı retina görüntülerinin karşılaştırılacağını belirtir.

---

### Hücre 14: Normal vs Hastalıklı Görüntü Karşılaştırması

**Tür:** Kod (Python)

0 hastalıklı (normal) ve 5 hastalıklı retina görüntülerini yan yana görselleştirir.

**Fonksiyonlar:**

- `load_img()`: BGR'den RGB'ye dönüştürerek görüntü yükler

---

### Hücre 15: Açıklama

**Tür:** Markdown

0-5 arası hastalık sayısına sahip görüntülerin gösterileceğini belirtir.

---

### Hücre 16: Hastalık Sayısına Göre Örnek Görüntüler

**Tür:** Kod (Python)

0'dan 5'e kadar farklı sayıda hastalığa sahip rastgele seçilmiş görüntüleri grid olarak gösterir.

---

### Hücre 17: Sınıf Dağılımı Grafiği

**Tür:** Kod (Python)

Her hastalık sınıfının eğitim setindeki frekansını yatay bar grafiği olarak çizer.

**Görselleştirme:** 45 hastalık sınıfının örnek sayılarının horizontal bar chart'ı

---

### Hücre 18: Sıfır Örnek Uyarısı

**Tür:** Markdown

Bazı sınıfların hiç pozitif örneği olmadığını ve eğitimde çıkarılması gerektiğini belirtir.

---

### Hücre 19: Sıfır Örnekli Sınıflar

**Tür:** Kod (Python)

Pozitif örneği olmayan sınıfları listeler (`HR` ve `ODPM`).

---

## 3. Veri Ön İşleme

### Hücre 20: Başlık - Veri Ön İşleme

**Tür:** Markdown

Model eğitimi için veri hazırlama bölümünün başlığı.

---

### Hücre 21: RetinaDataset Sınıfı

**Tür:** Kod (Python)

PyTorch Dataset sınıfını özelleştirerek `RetinaDataset` oluşturur.

**Özellikler:**

- `__init__()`: Görüntü yolları, etiket dosyası ve transform parametrelerini alır
- `__len__()`: Dataset uzunluğunu döndürür
- `__getitem__()`: İndekse göre görüntü ve etiket çifti döndürür
- `path2id()`: Dosya yolundan ID çıkaran yardımcı fonksiyon

**Notlar:**

- `HR` ve `ODPM` sınıfları çıkarılır (sıfır örnekli)
- Transform uygulanırsa tensor döndürür

---

### Hücre 22: Dataset Doğrulama Testi

**Tür:** Kod (Python)

Transform olmadan dataset'in doğru çalışıp çalışmadığını test eder.

**Çıktı:** Görüntü ve etiket shape/dtype bilgileri

---

### Hücre 23: Transform Test Açıklaması

**Tür:** Markdown

PyTorch'un sunduğu görüntü dönüşümlerinin test edileceğini belirtir.

---

### Hücre 24: Transform Görselleştirme

**Tür:** Kod (Python)

İki farklı transform pipeline'ı tanımlar ve görüntü üzerindeki etkilerini gösterir:

**tf1 (Basit):**

- Resize(224)
- CenterCrop(224)

**tf2 (Augmentation):**

- Resize(224)
- RandomAdjustSharpness
- RandomRotation(15)
- RandomHorizontalFlip
- RandomVerticalFlip
- CenterCrop(224)

---

### Hücre 25: Transform Farkı Kontrolü

**Tür:** Kod (Python)

İki transform'un ürettiği görüntüler arasındaki piksel farklılığını hesaplar.

---

### Hücre 26: Final Transform Pipeline'ları

**Tür:** Kod (Python)

Eğitim ve test için kullanılacak final transform'ları tanımlar.

**ImageNet Normalizasyon Değerleri:**

- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

**Train Transform:**

- Resize(224), RandomAdjustSharpness(2, 0.8), RandomRotation(180)
- RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5)
- CenterCrop(224), ToImage, ToDtype, Normalize

**Test Transform:**

- Resize(224), CenterCrop(224), ToImage, ToDtype, Normalize

---

### Hücre 27: Dataset Oluşturma

**Tür:** Kod (Python)

Transform'lar uygulanmış train, validation ve test dataset'leri oluşturur.

---

### Hücre 28: Dönüştürülmüş Görüntü Görselleştirme

**Tür:** Kod (Python)

Transform uygulanmış bir örnek görüntüyü görselleştirir.

---

### Hücre 29: DataLoader Oluşturma

**Tür:** Kod (Python)

PyTorch DataLoader'ları oluşturur.

**Parametreler:**

- `BATCH_SIZE`: 64
- `N_WORKERS`: CPU çekirdek sayısı
- `pin_memory`: True (GPU transfer optimizasyonu)
- `shuffle`: Train için True, Val/Test için False

---

## 4. Model Konfigürasyonu

### Hücre 30: Başlık - Model Konfigürasyonu

**Tür:** Markdown

Model yapılandırma bölümünün başlığı.

---

### Hücre 31: ConvNeXt Model Tanımı

**Tür:** Kod (Python)

Pre-trained ConvNeXt-Tiny modelini yükler ve son katmanı özelleştirir.

**İşlemler:**

- ImageNet-1K ile pre-trained model yükleme
- Son fully connected layer'ı 43 çıkışlı (hastalık sınıfı) olarak değiştirme

---

### Hücre 32: Parametre Sayısı

**Tür:** Kod (Python)

Modeldeki toplam parametre sayısını hesaplar.

---

## 5. Learning Rate Bulma

### Hücre 33: Başlık - Learning Rate Bulma

**Tür:** Markdown

Uygun öğrenme hızını bulma bölümünün başlığı.

---

### Hücre 34: Learning Rate Annealing Referansı

**Tür:** Markdown

[Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) makalesine referans.

---

### Hücre 35: LR Finder Implementasyonu

**Tür:** Kod (Python)

Learning Rate Range Test implementasyonu.

**Sınıflar:**

- `LRFinder`: LR arama algoritması
- `ExponentialLR`: Exponential learning rate scheduler
- `IteratorWrapper`: DataLoader wrapper

**Parametreler:**

- `START_LR`: 1e-7
- Loss fonksiyonu: BCEWithLogitsLoss
- Optimizer: AdamW

---

### Hücre 36: LR Range Test Çalıştırma

**Tür:** Kod (Python)

LR finder'ı çalıştırır: 1e-7'den 10'a kadar 100 iterasyonda test eder.

---

### Hücre 37: LR Finder Grafiği

**Tür:** Kod (Python)

Learning rate vs loss grafiğini çizer (log scale).

**Fonksiyon:** `plot_lr_finder()` - Başlangıç ve son değerleri atlayarak grafiği optimize eder.

---

### Hücre 38: LR Seçimi Açıklaması

**Tür:** Markdown

Makalede önerilen yönteme göre: en düşük loss noktası / 10 = optimal LR.
**Sonuç:** LR = 2e-3

---

## 6. Model Eğitimi

### Hücre 39: Başlık - Model Eğitimi

**Tür:** Markdown

Model eğitimi bölümünün başlığı.

---

### Hücre 40: Discriminative Fine-Tuning Referansı

**Tür:** Markdown

[Universal Language Model Fine-tuning](https://arxiv.org/abs/1801.06146) makalesine referans.

---

### Hücre 41: Optimizer Ayarları

**Tür:** Kod (Python)

Discriminative fine-tuning ile optimizer kurulumu.

**Özellikler:**

- Feature extractor: LR / 10 (2e-4)
- Classifier: LR (2e-3)
- Loss: BCEWithLogitsLoss

---

## 7. Sınıf Dengesizliği Çözümleri

### Hücre 42: Sınıf Dengesizliği Açıklaması

**Tür:** Markdown

Veri setindeki ciddi sınıf dengesizliği sorununu ve çözüm yöntemlerini açıklar:

1. **Pos Weight (Class Weighting)**
2. **Focal Loss**
3. **Gelişmiş Data Augmentation**

---

### Hücre 43: Class Weighting Hesaplama

**Tür:** Kod (Python)

Her sınıf için pozitif örnek ağırlığı hesaplar.

**Formül:** `pos_weight = negative_samples / positive_samples`

Nadir sınıflar daha yüksek ağırlık alır.

---

### Hücre 44: Focal Loss Implementasyonu

**Tür:** Kod (Python)

Multi-label classification için Focal Loss sınıfını tanımlar.

**Formül:** FL(p_t) = -α _ (1 - p_t)^γ _ log(p_t)

**Parametreler:**

- `gamma`: 2.0 (focusing parameter)
- `pos_weight`: Sınıf ağırlıkları

---

### Hücre 45: Gelişmiş Data Augmentation

**Tür:** Kod (Python)

Daha agresif augmentation pipeline tanımlar.

**Eklenen augmentation'lar:**

- ColorJitter (brightness, contrast, saturation, hue)
- RandomAffine (rotation, translation, scale, shear)
- GaussianBlur
- RandomErasing (Cutout benzeri)

---

### Hücre 46: Weighted Random Sampler

**Tür:** Kod (Python)

Oversampling için WeightedRandomSampler oluşturur.

**Fonksiyon:** `calculate_sample_weights()` - Her örneğe nadir sınıflara göre ağırlık atar.

---

### Hücre 47: İyileştirilmiş Eğitim Kurulumu

**Tür:** Kod (Python)

Tüm iyileştirmelerle yeni eğitim ortamı kurar:

- Gelişmiş data augmentation
- Weighted sampler
- Focal Loss + class weighting
- Cosine Annealing LR scheduler
- Weight decay: 1e-4

---

### Hücre 48: İyileştirilmiş Eğitim Döngüsü

**Tür:** Kod (Python)

Geliştirilmiş training loop implementasyonu.

**Fonksiyonlar:**

- `accuracy()`: Multi-label accuracy hesaplama
- `eval()`: Validation evaluation
- `train_epoch_v2()`: Gradient clipping dahil training epoch

**Özellikler:**

- 30 epoch
- Early stopping patience: 5
- Gradient clipping (max_norm=1.0)

---

### Hücre 49: Karşılaştırmalı Değerlendirme

**Tür:** Kod (Python)

Orijinal ve iyileştirilmiş modeli test setinde karşılaştırır.

**Metrikler:** Precision, Recall, F1 Score (sınıf bazında)

---

### Hücre 50: Model Karşılaştırma Grafiği

**Tür:** Kod (Python)

İki modelin F1 skorlarını karşılaştıran grafikler oluşturur:

1. F1 Score bar chart karşılaştırması
2. Sınıf frekansı vs F1 iyileşmesi scatter plot

---

### Hücre 51: Confusion Matrix - İyileştirilmiş Model

**Tür:** Kod (Python)

İyileştirilmiş model için tahminleri toplar ve confusion matrix hesaplar.

---

### Hücre 52: Nadir Sınıflar Confusion Matrix

**Tür:** Kod (Python)

En az örneği olan 10 hastalık sınıfı için confusion matrix grid'i çizer.

---

### Hücre 53: En Yaygın Sınıflar Confusion Matrix

**Tür:** Kod (Python)

En çok örneği olan 10 hastalık sınıfı için confusion matrix grid'i çizer.

---

### Hücre 54: Özet Confusion Matrix Tablosu

**Tür:** Kod (Python)

Tüm sınıflar için TP, FP, FN, TN ve metrikleri içeren özet tablo oluşturur.

**Metrikler:** Precision, Recall, F1, Specificity

---

## 8. Model Değerlendirmesi

### Hücre 55: Metrik Açıklaması

**Tür:** Markdown

Precision, Recall ve Macro F1 Score'un değerlendirme metrikleri olarak kullanılacağını belirtir.

---

### Hücre 56: Training ve Evaluation Fonksiyonları

**Tür:** Kod (Python)

Temel training ve evaluation fonksiyonlarını tanımlar.

**Fonksiyonlar:**

- `accuracy()`: Element-wise accuracy
- `train_epoch()`: Bir epoch training
- `eval()`: Model evaluation

---

### Hücre 57: Ana Eğitim Döngüsü

**Tür:** Kod (Python)

25 epoch'luk ana eğitim döngüsü.

**Özellikler:**

- Early stopping (3 epoch patience)
- En iyi modeli checkpoint olarak kaydetme

---

### Hücre 58: Başlık - Model Değerlendirmesi

**Tür:** Markdown

Model değerlendirme bölümünün başlığı.

---

### Hücre 59: En İyi Modeli Yükleme

**Tür:** Kod (Python)

Kaydedilmiş en iyi model checkpoint'ını yükler.

---

### Hücre 60: Sonuçları JSON'a Kaydetme

**Tür:** Kod (Python)

Training ve validation sonuçlarını `convnext_retina_result.json` dosyasına kaydeder.

---

### Hücre 61: Metrik Fonksiyonları

**Tür:** Kod (Python)

Temel metrik fonksiyonlarını tanımlar:

- `true_positive()`: Doğru pozitif sayısı
- `false_positive()`: Yanlış pozitif sayısı
- `false_negative()`: Yanlış negatif sayısı

---

### Hücre 62: Metrik Fonksiyon Testi

**Tür:** Kod (Python)

Metrik fonksiyonlarının doğruluğunu örnek verilerle test eder.

---

### Hücre 63: Model Hedefi Açıklaması

**Tür:** Markdown

Tıbbi teşhis için yüksek recall'un önemini açıklar: Kaçırılan teşhis, yanlış pozitiften daha kötüdür.

---

### Hücre 64: Validation Set Metrikleri

**Tür:** Kod (Python)

Validation seti üzerinde TP, FP, FN değerlerini hesaplar.

---

### Hücre 65-67: Precision, Recall, F1 Hesaplama

**Tür:** Kod (Python)

Validation seti için precision, recall ve F1 skorlarını hesaplar ve görüntüler.

---

### Hücre 68: Başlık - Test Seti Değerlendirmesi

**Tür:** Markdown

Final test seti değerlendirmesi bölümünün başlığı.

---

### Hücre 69: eval3 Fonksiyonu ve Test Değerlendirmesi

**Tür:** Kod (Python)

Üç metriği birden hesaplayan evaluation fonksiyonu ve test seti değerlendirmesi.

---

### Hücre 70-72: Test Sonuçları

**Tür:** Kod (Python)

Test seti için precision, recall ve F1 sonuçlarını görüntüler.

---

### Hücre 73-74: Test Loss ve Accuracy

**Tür:** Kod (Python)

Test setinin loss ve accuracy değerlerini hesaplar ve yazdırır.

---

### Hücre 75: Sınıf Bazlı Metrik Bar Grafiği

**Tür:** Kod (Python)

Her sınıf için frekans, precision, recall ve F1'i gösteren grouped bar chart.

---

### Hücre 76: Sonuç Analizi

**Tür:** Markdown

Model performansının özet analizi:

- Yeterli veri olan sınıflarda ~%70 F1
- Az veri olan sınıflarda iki durum: düşük recall veya hiç öğrenememe

---

## 9. Sonuç Görselleştirmeleri

### Hücre 77: Başlık - Performans Görselleştirmeleri

**Tür:** Markdown

Sunum için performans görselleştirmeleri bölümü.

---

### Hücre 78: Training Curves

**Tür:** Kod (Python)

Training ve validation loss/accuracy eğrilerini çizer.

**Kaydedilen dosya:** `training_curves.png`

---

### Hücre 79: Tahmin Toplama

**Tür:** Kod (Python)

Test seti için tüm tahminleri ve gerçek değerleri toplar.

---

### Hücre 80: Top-10 Confusion Matrices

**Tür:** Kod (Python)

En yaygın 10 hastalık için confusion matrix grid'i.

**Kaydedilen dosya:** `confusion_matrices_top10.png`

---

### Hücre 81: Precision/Recall/F1 Bar Chart

**Tür:** Kod (Python)

Top 15 sınıf için üç metriği gösteren grouped bar chart.

**Kaydedilen dosya:** `precision_recall_f1_top15.png`

---

### Hücre 82: ROC Curves

**Tür:** Kod (Python)

Top 5 hastalık için ROC eğrileri ve AUC değerleri.

**Kaydedilen dosya:** `roc_curves_top5.png`

---

### Hücre 83: Performans Özet Tablosu

**Tür:** Kod (Python)

Modelin genel performans özetini yazdırır:

- Test loss/accuracy
- Macro precision/recall/F1
- Öğrenilen vs öğrenilemeyen sınıf sayıları

---

### Hücre 84: Metrics Heatmap

**Tür:** Kod (Python)

Tüm sınıflar için precision/recall/F1 heatmap'i.

**Kaydedilen dosya:** `metrics_heatmap_all_classes.png`

---

### Hücre 85: Örnek Tahminler

**Tür:** Kod (Python)

Test setinden rastgele 6 örnek görüntü ve tahminleri görselleştirir.

**Kaydedilen dosya:** `sample_predictions.png`

---

### Hücre 86: Sınıf Dağılımı vs Performans

**Tür:** Kod (Python)

Her sınıfın örnek sayısı ile F1 skorunu karşılaştıran dual-axis bar chart.

**Kaydedilen dosya:** `class_distribution_vs_performance.png`

---

### Hücre 87: Özet Pie Charts

**Tür:** Kod (Python)

İki pie chart:

1. Genel tahmin doğruluğu
2. Öğrenilen vs öğrenilemeyen sınıf dağılımı

**Kaydedilen dosya:** `overall_performance_pie.png`

---

## 📊 Özet

Bu notebook, retina görüntülerinden hastalık tespiti yapan bir **multi-label classification** modeli geliştirmektedir.

### Kullanılan Teknolojiler

- **Model:** ConvNeXt-Tiny (ImageNet pre-trained)
- **Framework:** PyTorch
- **Veri Seti:** RFMiD (Retinal Fundus Multi-disease Image Dataset)

### Temel Teknikler

1. Transfer Learning (ImageNet weights)
2. Discriminative Fine-tuning
3. Learning Rate Range Test
4. Focal Loss (sınıf dengesizliği için)
5. Weighted Random Sampler (oversampling)
6. Data Augmentation (ColorJitter, RandomAffine, RandomErasing)
7. Cosine Annealing LR Scheduler
8. Early Stopping

### Değerlendirme Metrikleri

- Precision, Recall, F1 Score (sınıf bazında)
- Multi-label Confusion Matrix
- ROC Curves ve AUC

---

_Bu dokümantasyon otomatik olarak oluşturulmuştur._

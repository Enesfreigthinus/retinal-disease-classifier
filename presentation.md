# Retinal Disease Classifier - Sunum Taslağı

## 📋 İçindekiler

1. [Giriş ve Problem Tanımı](#1-giriş-ve-problem-tanımı)
2. [Veri Seti](#2-veri-seti)
3. [Model Mimarisi](#3-model-mimarisi)
4. [Veri Ön İşleme ve Augmentation](#4-veri-ön-işleme-ve-augmentation)
5. [Sınıf Dengesizliği ve Çözümler](#5-sınıf-dengesizliği-ve-çözümler)
6. [Eğitim Stratejisi](#6-eğitim-stratejisi)
7. [Sonuçlar ve Performans](#7-sonuçlar-ve-performans)
8. [Demo ve Örnek Tahminler](#8-demo-ve-örnek-tahminler)
9. [Sonuç ve Gelecek Çalışmalar](#9-sonuç-ve-gelecek-çalışmalar)

---

## 1. Giriş ve Problem Tanımı

### Söylenebilecekler:

- **Problem**: Retinal (göz dibi) görüntülerinden hastalık tespiti, oftalmologlar için zaman alıcı ve uzmanlık gerektiren bir süreçtir.
- **Motivasyon**: Diyabetik retinopati, yaşa bağlı makula dejenerasyonu gibi hastalıkların erken tespiti körlüğü önleyebilir.
- **Amaç**: Fundus görüntülerinden otomatik olarak **43 farklı retinal hastalığı** tespit edebilen bir derin öğrenme modeli geliştirmek.
- **Multi-label Classification**: Tek bir görüntüde birden fazla hastalık aynı anda bulunabilir (örn: hem diyabetik retinopati hem makula dejenerasyonu).
- **Klinik Önemi**: Yapay zeka destekli tanı sistemleri, özellikle uzman hekim erişiminin kısıtlı olduğu bölgelerde hayat kurtarıcı olabilir.

### Gösterilebilecek Görseller:

- Normal vs hastalıklı retina görüntüsü karşılaştırması (notebook'ta mevcut)
- 0'dan 5'e kadar hastalık içeren örnek fundus görüntüleri

---

## 2. Veri Seti

### Söylenebilecekler:

- **Kaynak**: RFMiD (Retinal Fundus Multi-disease Image Dataset) - Kaggle
- **Toplam Görüntü Sayısı**: 3200 fundus görüntüsü
  - Eğitim: 1920 görüntü
  - Doğrulama: 640 görüntü
  - Test: 640 görüntü
- **Sınıf Sayısı**: 43 farklı retinal hastalık (HR ve ODPM hariç tutulmuştur)
- **Görüntü Boyutu**: 224x224 piksel (yeniden boyutlandırılmış)

### Tespit Edilen Hastalıklar:

- Diyabetik Retinopati (DR)
- Yaşa Bağlı Makula Dejenerasyonu (ARMD)
- Makula Deliği (MH)
- Diyabetik Nefropati (DN)
- Miyopi (MYA)
- Retinal Ven Tıkanıklığı (BRVO)
- Ve 37 ek hastalık daha...

### Veri Seti Zorlukları:

- **Ciddi Sınıf Dengesizliği**: Bazı hastalıklar çok nadir (10'dan az örnek), bazıları yaygın (500+ örnek)
- **Multi-label Yapı**: Bir görüntüde 0-5+ hastalık bulunabilir

### Gösterilebilecek Görseller:

- Hastalık başına örnek sayısını gösteren bar chart (sınıf dağılımı)
- Görüntü başına hastalık sayısı histogramı
- Farklı sayıda hastalık içeren örnek fundus görüntüleri (0-5 hastalık)

---

## 3. Model Mimarisi

> **Toplam Slide Sayısı: 4 Slide**

---

### 📌 Slide 3.1: Neden ConvNeXt?

#### Slide İçeriği:

- Başlık: "Model Seçimi: Neden ConvNeXt-Tiny?"
- Alt başlık: "Modern CNN vs Geleneksel Yaklaşımlar"

#### Görseller:

- CNN mimarilerinin kronolojik gelişimi görseli (AlexNet → VGG → ResNet → EfficientNet → ConvNeXt)
- ConvNeXt vs ResNet vs ViT performans karşılaştırma tablosu (ImageNet sonuçları)

#### Anlatım Metni:

> "Model seçiminde neden ConvNeXt-Tiny'ı tercih ettik? ConvNeXt, 2022 yılında Facebook AI Research (Meta) tarafından geliştirilmiş ve 'A ConvNet for the 2020s' başlıklı makale ile tanıtılmıştır.
>
> ConvNeXt'in özelliği, Vision Transformer'ların başarısından sonra CNN mimarilerinin hâlâ rekabetçi olabileceğini göstermesidir. Araştırmacılar, ResNet mimarisini modern tasarım prensipleriyle güncellemiş ve ViT ile karşılaştırılabilir performans elde etmişlerdir.
>
> Bizim projemiz için ConvNeXt-Tiny ideal çünkü:
>
> - Görece küçük model boyutu (~28M parametre) ile yüksek performans sunar
> - Transfer learning için optimize edilmiştir
> - Tıbbi görüntü analizinde CNN'ler hâlâ çok etkilidir"

---

### 📌 Slide 3.2: ConvNeXt Mimarisi Detayları

#### Slide İçeriği:

- Başlık: "ConvNeXt-Tiny Mimari Yapısı"
- ConvNeXt bloğu şeması
- Katman detayları tablosu

#### Görseller:

- ConvNeXt block diyagramı (Depthwise Conv → LayerNorm → 1x1 Conv → GELU → 1x1 Conv)
- Stage yapısı görseli (4 stage: 96→192→384→768 kanal)

#### Tablo (Slide'a eklenecek):

| Stage   | Çıkış Boyutu | Kanal Sayısı | Blok Sayısı |
| ------- | ------------ | ------------ | ----------- |
| Stem    | 56×56        | 96           | 1           |
| Stage 1 | 56×56        | 96           | 3           |
| Stage 2 | 28×28        | 192          | 3           |
| Stage 3 | 14×14        | 384          | 9           |
| Stage 4 | 7×7          | 768          | 3           |

#### Anlatım Metni:

> "ConvNeXt mimarisine daha yakından bakalım. Mimari 4 ana stage'den oluşur ve her stage'de özellik haritalarının boyutu yarıya inerken kanal sayısı iki katına çıkar.
>
> ConvNeXt bloğunun temel bileşenleri:
>
> 1. **Depthwise Convolution (7×7)**: Her kanal için ayrı konvolüsyon, hesaplama verimliliği sağlar
> 2. **Layer Normalization**: Batch Norm yerine, daha stabil eğitim
> 3. **Pointwise Convolutions (1×1)**: Kanal etkileşimlerini öğrenir
> 4. **GELU Aktivasyonu**: ReLU'dan daha yumuşak, modern transformerlarda kullanılan aktivasyon
> 5. **Inverted Bottleneck**: Dar→Geniş→Dar yapısı, parametre verimliliği
>
> Bu tasarım, ViT'in başarılı özelliklerini CNN'e adapte eder. Örneğin, büyük kernel boyutu (7×7), transformer'daki geniş attention window'a karşılık gelir."

---

### 📌 Slide 3.3: Transfer Learning ve Fine-Tuning

#### Slide İçeriği:

- Başlık: "Transfer Learning Stratejimiz"
- Transfer learning akış diyagramı
- ImageNet pretraining açıklaması

#### Görseller:

- Transfer learning konsept görseli (ImageNet → Retinal Disease)
- Feature extractor + Classifier ayrımı görseli
- Discriminative Learning Rate grafiği

#### Anlatım Metni:

> "Projemizde transfer learning kullanıyoruz. Peki bu ne anlama geliyor?
>
> ConvNeXt-Tiny modeli önce ImageNet veri seti üzerinde eğitilmiş. ImageNet, 1.2 milyon görüntü ve 1000 sınıf içerir. Bu eğitim sırasında model:
>
> - Kenar, köşe, doku gibi düşük seviyeli özellikler
> - Şekil, pattern gibi orta seviyeli özellikler
> - Nesne parçaları gibi yüksek seviyeli özellikler öğrenmiştir.
>
> Bu öğrenilmiş özellikler retinal görüntüler için de geçerlidir! Damarlar, lekeler, renk değişimleri benzer düşük-orta seviye özelliklerdir.
>
> **Discriminative Fine-Tuning** stratejimiz:
>
> - Feature Extractor katmanları: Düşük learning rate (2e-4) → Öğrenilmiş özellikleri korur
> - Classifier katmanı: Yüksek learning rate (2e-3) → Yeni göreve hızla adapte olur
>
> Bu sayede hem pretrained bilgiyi korur hem de yeni göreve uyum sağlarız."

---

### 📌 Slide 3.4: Bizim Model Konfigürasyonu

#### Slide İçeriği:

- Başlık: "Retinal Disease Classifier: Model Özeti"
- Model akış diyagramı (Input → ConvNeXt → Classifier → Sigmoid → 43 Output)
- Parametre özet tablosu

#### Görseller:

- End-to-end model pipeline görseli
- Multi-label output açıklama görseli (43 bağımsız sigmoid)

#### Tablo (Slide'a eklenecek):

| Parametre            | Değer               |
| -------------------- | ------------------- |
| Backbone             | ConvNeXt-Tiny       |
| Pretrained Weights   | ImageNet-1K         |
| Input Size           | 224 × 224 × 3       |
| Feature Dimension    | 768                 |
| Output Classes       | 43                  |
| Total Parameters     | ~28.6 milyon        |
| Trainable Parameters | ~28.6 milyon        |
| Output Activation    | Sigmoid (per-class) |

#### Anlatım Metni:

> "Şimdi bizim model konfigürasyonumuzu özetleyelim.
>
> **Giriş**: 224×224 piksel boyutunda RGB fundus görüntüsü
>
> **Feature Extraction**: ConvNeXt-Tiny backbone, ImageNet pretrained ağırlıklarla başlatılır. 4 stage boyunca görüntüyü işler ve 7×7×768 boyutunda özellik haritası çıkarır.
>
> **Global Average Pooling**: Özellik haritasını 768 boyutlu vektöre dönüştürür.
>
> **Classifier**: 768→43 boyutlu fully connected layer. Orijinal 1000 sınıflı ImageNet classifier'ı bizim 43 hastalık sınıfımız için değiştirilmiştir.
>
> **Multi-label Output**: Her bir sınıf için bağımsız sigmoid aktivasyonu uygulanır. Bu sayede bir görüntüde birden fazla hastalık aynı anda tespit edilebilir. Örneğin, bir hasta hem diyabetik retinopati hem de makula dejenerasyonuna sahip olabilir.
>
> Threshold değeri 0.5'tir - sigmoid çıktısı 0.5'in üzerindeyse o hastalık 'var' olarak kabul edilir."

---

### 🖼️ Model Mimarisi Bölümü için Hazırlanabilecek Görseller Özeti:

| Görsel No | Açıklama                                  | Nereden Bulunabilir/Nasıl Hazırlanır |
| --------- | ----------------------------------------- | ------------------------------------ |
| 1         | CNN gelişim tarihi (AlexNet→ConvNeXt)     | İnternetten timeline görseli         |
| 2         | ConvNeXt Block Diyagramı                  | Orijinal paper'dan (Figure 2)        |
| 3         | ConvNeXt vs ResNet vs ViT karşılaştırması | Paper'dan accuracy tablosu           |
| 4         | Transfer Learning Konsept                 | Genel infografik                     |
| 5         | Feature Extractor + Classifier ayrımı     | Özel çizim                           |
| 6         | Discriminative LR görseli                 | LR değerlerini gösteren bar chart    |
| 7         | End-to-end pipeline                       | Input→Model→Output akış şeması       |
| 8         | Multi-label output açıklama               | 43 sigmoid çıkışı gösteren diyagram  |

---

### 📚 Model Mimarisi - Kaynak Referanslar:

1. **ConvNeXt Paper**: Liu et al., "A ConvNet for the 2020s", CVPR 2022
   - https://arxiv.org/abs/2201.03545
2. **ImageNet**: Deng et al., "ImageNet: A Large-Scale Hierarchical Image Database", CVPR 2009
3. **Transfer Learning Survey**: Zhuang et al., "A Comprehensive Survey on Transfer Learning", 2020

---

## 4. Veri Ön İşleme ve Augmentation

### Söylenebilecekler:

- **Normalizasyon**: ImageNet istatistikleri kullanıldı (mean, std)
- **Temel Augmentation**:
  - Resize (224x224)
  - Random Rotation (180°)
  - Horizontal/Vertical Flip (%50)
  - Sharpness Adjustment
  - Center Crop
- **Gelişmiş Augmentation**:
  - **ColorJitter**: Parlaklık, kontrast, doygunluk, ton değişimleri
  - **RandomAffine**: Döndürme, öteleme, ölçekleme, eğme
  - **GaussianBlur**: Bulanıklık efekti
  - **Random Erasing (Cutout)**: Rastgele bölge silme - overfitting'e karşı

### Neden Bu Augmentation'lar?

- Retinal görüntüler farklı açılardan çekilebilir → Rotation gerekli
- Aydınlatma koşulları değişken → ColorJitter gerekli
- Küçük veri seti → Daha fazla augmentation = daha iyi genelleme

### Gösterilebilecek Görseller:

- Aynı görüntünün farklı augmentation versiyonları
- Augmentation öncesi vs sonrası karşılaştırması

---

## 5. Sınıf Dengesizliği ve Çözümler

### Söylenebilecekler:

- **Problem**: 43 sınıf arasında ciddi dengesizlik
  - En yaygın sınıf: ~500 örnek
  - En nadir sınıf: <10 örnek
- **Çözümler**:

#### 1. Focal Loss

- Standart BCE loss'un geliştirilmiş versiyonu
- Kolay örnekleri down-weight eder, zor örneklere odaklanır
- Formül: FL(p_t) = -α(1-p_t)^γ log(p_t)
- γ (gamma) = 2.0 kullanıldı

#### 2. Weighted BCE Loss

- Pozitif sınıflara daha yüksek ağırlık verilir
- pos_weight = (negatif örnek sayısı) / (pozitif örnek sayısı)

#### 3. Asymmetric Loss

- Pozitif ve negatif örnekler için farklı gamma değerleri
- Negatif baskın veri setleri için özellikle etkili

#### 4. Weighted Random Sampler

- Her batch'te sınıf dağılımını dengelemek için kullanılır

### Gösterilebilecek Görseller:

- Sınıf frekansı vs F1 score grafiği
- Orijinal vs iyileştirilmiş model F1 karşılaştırması
- Focal Loss vs BCE Loss karşılaştırması

---

## 6. Eğitim Stratejisi

### Söylenebilecekler:

#### Optimizer: AdamW

- Adam + Weight Decay (L2 regularization)
- Weight Decay: 1e-4

#### Learning Rate Stratejisi:

- **LR Finder** kullanılarak optimal LR belirlendi
- **Discriminative Fine-tuning**:
  - Feature extractor: LR = 2e-4 (düşük)
  - Classifier: LR = 2e-3 (yüksek)
  - Önceden öğrenilmiş özellikler korunur, classifier hızla adapte olur

#### Scheduler: Cosine Annealing

- Learning rate zamanla azalarak 1e-6'ya düşer
- Daha yumuşak optimizasyon

#### Eğitim Ayarları:

- Batch Size: 64
- Epoch Sayısı: 30
- Early Stopping Patience: 5
- Gradient Clipping: 1.0 (gradyan patlamasını önler)

### Gösterilebilecek Görseller:

- LR Finder grafiği
- Training vs Validation Loss/Accuracy eğrileri
- Learning Rate schedule grafiği

---

## 7. Sonuçlar ve Performans

### Söylenebilecekler:

#### Test Metrikleri:

| Metrik                      | Değer  |
| --------------------------- | ------ |
| Test Accuracy               | %98.5  |
| Test Loss                   | 0.0494 |
| Öğrenilen Sınıflar (F1 > 0) | 16/43  |
| En İyi Sınıflar F1          | %70-80 |

#### Detaylı Analiz:

- **Yeterli veri olan sınıflar**: F1 score ~%70-80 başarı
- **Nadir sınıflar iki kategoriye ayrılıyor**:
  1. İyi öğrenilen ama düşük recall (precision > recall)
  2. Hiç öğrenilemeyen sınıflar (F1 = 0)
- **Multi-label Performans**: Birden fazla hastalığı başarıyla tespit edebiliyor

#### Sınıf Bazlı Metrikler:

- **Precision**: Model "hastalık var" dediğinde ne kadar doğru?
- **Recall**: Gerçekte hastalık olanların ne kadarı tespit edildi?
- **F1 Score**: Precision ve Recall'un harmonik ortalaması

### Gösterilebilecek Görseller:

- Training/Validation loss ve accuracy eğrileri
- Sınıf bazlı Precision, Recall, F1 bar chart
- Confusion matrix (en yaygın 10 hastalık için)
- Nadir sınıflar için confusion matrix grid
- ROC eğrileri (seçili sınıflar için)
- Model karşılaştırma grafiği (Orijinal vs İyileştirilmiş)

---

## 8. Demo ve Örnek Tahminler

### Söylenebilecekler:

- Eğitilmiş model ile yeni görüntüler üzerinde tahmin yapılabilir
- `inference.py` ile tek görüntü veya toplu tahmin desteği
- Threshold = 0.5 kullanılarak multi-label tahminler

### Demo Akışı:

1. Örnek bir fundus görüntüsü yükle
2. Model tahminlerini göster
3. Olasılık değerleri ile birlikte tespit edilen hastalıkları listele

### Kullanım:

```bash
# Tek görüntü tahmini
python inference.py --image ./test_image.png --model ./outputs/checkpoints/best_model.pth

# Toplu tahmin
python inference.py --folder ./test_images/ --output predictions.csv
```

### Gösterilebilecek Görseller:

- Örnek tahmin sonuçları (görüntü + tespit edilen hastalıklar)
- Olasılık çıktıları ile birlikte örnek tahminler

---

## 9. Sonuç ve Gelecek Çalışmalar

### Söylenebilecekler:

#### Başarılar:

- ✅ 43 sınıflı multi-label classification başarıyla gerçekleştirildi
- ✅ Transfer learning ile sınırlı veriyle yüksek performans
- ✅ Focal Loss ile sınıf dengesizliği kısmen aşıldı
- ✅ %98.5 test accuracy elde edildi
- ✅ Modüler, yeniden kullanılabilir kod yapısı

#### Limitasyonlar:

- ⚠️ Nadir sınıflarda düşük recall
- ⚠️ Bazı çok nadir hastalıklar (< 10 örnek) öğrenilemiyor
- ⚠️ Daha fazla veri ile performans artırılabilir

#### Gelecek Çalışmalar:

- 📈 Veri artırma: Daha fazla etiketli veri toplama
- 📈 Class-aware sampling stratejileri
- 📈 Knowledge distillation
- 📈 Ensemble modeller
- 📈 Explainability: Grad-CAM ile hastalık bölgelerini görselleştirme
- 📈 Çapraz doğrulama (Cross-validation)
- 📈 Daha büyük ConvNeXt modelleri (Small, Base)

### Gösterilebilecek Görseller:

- Proje özet tablosu
- Gelecek çalışmalar için yol haritası

---

## 📊 Notebookta Mevcut Önemli Görseller

| Görsel Tipi          | Açıklama                         | Notebook Hücresi |
| -------------------- | -------------------------------- | ---------------- |
| Sınıf Dağılımı       | Hastalık başına örnek sayısı     | Cell #12-13      |
| Örnek Görüntüler     | 0-5 hastalık içeren fundus       | Cell #14-17      |
| Training Curves      | Loss ve Accuracy grafikleri      | Cell #78         |
| F1 Karşılaştırması   | Orijinal vs İyileştirilmiş model | Cell #50         |
| Confusion Matrix     | Top-10 sınıflar için             | Cell #80         |
| Nadir Sınıf CM       | En az örneği olan 10 sınıf       | Cell #52         |
| Sınıf Frekansı vs F1 | İyileşme analizi                 | Cell #50         |
| Metrik Bar Chart     | Precision, Recall, F1            | Cell #75         |

---

## 🛠️ Teknoloji Yığını

| Kategori       | Teknoloji                                 |
| -------------- | ----------------------------------------- |
| Framework      | PyTorch                                   |
| Model          | ConvNeXt-Tiny (ImageNet pretrained)       |
| Loss Functions | Focal Loss, Weighted BCE, Asymmetric Loss |
| Optimizer      | AdamW + Discriminative LR                 |
| Scheduler      | Cosine Annealing                          |
| Ortam          | Google Colab (GPU) / Local                |
| Veri İşleme    | torchvision transforms                    |
| Görselleştirme | matplotlib, seaborn                       |
| Metrikler      | sklearn, custom metrics                   |

---

## 📚 Kaynaklar

1. **RFMiD Dataset**: https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification
2. **ConvNeXt Paper**: Liu et al., "A ConvNet for the 2020s", CVPR 2022
3. **Focal Loss Paper**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
4. **Asymmetric Loss Paper**: Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021

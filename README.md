# Ev Fiyat Tahmini (House Price Prediction)

## Proje Hakkında
Bu proje, makine öğrenmesi algoritmaları kullanılarak ev fiyatlarını tahmin etmeyi (regression) hedefleyen uçtan uca bir veri bilimi projesidir. Veri yükleme, temizleme, keşifçi veri analizi (EDA), model eğitimi ve model performansının görselleştirilmesi gibi klasik bir makine öğrenmesi pipeline'ının tüm aşamalarını içermektedir.

## Proje Yapısı ve İş Akışı
Projenin iş akışı birbirini takip eden 5 ana Jupyter Notebook dosyasından oluşmaktadır. Her bir adım, veri bilimi sürecindeki farklı bir aşamayı temsil eder:

1. **`Dataload.ipynb`**:
   - `home_price.csv` adlı ham veri setinin sisteme yüklenmesi.
   - Verinin genel yapısının, kolon tiplerinin ve ilk gözlemlerin incelenmesi.

2. **`dataPreprocessing.ipynb`**:
   - Ham veri üzerindeki eksik verilerin (missing values), gürültülerin veya aykırı değerlerin (outliers) tespit edilip temizlenmesi.
   - Kategorik değişkenlerin dönüştürülmesi ve özellik mühendisliği (feature engineering) uygulanması.
   - İşlenen veriler `clean_data.csv` adı altında modele hazır bir şekilde kaydedilir.

3. **`graphic.ipynb`**:
   - Keşifçi Veri Analizi (Exploratory Data Analysis) aşamasıdır.
   - Matplotlib, Seaborn ve Missingno gibi kütüphaneler ile verideki dağılımlar, korelasyonlar ve eksik veri yapıları kapsamlı bir şekilde görselleştirilir. Değişkenlerin hedef değişken (fiyat) üzerindeki etkileri incelenir.

4. **`model.ipynb`**:
   - Model eğitim aşamasıdır. Veriler eğitim (train) ve test algoritmaları (test_split) olarak ayrılır.
   - `StandardScaler` ile ölçeklendirme gibi ön hazırlıkların ardından birden fazla regresyon modeli eğitilir:
     - Linear Regression
     - Decision Tree Regressor
     - Random Forest Regressor
     - Gradient Boosting Regressor
     - XGBoost Regressor
     - Support Vector Regressor (SVR)
   - Modellerin analizi için `MAE`, `MSE` ve `R2 Score` gibi regresyon değerlendirme metrikleri hesaplanır.

5. **`model_graphics.ipynb`**:
   - Eğitilen modellerin ürettiği tahminlerin anlaşılır kılınması, karar ağaçlarının (decision tree vb.) görselleştirilmesi işlemlerini içerir.
   - Hata dağılımlarının, tahmin ve gerçek değer kıyaslamalarının grafiklere döküldüğü ve çalışmanın sonuçlandığı aşamadır.

## Kullanılan Teknolojiler ve Kütüphaneler
- **Veri Manipülasyonu:** Pandas, NumPy
- **Makine Öğrenmesi Algoritmaları:** Scikit-Learn, XGBoost
- **Veri Görselleştirme:** Matplotlib, Seaborn, Missingno
- **Geliştirme Ortamı:** Jupyter Notebook

## Nasıl Çalıştırılır?
Projeyi kullanabilmek ve notebook aşamalarını inceleyebilmek için aşağıdaki kütüphanelerin Python ortamınızda (tercihen sanal ortam) kurulu olması gerekmektedir:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn missingno jupyter
```

İndirilen veri setleri projeye dahilse (örneğin `home_price.csv`), notebook'ları sırasıyla `Dataload.ipynb` -> `dataPreprocessing.ipynb` -> `graphic.ipynb` -> `model.ipynb` -> `model_graphics.ipynb` şeklinde çalıştırarak aşamaları gerçek zamanlı görüntüleyebilir ve sonuçları kendi bilgisayarınızda test edebilirsiniz.


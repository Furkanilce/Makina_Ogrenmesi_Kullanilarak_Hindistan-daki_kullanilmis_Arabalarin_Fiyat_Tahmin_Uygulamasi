# 🚗 Hindistan'da İkinci El Araç Fiyat Tahmin Uygulaması

Bu proje, Hindistan’daki ikinci el araçların fiyatlarını tahmin etmek için **Makine Öğrenmesi** ve **Streamlit** kullanılarak geliştirilmiştir.  
Amaç, kullanıcıların araç bilgilerini girdiklerinde tahmini fiyatı görebilecekleri bir web arayüzü sağlamaktır.

---

## 📖 Makale Link
[Predicting Used Car Prices using Machine Learning Models - IEEE Xplore](https://ieeexplore.ieee.org/document/11076357)

## 📂 Veriseti Link
[Used Cars Dataset - Kaggle](https://www.kaggle.com/datasets/sukritchatterjee/used-cars-dataset-cardekho)

---

## 🧰 Kullanılan Teknolojiler

Python 3.x

Pandas & NumPy → Veri işleme

Scikit-learn → Veri ön işleme & modelleme

XGBoost → Regresyon modeli

Category Encoders → Kategorik değişken kodlama

Joblib → Model kaydetme & yükleme

Streamlit → Web arayüzü


---

## 📜 Lisans

Bu proje eğitim amaçlıdır. Veri seti Kaggle üzerinden, makale ise IEEE Xplore üzerinden erişilebilir.
Bu proje [MIT Lisansı](./LICENSE) ile lisanslanmıştır.

---
## ⚙️ Kurulum

1. Depoyu klonlayın:
   ```bash
   git clone https://github.com/KULLANICI_ADI/CarPricePrediction.git
   cd CarPricePrediction
2. Gerekli kütüphaneleri yükleyin:

```bash
  pip install -r requirements.txt
```
3. (Opsiyonel) Modeli yeniden eğitmek için:

```bash
python train_model.py
```

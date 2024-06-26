
# 🌟 Contrast Equalization for Image Processing 📊

Bu Python programı, gri tonlu görüntüler üzerinde kontrast eşitleme işlemi uygulayan interaktif bir araçtır. Program, matplotlib kütüphanesi kullanılarak görselleştirme ile birlikte gelir.

📄 **İçindekiler**
- [Nasıl Çalışır?](#-nasıl-çalışır)
- [Kurulum](#-kurulum)
- [Özellikler](#-özellikler)
- [Ek Bilgiler](#-ek-bilgiler)
- [Katılım](#-katılım)

## 🎯 **Nasıl Çalışır?**
### **Başlangıç:**
Program, bir Excel dosyasından görüntü verisini yükleyerek başlar.

### **Kontrast Eşitleme:**
- **Histogram ve CDF Hesaplama:**
  Görüntü verisi histogramı ve kümülatif dağılım fonksiyonu (CDF) hesaplanır.
- **Piksel Yoğunluklarının Eşitlenmesi:**
  Elde edilen CDF kullanılarak piksel yoğunlukları eşitlenir.

### **Sonuçların Görselleştirilmesi:**
- **Orijinal ve Eşitlenmiş Görüntüler:**
  Orijinal ve eşitlenmiş görüntüler yan yana gösterilir.
- **Histogramlar ve CDF:**
  Orijinal ve eşitlenmiş görüntülerin histogramları ve CDF grafiği gösterilir.

## 🛠️ **Kurulum**
### **Python Kurulumu:**
Python'u bilgisayarınıza yükleyin (eğer yüklü değilse): [Python İndirme Sayfası](https://www.python.org/downloads/)

### **Projeyi Başlatma:**
Projeyi bilgisayarınıza indirin veya kopyalayın.

### **Gerekli Kütüphanelerin Kurulumu:**
Gerekli kütüphaneleri yüklemek için terminal veya komut istemcisinde aşağıdaki komutu çalıştırın:
```bash
pip install pandas numpy matplotlib openpyxl
```

### **Oyunu Başlatma:**
Terminal veya komut istemcisinde aşağıdaki komutu çalıştırın:
```bash
python contrast_equalization.py
```

## 🚀 **Özellikler**
- **Görüntü Verisi Yükleme:** Excel dosyasından görüntü verisi yükleyebilme.
- **Kontrast Eşitleme:** Gri tonlu görüntülerde kontrast eşitleme işlemi.
- **Görselleştirme:** Orijinal ve eşitlenmiş görüntülerin, histogramların ve CDF grafiğinin görselleştirilmesi.
- **Sonuçların Kaydedilmesi:** Sonuçların Excel dosyasına kaydedilmesi.



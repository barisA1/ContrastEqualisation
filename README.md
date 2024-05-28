# ContrastEqualisation
# 📊 Contrast Stretching Script

Bu Python scripti, bir Excel dosyasından veri okuyup contrast stretching işlemi uygulamak için kullanılabilir.

## 📄 İçindekiler
- Gereksinimler
- Kullanım
- Fonksiyonlar
- Örnek
- İletişim

## 🛠️ Gereksinimler

Bu scriptin çalışabilmesi için aşağıdaki Python kütüphanelerinin yüklü olması gerekmektedir:

- pandas
- numpy
- matplotlib
- openpyxl (pandas'ın Excel dosyalarını okuyabilmesi için)

Bu kütüphaneleri aşağıdaki komutla yükleyebilirsiniz:

```bash
pip install pandas numpy matplotlib openpyxl
```

## 🎯 Kullanım

### 1. Dosya Yolu Ayarlaması

`main.py` dosyasında, verilerin okunacağı Excel dosyasının yolunu belirtin. Örneğin:

```python
file_path = r'C:\\Users\\Barış\\Desktop\\soru1_2_data.xlsx'
```

### 2. Scripti Çalıştırma

Aşağıdaki komutla scripti çalıştırın:

```bash
python main.py
```

### 3. Sonuçların Görüntülenmesi

Script çalıştırıldıktan sonra, contrast stretching işlemi uygulanmış veriler terminalde görüntülenecektir. Ek olarak, işlem sonucu oluşturulan grafikler de görüntülenecektir.

## 📑 Fonksiyonlar

### contrast_equalization(data, L=256)

Bu fonksiyon, verilen veri üzerinde contrast stretching işlemi uygular.

- **data**: Numpy array formatında veri.
- **L**: Çıkış verisinin maksimum değeri (varsayılan 256).

Fonksiyon, işlem uygulanmış veriyi numpy array formatında döner.

## 📝 Örnek

Örnek bir Excel dosyasının nasıl yükleneceği ve işleneceği aşağıda gösterilmiştir:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File path
file_path = r'C:\\Users\\Barış\\Desktop\\soru1_2_data.xlsx'

# Load the file
df = pd.read_excel(file_path)

# Convert data to numpy array
data = df.to_numpy()

# Contrast Equalization function
def contrast_equalization(data, L=256):
    # Compute histogram and CDF
    hist, bins = np.histogram(data.flatten(), bins=L, range=[0, L])
    cdf = hist.cumsum()
    cdf_normalized = cdf * (L - 1) / cdf[-1]

    # Compute new intensity values
    equalized_data = np.interp(data.flatten(), bins[:-1], cdf_normalized).reshape(data.shape)

    return equalized_data, hist, cdf_normalized, bins[:-1]

# Apply Contrast Equalization
equalized_data, hist, cdf_normalized, bins = contrast_equalization(data)

with np.printoptions(precision=2, suppress=True):
    print("Contrast Equalization Sonucu:\n", equalized_data)

# Display Histogram and CDF table
hist_cdf_df = pd.DataFrame({'Pixel Value': bins, 'Frequency': hist, 'CDF': cdf_normalized})
print(hist_cdf_df)

# Display Images and Plots
plt.figure(figsize=(18, 12))

# Original Image
plt.subplot(3, 2, 1)
plt.title('Original Image')
plt.imshow(data, cmap='gray', vmin=0, vmax=255)
plt.colorbar()

# Equalized Image
plt.subplot(3, 2, 2)
plt.title('Equalized Image')
plt.imshow(equalized_data, cmap='gray', vmin=0, vmax=255)
plt.colorbar()

# Original Histogram
plt.subplot(3, 2, 3)
plt.title('Original Histogram')
plt.hist(data.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)

# Equalized Histogram
plt.subplot(3, 2, 4)
plt.title('Equalized Histogram')
plt.hist(equalized_data.flatten(), bins=256, range=[0, 256], color='green', alpha=0.7)

# CDF
plt.subplot(3, 2, 5)
plt.title('CDF')
plt.plot(cdf_normalized, color='red')

plt.tight_layout()
plt.show()
```

## ℹ️ Ek Bilgiler

Bu script, özellikle görüntü işleme ve veri analizi projelerinde kullanılmak üzere tasarlanmıştır. Excel dosyanızdaki verileri contrast stretching ile dönüştürerek daha iyi analiz edilebilir hale getirebilirsiniz.

## 🤝 Katılım

Projeye katkıda bulunmak isterseniz:

1. Bu depoyu çatallayın (fork) ve geliştirmelerinizi yapın.
2. Yeni özellikler eklemek veya hataları düzeltmek için Pull Talepler (Pull Requests) gönderin.
3. Hataları bildirmek veya önerilerde bulunmak için konu (issue) açın.

## 📬 İletişim

Herhangi bir soru veya öneriniz için lütfen [aktasb723@gmail.com](mailto:aktasb723@gmail.com) e-posta adresi ile iletişime geçin.
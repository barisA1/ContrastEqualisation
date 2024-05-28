import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dosya yolunu tam olarak belirtiyoruz
file_path = r'C:\Users\Barış\Desktop\soru1_2_data.xlsx'

# Dosyayı yükle
df = pd.read_excel(file_path)

# Veriyi numpy array'e çevir
data = df.to_numpy()


# Contrast Equalization fonksiyonu
def contrast_equalization(data, L=256):
    # Histogram ve CDF hesapla
    hist, bins = np.histogram(data.flatten(), bins=L, range=[0, L])
    cdf = hist.cumsum()
    cdf_normalized = cdf * (L - 1) / cdf[-1]

    # Yeni yoğunluk değerlerini hesapla
    equalized_data = np.interp(data.flatten(), bins[:-1], cdf_normalized).reshape(data.shape)

    return equalized_data, hist, cdf_normalized, bins[:-1]


# Contrast Equalization uygulaması
equalized_data, hist, cdf_normalized, bins = contrast_equalization(data)
print("Contrast Equalization Sonucu:\n", equalized_data)

# Histogram ve CDF tablosunu göster
hist_cdf_df = pd.DataFrame({'Pixel Value': bins, 'Frequency': hist, 'CDF': cdf_normalized})
print(hist_cdf_df)

# Ara Tabloları ve Matris Görüntülerini Göster
plt.figure(figsize=(18, 12))

# Orijinal Görüntü
plt.subplot(3, 2, 1)
plt.title('Original Image')
plt.imshow(data, cmap='gray', vmin=0, vmax=255)
plt.colorbar()

# Eşitlenmiş Görüntü
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

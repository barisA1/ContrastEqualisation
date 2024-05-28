import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dosya yolunu tam olarak belirtiyoruz
file_path = r'C:\Users\Barış\Desktop\soru1_2_data.xlsx'

# Dosyayı yükle
df = pd.read_excel(file_path)

# Veriyi numpy array'e çevir
data = df.to_numpy()

# L değeri (gri ton seviyeleri)
L = 256

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
equalized_data, hist, cdf_normalized, bins = contrast_equalization(data, L)

# Piksel değerleri, frekanslar, CDF ve eşitlenmiş piksel değerlerini tabloya dönüştürme
original_values = data.flatten()
equalized_values = equalized_data.flatten()

# Piksel değerlerinin benzersiz ve sıralanmış bir listesini oluşturma
unique_values = np.sort(np.unique(original_values))

# Frekans, CDF ve eşitlenmiş piksel değerleri hesaplama
frequencies = [np.sum(original_values == value) for value in unique_values]
cdfs = [np.sum(frequencies[:i+1]) for i in range(len(frequencies))]
equalized_values_unique = [round(((cdf - cdfs[0]) / (len(original_values) - cdfs[0])) * (L - 1)) for cdf in cdfs]

# Tablo oluşturma
results_df = pd.DataFrame({
    'v, Pixel Intensity': unique_values,
    'Frequency': frequencies,
    'cdf(v)': cdfs,
    'h(v), Equalized v': equalized_values_unique
})

# Tablonun gösterilmesi
print(results_df)

# Tablonun Excel dosyasına kaydedilmesi
output_file_path = r'C:\Users\Barış\Desktop\contrast_equalization_results.xlsx'
results_df.to_excel(output_file_path, index=False)
print(f'Tablo başarıyla {output_file_path} dosyasına kaydedildi.')

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

# Orijinal Histogram
plt.subplot(3, 2, 3)
plt.title('Original Histogram')
plt.hist(data.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)

# Eşitlenmiş Histogram
plt.subplot(3, 2, 4)
plt.title('Equalized Histogram')
plt.hist(equalized_data.flatten(), bins=256, range=[0, 256], color='green', alpha=0.7)

# CDF
plt.subplot(3, 2, 5)
plt.title('CDF')
plt.plot(cdf_normalized, color='red')

plt.tight_layout()
plt.show()

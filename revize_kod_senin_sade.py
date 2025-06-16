import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import t, chi2

# Rastgele veri üretimi (seyirci sayıları)
np.random.seed(42)
veri_sayisi = 500
veri_cercevesi = pd.DataFrame()
veri_cercevesi['seyirci_sayisi'] = np.random.normal(35000, 7000, veri_sayisi).clip(5000, 80000).astype(int)

seyirci_verisi = veri_cercevesi['seyirci_sayisi']
ortalama = sum(seyirci_verisi) / veri_sayisi

# Medyan hesaplama
sirali_veri = sorted(seyirci_verisi)
if veri_sayisi % 2 == 0:
    medyan = (sirali_veri[veri_sayisi//2 - 1] + sirali_veri[veri_sayisi//2]) / 2
else:
    medyan = sirali_veri[veri_sayisi//2]

# Varyans, standart sapma ve standart hata hesaplama
varyans = sum((x - ortalama) ** 2 for x in seyirci_verisi) / (veri_sayisi - 1)
standart_sapma = varyans ** 0.5
standart_hata = standart_sapma / math.sqrt(veri_sayisi)

# %95 Güven aralığı (Ortalama için)
t_kritik = t.ppf(0.975, df=veri_sayisi-1)
ortalama_guven_alt = ortalama - t_kritik * standart_hata
ortalama_guven_ust = ortalama + t_kritik * standart_hata

# %95 Güven aralığı (Varyans için)
chi2_alt = chi2.ppf(0.025, df=veri_sayisi-1)
chi2_ust = chi2.ppf(0.975, df=veri_sayisi-1)
varyans_guven_alt = (veri_sayisi - 1) * varyans / chi2_ust
varyans_guven_ust = (veri_sayisi - 1) * varyans / chi2_alt

# Örneklem büyüklüğü hesabı (%90 güven, 0.1 hata)
z_degeri = 1.645
hata_payi = 0.1
gerekli_orneklem_sayisi = math.ceil((z_degeri * standart_sapma / hata_payi) ** 2)

# Hipotez testi: Ortalama 40.000 mi?
populasyon_ortalama = 40000
t_istatistigi = (ortalama - populasyon_ortalama) / standart_hata
p_degeri = 2 * (1 - t.cdf(abs(t_istatistigi), df=veri_sayisi - 1))
hipotez_karar = "H0 reddedildi. Ortalama 40.000'den farklıdır." if p_degeri < 0.05 else "H0 kabul edilir. Ortalama 40.000 olabilir."

# Sonuçlar
print(f"Ortalama: {ortalama:.2f}")
print(f"Medyan: {medyan:.2f}")
print(f"Varyans: {varyans:.2f}")
print(f"Standart Sapma: {standart_sapma:.2f}")
print(f"Standart Hata: {standart_hata:.2f}")
print(f"\nOrtalama için %95 Güven Aralığı: [{ortalama_guven_alt:.2f}, {ortalama_guven_ust:.2f}]")
print(f"Varyans için %95 Güven Aralığı: [{varyans_guven_alt:.2f}, {varyans_guven_ust:.2f}]")
print(f"\n0.1 hata payı ile gerekli minimum örneklem büyüklüğü: {gerekli_orneklem_sayisi}")
print(f"\nHipotez Testi: Ortalama 40.000 mi?")
print(f"t-istatistiği: {t_istatistigi:.4f}")
print(f"p-değeri: {p_degeri:.5f}")
print(f"Karar: {hipotez_karar}")

# Grafikler
plt.figure(figsize=(14, 6))

# Histogram
plt.subplot(1, 2, 1)
plt.hist(seyirci_verisi, bins=30, color='skyblue', edgecolor='black')
plt.title("Seyirci Sayısı Dağılımı (Histogram)")
plt.xlabel("Seyirci Sayısı")
plt.ylabel("Maç Sayısı")
plt.grid(True, axis='y', linestyle='--', alpha=0.6)

# Boxplot
plt.subplot(1, 2, 2)
plt.boxplot(seyirci_verisi, vert=False, patch_artist=True, boxprops=dict(facecolor='orange'))
plt.title("Seyirci Sayısı Dağılımı (Boxplot)")
plt.xlabel("Seyirci Sayısı")

plt.tight_layout()
plt.show()


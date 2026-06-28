# IoB Tabanlı Hava Durumu Analizleriyle Gıda Sektöründe Stratejik Müşteri Davranışı Yönetimi

Bu proje, **Internet of Behaviors (IoB)** yaklaşımını kullanarak lokanta müşterilerinin davranışlarını hava durumu verileriyle ilişkilendirmeyi amaçlar.

**Güncel kapsam:** Yalnızca **2021 ve sonrası** dönem.

## Proje Yapısı

```
proje_ana_dizini/
├── eski_analizler_2012_2025/          # Eski 2012-2025 analiz dosyaları (arsiv)
├── 00_veri_hazirlama_2021_sonrasi.ipynb
├── 01_descriptive_analiz.ipynb
├── 02_zaman_serisi_analizi.ipynb
├── 03_korelasyon_ve_regresyon.ipynb
├── 04_birliktelik_kurallari.ipynb
├── Veriler/
│   ├── oturum_hava_birlesik_2021_ve_sonrasi.csv   # Ham birlesik veri
│   └── oturum_hava_temiz_2021_sonrasi.csv         # Temizlenmis veri (00 ciktisi)
├── Outputs/
│   └── Veri_Hazirlama/                           # Veri hazirlama raporlari
└── rapor/
    └── bulgular.md
```

## Calisma Sirasi

1. `00_veri_hazirlama_2021_sonrasi.ipynb` — veri dogrulama, temizleme, ozellik uretimi
2. `01_descriptive_analiz.ipynb` — betimsel analiz
3. `02_zaman_serisi_analizi.ipynb` — zaman serisi
4. `03_korelasyon_ve_regresyon.ipynb` — korelasyon ve regresyon
5. `04_birliktelik_kurallari.ipynb` — birliktelik kurallari

## Veri Hazirlama

Ham girdi: `Veriler/oturum_hava_birlesik_2021_ve_sonrasi.csv`

`00` notebook'u calistirildiginda uretilen ciktilar:

| Dosya | Aciklama |
|-------|----------|
| `Veriler/oturum_hava_temiz_2021_sonrasi.csv` | Genel analiz (120.749 kayit, outlier_flag=NaN) |
| `Veriler/oturum_hava_uzun_oturum_2021_sonrasi.csv` | Uzun oturum incelemesi (9.037 kayit) |
| `Veriler/oturum_hava_tum_2021_sonrasi.csv` | Tum kayitlar (flag'li) |
| `Outputs/Veri_Hazirlama/data_quality_report.csv` | Sutun kalite raporu |
| `Outputs/Veri_Hazirlama/enflasyon_kontrolu_yillik.csv` | TÜFE deflate kontrolu |
| `Outputs/Veri_Hazirlama/kpi_ozet.json` | KPI ozeti |

Veri hazirlama adimlari: outlier ayrimi, TÜFE deflate, ruzgar/nem/yagis/bulut gruplari, mevsim/covid/gun tipi/oglen-aksam turetimi.

## Teknolojiler

- Python, pandas, Jupyter Notebook
- Open-Meteo Archive API (hava verisi)
- scikit-learn, scipy (analiz notebook'lari)

Proje, TÜBİTAK 2209-A kapsamında geliştirilmektedir.

## Lisans

Eğitim ve araştırma amaçlıdır. Kaynak gösterilerek kullanılabilir.

Son güncelleme: 2026

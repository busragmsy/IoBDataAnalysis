# Post-2021 Yeni Analiz Yapisi

Bu klasor, projenin IoB kapsamini koruyarak 2021 ve sonrasi veriler icin sifirdan kurulan yeni analiz omurgasini icerir.

## Dosyalar

- `YONTEM_LISTESI.md`: Kullanilan yontemlerin listesi ve sutun-kapsam haritasi.
- `post2021_kapsamli_pipeline.py`: Tum analizleri tek akista ureten ana pipeline.

## Cikti Formati

Pipeline, coklu CSV yerine tek bir analiz paketi uretir:

- `Outputs/Post2021_YeniYapi/post2021_analiz_paketi.sqlite`
- `Outputs/Post2021_YeniYapi/post2021_analiz_ozet.json`

SQLite dosyasi icinde veri kalite, KPI, parametre tabanli gruplamalar, etki skorlar, model metrikleri ve urun analizi tablolari bulunur.

## Calistirma

Proje kokunden:

```powershell
c:/Users/BUSRA/Documents/GitHub/IoBDataAnalysis/.venv/Scripts/python.exe .\post2021_yeni_yapi\post2021_kapsamli_pipeline.py
```

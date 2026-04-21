# Yontem Listesi (2021 Sonrasi Yeni Yapi)

## 1. Kapsam ve Analitik Cati

Bu yeni yapi, projenin temel kapsamina uygun olarak su IoB eksenlerini birlikte ele alir:

1. Musteri tercih davranisi: `masa_grup` (Bahce, Ic Salon, Paket)
2. Oturum davranisi: sure, saat, gun, donemsellik
3. Harcama davranisi: `toplam_tutar`, `toplam_miktar`, `urun_sayisi`, `urun_listesi`
4. Hava etkisi: sicaklik, nem, yagis, ruzgar, bulut, basinc, radyasyon
5. Veri guvenilirligi: outlier kontrolu, bos veri, tekillik, zaman butunlugu

## 2. Yontem Seti

1. Veri kalite denetimi: bosluk oranlari, tekillik, tip kontrolu, ornek deger dogrulamasi
2. Zaman butunlugu kontrolu: 2021+ donem kapsami, gun kapsama oranlari, acik/kapali gun izi
3. Parametre tabanli gruplama: hava ve zaman parametrelerine gore tercih payi
4. Coklu hedef davranis ozetleri: tercih payi + oturum sure + tutar + miktar + urun sayisi
5. Iliski gucu analizi: Chi-square + Cramer's V
6. Dagilim fark testi: Kruskal-Wallis (oturum sure, tutar, miktar, urun sayisi)
7. Tahminleme modeli (siniflandirma): `masa_grup` tahmini
8. Tahminleme modeli (regresyon): `oturum_sure_dk` ve `toplam_tutar` tahmini
9. Aciklanabilirlik: feature importance cikarimi
10. Urun metin analizi: `urun_listesi` parse + urun frekans/bağlam iliskisi
11. Tek paket cikti: tum analiz tablolarini tek SQLite dosyasina yazma

## 3. Sutun-Kullanim Haritasi

### Kimlik ve oturum

1. `MASANO`: masa bazli tekrar eden davranis ve masa yogunlugu
2. `CEKNO`: oturum/cek benzersizlik kontrolu ve gunluk oturum adedi
3. `acilis_datetime`: oturum baslangic zamani, saat dilimi davranisi
4. `kapama_datetime`: oturum bitis zamani, sure tutarlilik kontrolu
5. `oturum_sure_dk`: temel davranis hedefi ve modelleme hedefi

### Harcama ve sepet

1. `toplam_miktar`: sepet miktar davranisi
2. `toplam_tutar`: gelir davranisi ve modelleme hedefi
3. `urun_sayisi`: cesitlilik ve sepet genisligi
4. `urun_listesi`: urun bazli metin analizi ve tercih baglami

### Tercih ve zaman

1. `masa_grup`: ana tercih hedef degiskeni
2. `tarih`: gunluk konsolidasyon, acik/kapali gun izi
3. `saat`: gun icindeki davranis degisimi
4. `gun_adi`: hafta ici/sonu davranis farki
5. `ay`: mevsimsel tercih degisimi
6. `yil`: donem filtresi ve kapsam kontrolu
7. `hafta_no`: kisa donemsel dalga analizi
8. `merge_saati`: hava-oturum eslesme kalite kontrolu

### Hava degiskenleri

1. `temperature_2m`: temel sicaklik etkisi
2. `apparent_temperature`: hissedilen sicaklik etkisi
3. `relative_humidity_2m`: nem etkisi
4. `dewpoint_2m`: hissedilen rutubet konforu
5. `precipitation`: toplam yagis etkisi
6. `rain`: yagmur siddeti etkisi
7. `showers`: saganak etkisi
8. `snowfall`: kar etkisi
9. `windspeed_10m`: ruzgar siddeti etkisi
10. `winddirection_10m`: ruzgar yonu etkisi
11. `cloudcover`: bulutluluk etkisi
12. `pressure_msl`: basinç kosulu etkisi
13. `is_day`: gunduz/gece baglami
14. `shortwave_radiation`: guneslenme/radyasyon etkisi
15. `yagis_kategori`: yorumsal yagis sinifi
16. `sicaklik_aralik`: yorumsal sicaklik sinifi

### Veri guvenilirligi

1. `outlier_flag`: temiz/aykiri ayrimi, hassasiyet analizi

## 4. Cikti Tasarimi

Ciktilar tek analiz paketinde toplanir:

1. `post2021_analiz_paketi.sqlite` icindeki tablolar:
   - `column_usage`
   - `data_quality`
   - `kpi_summary`
   - `daily_summary`
   - `parameter_behavior_long`
   - `dominant_preferences`
   - `association_scores`
   - `kruskal_scores`
   - `model_metrics`
   - `model_feature_importance`
   - `product_top_overall`
   - `product_top_context`
2. `post2021_analiz_ozet.json`: yonetici ozeti ve ana metrikler

## 5. Neden Bu Yapi?

1. Projenin ana kapsami olan hava x musteri davranisi iliskisini korur.
2. Tum sutunlarin en az bir anlamli analizde kullanilmasini garanti eder.
3. Coklu dosya daginikligini azaltir, sorgulanabilir tek bir paket uretir.
4. Sunum, raporlama ve tekrar uretilebilirlik icin uygun bir omurga saglar.

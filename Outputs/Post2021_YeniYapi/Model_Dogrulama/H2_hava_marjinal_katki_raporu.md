# Is Paketi 4- Hava Durumunun Marjinal Katkisi ve Ek Sizinti Kontrolu

Bu rapor `is_paketi5_hava_katkisi.py` script'i tarafindan otomatik uretilmistir.

## masa_grup: 4 Model Karsilastirmasi (zaman sirasina saygili holdout)

| variant        |   holdout_accuracy |   holdout_f1_macro |   holdout_auc_ovr_macro |   holdout_test_size |
|:---------------|-------------------:|-------------------:|------------------------:|--------------------:|
| weather_only   |           0.48075  |           0.448526 |                0.647475 |               26650 |
| behavior_only  |           0.543415 |           0.471997 |                0.722856 |               26650 |
| full_model     |           0.612158 |           0.60886  |                0.792191 |               26650 |
| naive_baseline |           0.297974 |           0.153046 |              nan        |               26650 |

## oturum_sure_dk: Es-zamanli Sonuc Degiskenleri Testi

| variant                     |   holdout_r2 |   holdout_rmse |   holdout_mae |   rmse_improvement_over_baseline_pct |   durbin_watson | durbin_watson_yorum                                  |
|:----------------------------|-------------:|---------------:|--------------:|-------------------------------------:|----------------:|:-----------------------------------------------------|
| with_concurrent_outcomes    |     0.39349  |        18.5029 |       12.6598 |                              52.8285 |         1.87952 | onemli bir otokorelasyon belirtisi yok (~2'ye yakin) |
| without_concurrent_outcomes |    -0.129031 |        25.2449 |       16.4375 |                              35.6403 |         1.84237 | onemli bir otokorelasyon belirtisi yok (~2'ye yakin) |

## urun_sayisi: Es-zamanli Sonuc Degiskenleri Testi (H2 ana hedefi)

| variant                     |   holdout_r2 |   holdout_rmse |   holdout_mae |   rmse_improvement_over_baseline_pct |   durbin_watson | durbin_watson_yorum                                  |
|:----------------------------|-------------:|---------------:|--------------:|-------------------------------------:|----------------:|:-----------------------------------------------------|
| with_concurrent_outcomes    |     0.735615 |        1.05708 |      0.742942 |                              48.6405 |         1.98424 | onemli bir otokorelasyon belirtisi yok (~2'ye yakin) |
| without_concurrent_outcomes |     0.20293  |        1.83543 |      1.35064  |                              10.8234 |         1.97076 | onemli bir otokorelasyon belirtisi yok (~2'ye yakin) |

## toplam_tutar: Es-zamanli Sonuc Degiskenleri Testi (tutarlilik kontrolu)

| variant                     |   holdout_r2 |   holdout_rmse |   holdout_mae |   rmse_improvement_over_baseline_pct |   durbin_watson | durbin_watson_yorum                                                                         |
|:----------------------------|-------------:|---------------:|--------------:|-------------------------------------:|----------------:|:--------------------------------------------------------------------------------------------|
| with_concurrent_outcomes    |    -0.162025 |        812.605 |       583.467 |                             15.8219  |        0.935935 | pozitif otokorelasyon belirtisi (model artik hatalari sistematik oruntu birakiyor olabilir) |
| without_concurrent_outcomes |    -0.534786 |        933.89  |       604.457 |                              3.25791 |        1.2119   | pozitif otokorelasyon belirtisi (model artik hatalari sistematik oruntu birakiyor olabilir) |

## Marjinal Katki ve Sizinti Testi Notlari

- **Hava/zaman degiskenlerinin davranissal modele EK katkisi**: full_model (0.612) - behavior_only (0.543) = **+0.069** (holdout accuracy puani).
- **Hava durumunun TEK BASINA naif taban cizgisine gore katkisi**: weather_only (0.481) - naive_baseline (0.298) = **+0.183**.
- **Davranissal/finansal degiskenlerin taban cizgisine gore katkisi**: behavior_only (0.543) - naive_baseline (0.298) = **+0.245**.
- **oturum_sure_dk icin sizinti testi**: es-zamanli sonuc degiskenleri (toplam_tutar, toplam_miktar, urun_sayisi) modelde iken holdout R^2=0.393; bu degiskenler CIKARILINCA (sadece hava+zaman+masa_grup) holdout R^2=-0.129 (fark: +0.523). Buyuk dusus, onceki yuksek R^2'nin buyuk olcude es-zamanli sonuc degiskenlerinden kaynaklandigini gosterir; hava durumunun kendisi bu hedefi yuksek dogrulukla aciklamiyor. H2 degerlendirmesinde bu hedef icin 'without_concurrent_outcomes' sonucu esas alinmalidir.
- **urun_sayisi icin sizinti testi**: es-zamanli sonuc degiskenleri (toplam_tutar, toplam_miktar, oturum_sure_dk) modelde iken holdout R^2=0.736; bu degiskenler CIKARILINCA (sadece hava+zaman+masa_grup) holdout R^2=0.203 (fark: +0.533). Buyuk dusus, onceki yuksek R^2'nin buyuk olcude es-zamanli sonuc degiskenlerinden kaynaklandigini gosterir; hava durumunun kendisi bu hedefi yuksek dogrulukla aciklamiyor. H2 degerlendirmesinde bu hedef icin 'without_concurrent_outcomes' sonucu esas alinmalidir.
- **toplam_tutar icin sizinti testi**: es-zamanli sonuc degiskenleri (toplam_miktar, urun_sayisi, oturum_sure_dk) modelde iken holdout R^2=-0.162; bu degiskenler CIKARILINCA (sadece hava+zaman+masa_grup) holdout R^2=-0.535 (fark: +0.373). Buyuk dusus, onceki yuksek R^2'nin buyuk olcude es-zamanli sonuc degiskenlerinden kaynaklandigini gosterir; hava durumunun kendisi bu hedefi yuksek dogrulukla aciklamiyor. H2 degerlendirmesinde bu hedef icin 'without_concurrent_outcomes' sonucu esas alinmalidir.

## H2 Hipotezi -- Nihai (Sizinti-Duzeltilmis) Degerlendirme

`is_paketi4_model_dogrulama.py`'nin `urun_sayisi` icin raporladigi H2 sonucu, es-zamanli sonuc degiskenlerini (toplam_tutar, toplam_miktar, oturum_sure_dk) ozellik olarak icerdigi icin sizintili olabilir (bkz. yukaridaki sizinti testi). Bu nedenle H2'nin nihai degerlendirmesi icin, **yalnizca hava durumu + zaman + masa_grup degiskenleriyle** (es-zamanli sonuc degiskenleri OLMADAN) elde edilen sonuc esas alinmistir:

- Holdout R^2 = 0.203
- Naif ortalama tahminine gore RMSE iyilesmesi = %10.8
- Sonuc: **ORTA duzeyde, sinirli dogruluk destegi bulunmustur**

Onemli: bu sonuc, `is_paketi4`'un kendi H2 raporundaki (sizintili) sayilardan farkli olabilir. Rapor yazilirken bu bolumdeki sayilar esas alinmali, `is_paketi4`'un urun_sayisi icin verdigi orijinal R2/RMSE sayilari 'sizinti supheli' olarak isaretlenmelidir.
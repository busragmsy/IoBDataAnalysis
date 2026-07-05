# Is Paketi 4 - Modelin Test Edilmesi ve Dogrulanmasi

Bu rapor `is_paketi4_model_dogrulama.py` script'i tarafindan otomatik uretilmistir.

## Capraz Dogrulama Ozeti (TimeSeriesSplit, 5 kat)

| target         |   cv_rmse_mean |   cv_rmse_std |   cv_mae_mean |   cv_mae_std |   cv_r2_mean |   cv_r2_std |   n_splits |
|:---------------|---------------:|--------------:|--------------:|-------------:|-------------:|------------:|-----------:|
| toplam_tutar   |      396.007   |   267.3       |    280.167    |  194.982     |     0.11537  |   0.266768  |          5 |
| oturum_sure_dk |       14.7736  |     3.69983   |      9.45715  |    0.691253  |     0.869488 |   0.119076  |          5 |
| urun_sayisi    |        1.07353 |     0.0181779 |      0.751007 |    0.0123971 |     0.727283 |   0.0185445 |          5 |

## Holdout Seti Tanisal Sonuclari (Bias, Durbin-Watson, ADF)

| target         | label                                 |   holdout_test_size |   holdout_rmse |   holdout_mae |   holdout_r2 |   baseline_rmse_mean_predictor |   rmse_improvement_over_baseline_pct |   bias_overall_mean_residual |   durbin_watson | durbin_watson_yorum                                                                         |   adf_statistic_on_residuals |   adf_pvalue_on_residuals | adf_yorum                                                                                           |
|:---------------|:--------------------------------------|--------------------:|---------------:|--------------:|-------------:|-------------------------------:|-------------------------------------:|-----------------------------:|----------------:|:--------------------------------------------------------------------------------------------|-----------------------------:|--------------------------:|:----------------------------------------------------------------------------------------------------|
| toplam_tutar   | Gelir davranisi (toplam_tutar)        |               26650 |      812.536   |    583.469    |    -0.161828 |                       965.34   |                              15.829  |                  580.386     |        0.935681 | pozitif otokorelasyon belirtisi (model artik hatalari sistematik oruntu birakiyor olabilir) |                     -14.877  |                1.6307e-27 | residualler duragan (p<0.05) -> modelin aciklayamadigi sistematik bir trend/kalinti yapi gorulmuyor |
| oturum_sure_dk | Oturum suresi davranisi               |               26650 |       12.5005  |      8.82017  |     0.723169 |                        39.2248 |                              68.131  |                   -0.879279  |        1.93704  | onemli bir otokorelasyon belirtisi yok (~2'ye yakin)                                        |                    -111.412  |                0          | residualler duragan (p<0.05) -> modelin aciklayamadigi sistematik bir trend/kalinti yapi gorulmuyor |
| urun_sayisi    | Satis / siparis adedi (H2 ana hedefi) |               26650 |        1.05691 |      0.742767 |     0.735701 |                         2.0582 |                              48.6488 |                    0.0213722 |        1.98478  | onemli bir otokorelasyon belirtisi yok (~2'ye yakin)                                        |                     -72.8862 |                0          | residualler duragan (p<0.05) -> modelin aciklayamadigi sistematik bir trend/kalinti yapi gorulmuyor |

### H2 Hipotezi Degerlendirmesi (hedef: satis/siparis adedi - `urun_sayisi`)

Capraz dogrulama ortalama R2 = 0.727, naif ortalama tahminine gore RMSE iyilesmesi = %48.6. Sonuc: YUKSEK dogruluk destegi bulunmustur.


Grafikler icin bkz: `Outputs/Post2021_YeniYapi/Model_Dogrulama/grafikler/`
## İş Paketi 4: Model Doğrulama ve H2 Hipotezi Testi

Öneride vaat edilen model doğrulama adımları (eğitim/test ayrımı, çapraz 
doğrulama, RMSE/MAE, bias analizi, Durbin-Watson) tamamlanmış ve H2 
hipotezi ("IoB sistemi ile hava durumu verilerinin entegrasyonu, müşteri 
davranışlarının öngörüsünde yüksek doğruluk sağlar") sayısal olarak test 
edilmiştir.

**Yöntem:** Zaman serisi yapısına uygun olarak TimeSeriesSplit ile 
5 katlı çapraz doğrulama yapılmış, kronolojik son %20'lik dilim bağımsız 
test (holdout) seti olarak ayrılmıştır. RandomForestRegressor modelleri 
üç hedef değişken için eğitilmiştir.

**Sonuçlar:**

| Hedef Değişken | CV R² | Holdout R² | RMSE İyileşmesi (baseline'a göre) | Durbin-Watson |
|---|---|---|---|---|
| Satış/sipariş adedi (urun_sayisi) | 0.727 | 0.736 | %48.6 | 1.985 |
| Oturum süresi (oturum_sure_dk) | 0.869 | 0.723 | %68.1 | 1.937 |
| Gelir (toplam_tutar) | 0.115 | -0.162 | %15.8 | 0.936 |

**H2 Hipotezi Değerlendirmesi:** Satış/sipariş adedi ve oturum süresi 
tahminlerinde model, hem çapraz doğrulamada hem bağımsız test setinde 
tutarlı ve yüksek doğruluk göstermiştir; residuallerde anlamlı 
otokorelasyon veya sistematik önyargı (bias) gözlenmemiştir. Bu bulgular 
H2 hipotezini **desteklemektedir**.

**Sınırlama:** Gelir (`toplam_tutar`) tahmininde model performansı 
yetersiz kalmıştır (holdout R² = -0.16, Durbin-Watson = 0.94). Bunun 
başlıca nedeni, incelenen dönemde gözlenen enflasyon kaynaklı fiyat 
artışıdır: model, eğitim döneminde gördüğü fiyat aralığının dışına 
(daha yüksek tutarlara) ekstrapolasyon yapamamakta ve test döneminde 
sistematik olarak düşük tahmin üretmektedir (ortalama residual: +580 TL). 
Bu, klasik bir "concept drift" (zamanla veri dağılımının kayması) 
örneğidir ve H2 hipotezi gelir değişkeni özelinde desteklenmemektedir.
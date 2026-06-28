# IoB Tabanlı Hava Durumu — Müşteri Davranışı Analizi
## Sonuç Raporu (2021 ve Sonrası)

**Proje:** TÜBİTAK 2209-A — Internet of Behaviors (IoB) yaklaşımıyla lokanta müşteri davranışlarının hava durumu verileriyle ilişkilendirilmesi  
**Analiz dönemi:** 2 Ocak 2021 — 28 Mayıs 2025  
**Temiz analiz seti:** 120.749 oturum (133.247 ham kayıttan türetilmiş)  
**Rapor tarihi:** Haziran 2026

---

## 1. Yönetici Özeti

Bu rapor, 2021 ve sonrası döneme odaklanan beş aşamalı analiz hattının (`00`–`04` notebook'ları) tüm çıktılarını, bulgularını ve yöntem tercihlerini kapsar. Ham POS (satış noktası) kayıtları ile Open-Meteo saatlik hava verisi birleştirilerek oturum düzeyinde zenginleştirilmiş bir veri seti oluşturulmuş; outlier ayrımı, enflasyon düzeltmesi ve kategorik özellik üretimi sonrasında dört istatistiksel analiz katmanı uygulanmıştır:

| Aşama | Yöntem | Ana bulgu |
|-------|--------|-----------|
| **01 Betimsel** | Tanımlayıcı istatistik, çapraz tablo, görselleştirme | İç Salon %39 baskın; Paket 2021'de lider iken 2025'te İç Salon %51'e yükseldi |
| **02 Zaman serisi** | Günlük agregasyon, decomposition, ADF, ACF/PACF, CCF | Haftalık mevsimsellik belirgin; sıcaklık anomalisi ↔ bahçe tercihi eşzamanlı zayıf-orta (r = +0,20) |
| **03 Korelasyon/Regresyon** | Spearman, Ki-Kare, Multinomial Logit | Nem grubu en güçlü kategorik etki (Cramér's V = 0,13); sıcaklık bahçe odds'unu %12 artırıyor (OR = 1,12) |
| **04 Birliktelik kuralları** | Hava kümeleme + Apriori | Soğuk/yağışlı günlerde CORBA→SU lift değeri sıcak günlere göre %23,9 daha yüksek |

**Genel yorum:** Hava durumu müşteri tercihlerini istatistiksel olarak anlamlı biçimde etkiler; ancak sürekli değişkenlerde korelasyon katsayıları zayıftır (|ρ| < 0,03). En güçlü ve operasyonel olarak eyleme dönüştürülebilir bulgular **masa tercihi** (sıcaklık, yağış, nem) ve **ürün birliktelikleri** (CORBA+SU soğukta, KÜNEFE+DONDURMA sıcakta) düzeyindedir.

---

## 2. Veri Kaynağı ve Ham Veriden İşlenebilir Veriye Dönüşüm

### 2.1 Ham veri bileşenleri

Analiz hattının girdisi `Veriler/oturum_hava_birlesik_2021_ve_sonrasi.csv` dosyasıdır (133.247 satır × 34 sütun). Bu dosya iki kaynağın birleştirilmesiyle oluşturulmuştur:

**A) Restoran oturum verisi (POS)**  
- Kaynak: `grup_oturum_bazli_tubitak.csv` — aynı çek (CEKNO) altındaki satırlar oturum bazında gruplanmış kayıtlar  
- Her satır bir müşteri oturumunu temsil eder: masa numarası (MASANO), açılış/kapanış zamanı, toplam miktar, tutar, ürün sayısı, ürün listesi  

**B) Hava durumu verisi (Open-Meteo Archive API)**  
- Kaynak: `hava-durumu2012-2025.csv` — saatlik çözünürlükte arşiv verisi  
- Değişkenler: sıcaklık, hissedilen sıcaklık, nem, çiğ noktası, yağış (rain/showers/snowfall), rüzgar hızı/yönü, bulutluluk, basınç, kısa dalga radyasyonu, gündüz/gece göstergesi  

### 2.2 Birleştirme (merge) mantığı

Oturum kayıtları ile hava verisi **saat bazında** eşleştirilmiştir:

1. Hava verisindeki `datetime` sütunu saatlik frekansa yuvarlanır (`floor('h')`) → `merge_saati`
2. Oturum verisindeki `merge_saati` ile left join yapılır
3. Böylece her oturum, açılış anına en yakın saatlik hava gözlemiyle eşleşir

Bu yaklaşım, IoB çerçevesinde **davranış anındaki çevresel bağlamı** oturuma bağlamayı amaçlar.

### 2.3 Oturum düzeyinde türetilmiş alanlar (ham birleşik veride)

Ham birleştirme aşamasında (`data-cleaning.ipynb`, arşiv: `eski_analizler_2012_2025/`) şu dönüşümler uygulanmıştır:

| Adım | İşlem | Açıklama |
|------|-------|----------|
| Oturum süresi | `(kapama_datetime − acilis_datetime)` dakika cinsinden | POS kayıtlarından hesaplanır |
| Masa grubu | MASANO önekine göre sınıflandırma | **Bahçe** (B…), **İç Salon** (I…), **Paket** (P…) |
| Zaman değişkenleri | tarih, saat, gün adı, ay, yıl, hafta no | Davranışsal zaman analizleri için |
| Yağış kategorisi | precipitation + rain/showers/snowfall | Açık / Bulutlu / Yağmur / Kar |
| Sıcaklık aralığı | temperature_2m bantları | Soğuk (0–10°), Ilık (10–20°), Ilıman (20–30°), Sıcak (30°+) |
| Outlier işaretleme | oturum_sure_dk = 0 → `sifir_sureli`; > 240 dk → `uzun_oturum` | Temiz sette `outlier_flag = NaN` |

2021+ filtresi uygulanarak 133.247 kayıtlık dönemsel dosya üretilmiştir.

### 2.4 Veri hazırlama notebook'u (`00_veri_hazirlama_2021_sonrasi.ipynb`)

Ham birleşik dosya, analiz öncesinde aşağıdaki ek işlemlerden geçirilmiştir:

#### 2.4.1 Outlier yönetimi ve veri seti ayrımı

| Grup | Kayıt | Oran | Kullanım |
|------|-------|------|----------|
| **Temiz set** (`outlier_flag = NaN`) | 120.749 | %90,6 | Genel analizler (01–04) |
| **Uzun oturum** (`uzun_oturum`, > 240 dk) | 9.037 | %6,8 | Ayrı uzun oturum–hava incelemesi |
| **Sıfır süreli** (`sifir_sureli`, = 0 dk) | 3.461 | %2,6 | Genel analizden hariç (çoğunlukla paket siparişleri) |

**Gerekçe:** Sıfır süreli kayıtlar POS sisteminde anlık paket işlemlerini yansıtır; oturum süresi analizlerini bozar. 240 dakikayı aşan oturumlar operasyonel anomalilerdir (masa unutulması, sistem hatası vb.) ve ortalama süre istatistiklerini yukarı çeker.

**Çıktı dosyaları:**
- `Veriler/oturum_hava_temiz_2021_sonrasi.csv` — 66 sütun, 120.749 kayıt
- `Veriler/oturum_hava_uzun_oturum_2021_sonrasi.csv`
- `Veriler/oturum_hava_tum_2021_sonrasi.csv` (flag'li tam set)

#### 2.4.2 Enflasyon normalleştirmesi (TÜFE deflate)

Türkiye'deki yüksek enflasyon nedeniyle nominal `toplam_tutar` yıllar arası doğrudan karşılaştırılamaz:

| Metrik | 2021 | 2025† |
|--------|------|-------|
| Nominal medyan tutar | 74 TL | 820 TL |
| Deflate medyan tutar (2021 baz) | 90 TL | 195 TL |

**Yöntem:** TÜİK TÜFE Genel Endeks (2003=100) aylık değerleri kullanılarak:

```
toplam_tutar_deflate_2021 = toplam_tutar × (TUFE_2021_12 / TUFE_aylik)
```

**Ana fiyat-bağımsız hedefler:** `toplam_miktar`, `urun_sayisi` — enflasyondan etkilenmez, yıllar arası karşılaştırma için tercih edilir.

#### 2.4.3 Yeni kategorik hava değişkenleri

| Değişken | Sınıflar | Eşikler |
|----------|--------|---------|
| `ruzgar_seviyesi` | Sakin / Hafif / Orta / Güçlü | ≤7 / ≤15 / ≤25 / >25 km/s |
| `nem_grubu` | Kuru / Normal / Nemli / Çok Nemli | <40 / ≤60 / ≤80 / >80 % |
| `yagis_yogunlugu` | Yağışsız / Hafif / Orta / Yoğun | 0 / ≤1 / ≤5 / >5 mm |
| `bulut_grubu` | Açık / Parçalı Bulutlu / Kapalı | ≤25 / ≤75 / >75 % |

Ayrıca tüm sürekli hava değişkenleri için **beşli quantile bin** (`*_qbin`) üretilmiştir.

#### 2.4.4 Zaman ve bağlam değişkenleri

| Değişken | Tanım |
|----------|-------|
| `mevsim` | Kış (Ara–Şub), İlkbahar (Mar–May), Yaz (Haz–Ağu), Sonbahar (Eyl–Kas) |
| `covid_donemi` | 2021 → "Geçiş"; 2022+ → "Tam Post-COVID" |
| `gun_tipi` | Hafta içi / Hafta sonu |
| `oglen_aksam` | Öğle (11–14), İkindi (14–17), Akşam (17–22), Diğer |
| `hava_hissedilen_fark` | temperature_2m − apparent_temperature |
| `yagis_var` | precipitation > 0 ise 1, aksi halde 0 |
| `ruzgar_yonu_sin/cos` | Rüzgar yönünün dairesel kodlaması |

#### 2.4.5 Kapalı gün analizi

İşletmenin kapalı olduğu günler analiz döneminde tespit edilmiştir:

| Metrik | Değer |
|--------|-------|
| Toplam takvim günü | 1.609 |
| Açık gün | 1.390 |
| Kapali gün | 219 (%13,6) |
| En uzun kapalı dönem | 37 gün (10 Nisan – 16 Mayıs 2021) |

**Çıktı:** `Outputs/Veri_Hazirlama/kapali_gun_ozet.txt`, `kpi_ozet.json`

#### 2.4.6 Temiz set KPI özeti

| Metrik | Değer |
|--------|-------|
| Dönem | 2021-01-02 — 2025-05-28 |
| Ortalama oturum süresi | 33,0 dk (medyan: 25 dk) |
| Ortalama miktar | 5,91 birim |
| Ortalama ürün sayısı | 3,41 |
| Deflate ortalama tutar | 172,9 TL (2021 bazlı) |
| Benzersiz çek sayısı | 120.749 (birebir eşleşme) |

---

## 3. Analiz Mimarisi ve Yöntem Seçim Gerekçeleri

```
Ham POS + Hava (birleşik CSV)
        ↓
[00] Veri hazırlama → temiz set (120.749 oturum)
        ↓
[01] Betimsel analiz ──── "Ne oluyor?" (dağılımlar, trendler)
        ↓
[02] Zaman serisi ──────── "Ne zaman, hangi ritimle?" (trend/mevsim/lag)
        ↓
[03] Korelasyon/Regresyon "Hava ↔ davranış ilişkisi ne kadar güçlü?"
        ↓
[04] Birliktelik kuralları ─ "Hangi ürünler birlikte, hava koşuluna göre?"
```

Her aşama bir öncekinin bulgularına dayanır: 02'deki CCF eşzamanlı etki bulgusu, 03'te anlık (lag ≈ 0) modelleme tercihini destekler; 04'te hava kümeleri 03'teki kategorik gruplama mantığıyla uyumludur.

---

## 4. Bölüm 01 — Betimsel Analiz

**Notebook:** `01_descriptive_analiz.ipynb`  
**Veri:** `oturum_hava_temiz_2021_sonrasi.csv` (120.749 × 66)  
**Çıktı klasörü:** `Outputs/Analizler/01_descriptive/`

### 4.1 Yöntem ve neden tercih edildi?

Betimsel analiz, herhangi bir çıkarımsal modele geçmeden önce verinin **yapısını, dağılımını ve segmentler arası farkları** ortaya koymak için uygulanır. IoB projelerinde bu aşama, "hangi davranış metriklerinin hava ile birlikte inceleneceğini" ve "hangi alt grupların (masa, mevsim, gün) anlamlı ayrıştığı" sorusuna yanıt verir.

**Hedef metrikler:** `oturum_sure_dk`, `toplam_miktar`, `urun_sayisi`, `toplam_tutar_deflate_2021`

### 4.2 Sayısal özet (Bölüm 2.1)

| Değişken | Ort. | Medyan | Std | Min | Max | Q1 | Q3 |
|----------|------|--------|-----|-----|-----|----|----|
| toplam_miktar | 5,91 | 5,0 | 4,80 | 0,75 | 191 | 3,0 | 8,0 |
| urun_sayisi | 3,41 | 3,0 | 2,05 | 1,0 | 25 | 2,0 | 4,0 |
| oturum_sure_dk | 33,0 | 25,0 | 34,0 | 1,0 | 240 | 15,0 | 38,0 |

**Yorum:** Tüm davranış değişkenleri sağa çarpık (medyan < ortalama); özellikle oturum süresinde geniş std (34 dk) uzun kuyruklu dağılımı işaret eder. Bu bulgu, 03. analizde Spearman korelasyonunun Pearson'a tercih edilmesini destekler.

### 4.3 Masa grubu dağılımı ve segment profilleri

| Masa grubu | Oturum | Pay (%) | Ort. süre (dk) | Ort. miktar | Ort. ürün |
|------------|--------|---------|----------------|-------------|-----------|
| İç Salon | 47.052 | **38,97** | 27,0 | **6,71** | **4,01** |
| Bahçe | 37.622 | 31,16 | 30,2 | 6,28 | 3,79 |
| Paket | 36.075 | 29,88 | 43,8* | 4,49 | 2,22 |

*Paket ortalama süresi yüksek std (55 dk) ile birlikte yorumlanmalıdır; medyan 24 dk'dır.

**Sonuç:** İç Salon en yüksek sepet hacmi ve ürün çeşitliliğine sahiptir; Paket en düşük miktar/ürün ortalamasına sahip ancak operasyonel olarak farklı bir kanaldır.

### 4.4 Hava durumu — yıllık trend (Bölüm 2.2)

| Yıl | Ort. sıcaklık (°C) | Ort. nem (%) | Toplam yağış* | Ort. rüzgar | Ort. bulut | Oturum |
|-----|-------------------|--------------|---------------|-------------|------------|--------|
| 2021 | 17,0 | 65,7 | 2.831 | 16,2 | 53,8 | 28.138 |
| 2022 | 17,1 | 64,4 | 2.015 | 16,4 | 48,2 | 28.982 |
| 2023 | 18,3 | 66,0 | 2.486 | 16,9 | 54,8 | 28.794 |
| 2024 | 18,0 | 65,7 | 2.431 | 15,7 | 50,5 | 26.996 |
| 2025† | 12,5 | 65,2 | 593 | 11,9 | 54,6 | 7.839 |

*Oturum bazlı precipitation toplamı · †2025 yalnızca Ocak–Mayıs

**Yağış kategorisi payı (yıllık ort.):** Açık ~%52–58, Bulutlu ~%28–32, Yağmur ~%13–19. Kar payı 2025'te %2,4'e yükselmiştir.

#### Grafik: `2_2_hava_yillik_trend.png`
Yıllara göre ortalama sıcaklık, nem, rüzgar ve bulutluluk çizgileri. **2023 en sıcak yıl** (18,3°C); 2025 kısmi yıl verisi nedeniyle düşük görünür.

#### Grafik: `2_2_yagis_kategori_yillik_stacked.png`
Yıllık yağış kategorisi dağılımının yığılmış bar grafiği. Açık günlerin baskınlığı ve 2022'nin en kurak yıl olması görselleştirilir.

### 4.5 Müşteri davranışı — zaman dağılımı (Bölüm 2.3)

**Yıllık sipariş trendi (enflasyondan bağımsız):**  
2021: 28.138 → 2022: 28.982 (tepe) → 2024: 26.996 → 2025†: 7.839

**Hafta sonu vs hafta içi:**
- Hafta sonu payı: %17,7
- Hafta sonu ort. miktar: 6,47 vs hafta içi: 5,79

**Günlük dağılım:**
- En yoğun: Perşembe (21.060 oturum)
- En düşük: Pazar (9.024 oturum)
- Pazar en yüksek ort. miktar: 6,81

**Mevsimsel metrikler:**

| Mevsim | Oturum | Ort. süre (dk) | Ort. miktar |
|--------|--------|----------------|-------------|
| Kış | 33.740 | 33,3 | 5,89 |
| İlkbahar | 23.826 | 32,6 | 5,91 |
| Yaz | 28.727 | 32,6 | 5,83 |
| Sonbahar | 34.456 | **33,4** | **6,00** |

**Sonbahar** en uzun ortalama süre ve en yüksek ortalama miktarı gösterir.

#### Grafik: `2_3_gunluk_dagilim.png`
Haftanın günlerine göre oturum sayısı ve ortalama miktar. Perşembe yoğunluğu ve Pazar'da yüksek sepet ortalaması görülür.

#### Grafik: `2_3_hafta_ici_sonu.png`
Hafta içi/sonu karşılaştırması; hafta sonu daha az oturum ama daha yüksek ortalama miktar.

#### Grafik: `2_3_mevsimsel_metrikler.png`
Mevsim bazında ortalama süre ve miktar bar grafikleri.

#### Grafik: `2_3_saat_ogun_dagilim.png`
Saat ve öğün (Öğle/İkindi/Akşam) bazında oturum yoğunluğu; akşam pik saatleri belirgin.

#### Grafik: `2_3_aylik_siparis_cizgi.png`
Aylık sipariş adedi zaman serisi; mevsimsel dalgalanma ve 2021 Nisan–Mayıs kapalı dönem etkisi izlenebilir.

### 4.6 Masa grubu davranışı ve hava etkileşimi (Bölüm 2.4)

**Yıllık masa grubu payı değişimi:**

| Yıl | Bahçe | Paket | İç Salon |
|-----|-------|-------|----------|
| 2021 | %32,7 | **%38,4** | %28,9 |
| 2024 | %28,9 | %29,0 | **%42,1** |
| 2025† | %19,1 | %30,3 | **%50,7** |

**Yapısal dönüşüm:** 2021'de Paket baskın (%38); sonraki yıllarda İç Salon sürekli yükselerek 2025'te %51'e ulaşmıştır. Bahçe payı 2025'te %19'a gerilemiştir — kısmi yıl ve kış/ilkbahar ağırlığı etkili olabilir.

#### Grafik: `2_4_masa_grup_yillik_trend.png`
Üç masa grubunun yıllık pay değişimini gösteren çizgi grafik; Paket→İç Salon kayması net görülür.

#### Grafik: `2_4_hava_masa_heatmap.png`
Yağış kategorisi × masa grubu tercih ısı haritası. Yağmurlu günlerde İç Salon ve Paket tercihi artış eğilimi; açık günlerde Bahçe baskınlığı.

#### Grafik: `sicaklik_masa_tercih_heatmap.png`
Sıcaklık aralığı × masa grubu çapraz dağılım. Ilıman–sıcak bantlarda Bahçe payı yükselir; soğuk bantlarda İç Salon baskın.

#### Grafik: `hava_oturum_suresi_boxplot.png`
Hava kategorilerine göre oturum süresi dağılımı (boxplot). Kategoriler arası medyan farkları sınırlı; aykırı değerler geniş aralık gösterir.

#### Grafik: `betimsel_ozet_grafikleri.png`
Masa dağılımı, mevsimsel süre/miktar ve temel KPI'ların özet panel grafiği.

---

## 5. Bölüm 02 — Zaman Serisi Analizi

**Notebook:** `02_zaman_serisi_analizi.ipynb`  
**Çıktı klasörü:** `Outputs/Analizler/02_zaman_serisi/`

### 5.1 Yöntem ve neden tercih edildi?

Oturum düzeyindeki gözlemler aynı gün içinde saat/öğün bazında yoğun **mikro-yapısal gürültü** taşır; trend ve mevsimsellik bileşenlerinin ayrıştırılmasını zorlaştırır. Bu nedenle veri **günlük frekansa (D)** indirgenmiştir.

**Günlük metrikler:** sipariş adedi, toplam miktar (sum), ort. oturum süresi, ort. deflate tutar; CCF için sıcaklık anomalisi, bahçe/paket oranları.

**Eksik gün tamamlama:** 218 kapalı/eksik gün zaman-esaslı doğrusal interpolasyon (`interpolate(method='time')`) ile doldurulmuştur. Toplam 1.608 takvim günü.

| Yöntem | Gerekçe |
|--------|---------|
| **Seasonal decomposition (period=7, additive)** | Restoran talebinde hafta içi–sonu ritmi belirgin 7 günlük periyot oluşturur |
| **ADF testi** | Serinin durağanlığını test eder; ARIMA vb. modeller için ön koşul |
| **ACF/PACF** | Haftalık bellek yapısını (7. lag spike) doğrular |
| **CCF (±14 gün)** | Hava değişkenleri ile masa tercihi arasındaki **gecikmeli (lagged)** ilişkiyi araştırır |

### 5.2 Seasonal decomposition sonuçları

| Seri | Trend (std) | Mevsimsellik (std) | Kalıntı (std) | Yorum |
|------|-------------|-------------------|---------------|-------|
| Sipariş adedi | 12,6 | **18,0** | 23,2 | Haftalık mevsimsellik belirgin; kalıntı varyansı en yüksek (dışsal şoklar) |
| Toplam miktar | 89,5 | **96,5** | 155,8 | Sepet hacminde güçlü haftalık ritim + yüksek oynaklık |
| Oturum süresi | 6,4 | **0,5** | 6,0 | Sürede haftalık döngü zayıf; trend/kalıntı baskın |

#### Grafik: `decompose_siparis_adedi.png`
Dört panel: orijinal seri, trend, 7 günlük mevsimsellik, kalıntı. Hafta sonu–içi dalgalanması mevsimsellik panelinde ±25 sipariş bandında görülür.

#### Grafik: `decompose_toplam_miktar.png`
Günlük toplam miktar ayrıştırması; mevsimsellik std (96,5) trend std'ye (89,5) yakın — haftalık ritim sepet hacminde de güçlü.

#### Grafik: `decompose_oturum_suresi.png`
Oturum süresinde mevsimsellik neredeyse düz (std = 0,5); süre daha çok trend ve rastgele kalıntı ile açıklanır.

### 5.3 Durağanlık ve bellek analizi

**ADF testi — günlük sipariş adedi:**

| Metrik | Değer |
|--------|-------|
| Test istatistiği | −5,92 |
| p-değeri | < 0,001 |
| Lag | 20 |
| Gözlem | 1.587 |
| **Sonuç** | Seri %5 düzeyinde **durağan** (H₀ reddedildi) |

#### Grafik: `acf_pacf_siparis_adedi.png`
30 lag'e kadar ACF ve PACF. 7., 14., 21. lag'lerde tekrarlayan pozitif spike'lar **haftalık mevsimselliği** doğrular.

### 5.4 Çapraz korelasyon (CCF) sonuçları

| İlişki | En güçlü lag | r | Yorum |
|--------|--------------|---|-------|
| Sıcaklık anomalisi ↔ Bahçe tercihi (%) | **0 gün** | **+0,20** | Eşzamanlı, zayıf-orta pozitif: anomalik ılıman günlerde bahçe payı artış eğilimi |
| Günlük yağış ↔ Paket oranı (%) | −4 gün | +0,07 | Gecikmeli etki istatistiksel olarak **zayıf**; yağış şokunun paket tercihine güçlü lagged etkisi kanıtlanmadı |

**Sıcaklık anomalisi:** Günlük ortalama sıcaklıktan decomposition trend bileşeni çıkarılarak elde edilmiştir; uzun dönem iklim eğilimi ile kısa vadeli hava şokları ayrıştırılır.

#### Grafik: `ccf_hava_masa_grubu.png`
İki CCF bar grafiği (±14 gün lag). Sıcaklık anomalisi–bahçe ilişkisinde lag 0'da en yüksek korelasyon; yağış–paket ilişkisinde tüm lag'lerde zayıf.

---

## 6. Bölüm 03 — Korelasyon ve Regresyon Analizi

**Notebook:** `03_korelasyon_ve_regresyon.ipynb`  
**Çıktı klasörü:** `Outputs/Analizler/03_korelasyon_regresyon/`

02'deki CCF bulgusu (lag ≈ 0) ile tutarlı olarak, bu analiz **oturum anındaki eşzamanlı** hava–tercih ilişkisine odaklanır.

### 6.1 ADIM 1 — Korelasyon (Pearson / Spearman)

**Yöntem seçimi:**

| Test | Ne zaman? |
|------|-----------|
| **Shapiro-Wilk** | Davranış değişkenlerinin normalliğini test eder |
| **Pearson r** | Doğrusal ilişki, normal dağılım varsayımı |
| **Spearman ρ** | Monoton ilişki, aykırı değerlere dayanıklı — **birincil yorum** |

**Normallik sonucu:** Tüm davranış değişkenlerinde normallik reddedildi (p ≈ 0) → **birincil yorum Spearman ρ**.

**En güçlü Spearman ilişkileri:**

| Hava | Davranış | ρ | p | Anlamlı? |
|------|----------|---|---|----------|
| temperature_2m | urun_sayisi | +0,026 | <0,001 | Evet |
| relative_humidity_2m | urun_sayisi | −0,019 | <0,001 | Evet |
| temperature_2m | oturum_sure_dk | +0,019 | <0,001 | Evet |
| relative_humidity_2m | oturum_sure_dk | −0,017 | <0,001 | Evet |
| windspeed_10m | urun_sayisi | +0,008 | 0,005 | Evet |

**Yorum:** İlişkiler istatistiksel olarak anlamlı olsa da **etki büyüklüğü zayıftır** (|ρ| < 0,03). Büyük örneklem (n = 120.749) küçük farkları anlamlı kılar; pratik etki sınırlıdır.

#### Grafik: `korelasyon_heatmap_spearman.png`
Hava değişkenleri × davranış metrikleri Spearman ısı haritası. Sıcaklık–ürün sayısı hafif pozitif; nem–ürün sayısı hafif negatif tonlanma.

### 6.2 ADIM 2 — Ki-Kare (kategorik hava × masa tercihi)

**Yöntem:** Ki-Kare Bağımsızlık Testi (χ²) + Cramér's V etki büyüklüğü  
**H₀:** Hava kategorisi ile masa tercihi birbirinden bağımsızdır.

| Hava kategorisi | χ² | DoF | p | Cramér's V | Anlamlı? |
|-----------------|-----|-----|---|------------|----------|
| nem_grubu | 4.076 | 6 | <0,001 | **0,130** | Evet |
| bulut_grubu | 2.529 | 4 | <0,001 | 0,102 | Evet |
| yagis_yogunlugu | 729 | 6 | <0,001 | 0,055 | Evet |
| ruzgar_seviyesi | 113 | 6 | <0,001 | 0,022 | Evet |

**Sonuç:** Tüm kategorik hava değişkenleri ile masa tercihi arasında anlamlı ilişki vardır. En güçlü kategorik etki **nem_grubu** (V = 0,13 — zayıf-orta sınırında).

### 6.3 ADIM 3 — Multinomial Logit (MNLogit)

**Yöntem seçimi:** Ki-Kare yalnızca ilişkinin varlığını kanıtlar; MNLogit her hava faktörünün **yönünü, büyüklüğünü ve koşullu etkisini** odds oranı (OR) ile ölçer.

- **Bağımlı değişken:** masa_grup (3 kategori)
- **Referans:** İç Salon (en sık kategori)
- **Bağımsız değişkenler:** temperature_2m, windspeed_10m, relative_humidity_2m, precipitation
- **Pseudo R²:** 0,054

**Anlamlı katsayılar — Bahçe vs İç Salon (referans):**

| Değişken | OR | p | Yorum |
|----------|----|---|-------|
| temperature_2m | **1,12** | <0,001 | 1°C artış → Bahçe odds'u %12 artar |
| precipitation | **0,88** | <0,001 | Yağış artışı Bahçe tercihini azaltır |
| windspeed_10m | 0,99 | <0,001 | Rüzgar artışı Bahçe odds'unu düşürür |
| relative_humidity_2m | 1,01 | <0,001 | Nem artışı hafif Bahçe yönelimi |

**Anlamlı katsayılar — Paket vs İç Salon:**

| Değişken | OR | p | Anlamlı (α=0,05)? |
|----------|----|---|-------------------|
| temperature_2m | 1,02 | <0,001 | Evet |
| windspeed_10m | 0,99 | <0,001 | Evet |
| relative_humidity_2m | 1,003 | <0,001 | Evet |
| precipitation | 1,03 | 0,064 | **Hayır** |

**Somut bulgu:** Bahçe tercihini en güçlü **artıran** faktör **sıcaklık** (OR = 1,12); en güçlü **azaltan** faktör **yağış** (OR = 0,88). Bu, CCF'deki eşzamanlı sıcaklık–bahçe korelasyonunu (r = +0,20) regresyon düzeyinde doğrular.

---

## 7. Bölüm 04 — Birliktelik Kuralları (Apriori)

**Notebook:** `04_birliktelik_kurallari.ipynb`  
**Çıktı klasörü:** `Outputs/Analizler/04_birliktelik_kurallari/`

### 7.1 Yöntem ve neden tercih edildi?

Apriori algoritması **işlem (transaction) düzeyinde** sepet içeriğini analiz eder. Ham oturum-hava verisinde hava koşulları ile ürün kalemleri aynı satırda karışık durduğundan, önce veri **hava durumunun davranış üzerindeki etkisini ayrıştırabilecek uç kümeler** halinde segmentlenmiştir.

**Hava kümeleme kuralları:**

| Küme | Tanım | Oturum | Pay |
|------|-------|--------|-----|
| **Sıcak ve Güneşli** | Temp > 15°C, yağışsız | 61.735 | %51,1 |
| **Soğuk ve Yağışlı** | Temp ≤ 15°C, yağışlı veya kapalı | 31.067 | %25,7 |
| **Diğer** | Ara koşullar | 27.947 | %23,1 |

**Apriori parametreleri:**
- `min_support = 0.01` (sepetlerin en az %1'inde görülen kalıplar)
- `min_lift > 1.0` (rastgele birliktelikten güçlü kurallar)
- Her küme için **ayrı ayrı** çalıştırılmıştır

**Metrikler:**
- **Support:** Kuralın tüm işlemlerde görülme oranı
- **Confidence:** Antecedent verildiğinde consequent'in görülme olasılığı
- **Lift:** Confidence / consequent'in marjinal olasılığı (>1 = pozitif birliktelik)

### 7.2 Küme karşılaştırması — öne çıkan bulgu

> **Soğuk ve Yağışlı** günlerde **AZ CORBA → SU** birliktelik kuralının Lift değeri (**2,19**), Sıcak günlere göre **%23,9 daha yüksektir** (sıcak lift = 1,77).

Benzer şekilde **CORBA → SU** kuralı soğukta lift **1,72** ile sıcağa göre **%13,8** daha güçlüdür.

**Lift farkı Top 5 (soğuk vs sıcak):**

| Kural | Lift (soğuk) | Lift (sıcak) | Fark (%) |
|-------|-------------|-------------|----------|
| AZ CORBA → SU | 2,19 | 1,77 | +23,9 |
| SU → AZ CORBA | 2,19 | 1,77 | +23,9 |
| ACISIZ YARIM FULL → ACISIZ FULL | 3,72 | 3,25 | +14,3 |
| SU → CORBA | 1,72 | 1,51 | +13,8 |

### 7.3 En güçlü kurallar

**Soğuk/Yağışlı — Top 3 (lift):**

| Kural | Lift | Confidence |
|-------|------|------------|
| KAYMAK → AYRAN + KÜNEFE | 7,73 | 0,53 |
| AYRAN + KÜNEFE → KAYMAK | 7,73 | 0,20 |
| KÜNEFE + ŞİŞE AYRAN → DONDURMA | 7,63 | 0,27 |

**Sıcak/Güneşli — Top 3 (lift):**

| Kural | Lift | Confidence |
|-------|------|------------|
| KÜNEFE + SU → DONDURMA | 8,04 | 0,37 |
| DONDURMA → KÜNEFE + SU | 8,04 | 0,23 |
| KÜNEFE + ŞİŞE AYRAN → DONDURMA | 7,96 | 0,37 |

**Ortak çekirdek:** Her iki kümede de AYRAN + FULL birlikteliği güçlüdür; hava koşuluna göre **yan ürün stratejisi** değişir (soğukta CORBA, sıcakta DONDURMA).

#### Grafik: `network_soguk_yagisli.png`
Soğuk/yağışlı kümede en güçlü birliktelik kurallarının ağ grafiği; KAYMAK–KÜNEFE–AYRAN üçgeni ve CORBA–SU bağlantısı görülür.

#### Grafik: `network_sicak_gunesli.png`
Sıcak/güneşli kümede KÜNEFE–DONDURMA–SU ağ yapısı baskın.

#### Grafik: `heatmap_lift_karsilastirma.png`
Soğuk vs sıcak lift değerlerinin karşılaştırmalı ısı haritası; CORBA–SU kurallarındaki hava koşuluna özgü fark net biçimde görselleştirilir.

---

## 8. Entegre Bulgular ve Yorum

### 8.1 Tutarlılık matrisi (analizler arası)

| Bulgu | 01 Betimsel | 02 Zaman serisi | 03 Regresyon | 04 Apriori |
|-------|-------------|-----------------|--------------|------------|
| Sıcaklık ↑ → Bahçe ↑ | sicaklik_masa_heatmap | CCF lag=0, r=+0,20 | OR=1,12 | — |
| Yağış ↑ → Bahçe ↓ | hava_masa_heatmap | CCF zayıf | OR=0,88 | — |
| Soğukta CORBA+SU güçlü | — | — | — | Lift fark +%24 |
| Sıcakta KÜNEFE+DONDURMA | — | — | — | Lift > 7 |
| Haftalık ritim baskın | gunluk/mevsim grafikleri | Decompose period=7 | — | — |
| İç Salon yapısal yükseliş | masa_grup_yillik_trend | — | Referans kategori | — |

### 8.2 Stratejik öneriler

1. **Soğuk/yağışlı günler:** CORBA + SU / AZ CORBA çapraz satış paketleri menü ve POS'ta öne çıkarılmalı (lift soğukta ~%14–24 daha yüksek — bilimsel kanıt).
2. **Sıcak/güneşli günler:** KÜNEFE + DONDURMA tatlı bundle'ı (lift > 7) yaz menüsü promosyonuna alınmalı.
3. **Masa yönetimi:** Hava tahminine göre bahçe kapasitesi planlaması — 1°C sıcaklık artışı bahçe odds'unu %12 artırır; yağışlı günlerde iç salon kapasitesi rezerve edilmeli.
4. **Genel:** AYRAN + FULL çekirdek birliktelik her iki kümede de güçlü; hava koşuluna göre yan ürün stratejisi değişir.
5. **Metrik seçimi:** Yıllar arası karşılaştırmalarda `toplam_miktar` ve `urun_sayisi` tercih edilmeli; tutar karşılaştırmalarında `toplam_tutar_deflate_2021` kullanılmalı.

### 8.3 Sınırlılıklar

- **2025 verisi kısmi** (Ocak–Mayıs); yıllık karşılaştırmalarda dikkatli yorumlanmalıdır.
- **Korelasyon ≠ nedensellik:** Zayıf Spearman katsayıları operasyonel kararlar için tek başına yeterli değildir; MNLogit ve Apriori bulguları birlikte değerlendirilmelidir.
- **218 interpolasyonlu gün:** Zaman serisi analizlerinde kapalı dönemler kalıntı bileşeninde yoğunlaşabilir.
- **Outlier ayrımı:** 9.037 uzun oturum ve 3.461 sıfır süreli kayıt genel analizden çıkarılmıştır; paket kanalı davranışı kısmen eksiltilmiş olabilir.

---

## 9. Çıktı Dosya Envanteri

### Veri hazırlama (`Outputs/Veri_Hazirlama/`)
| Dosya | İçerik |
|-------|--------|
| `kpi_ozet.json` | Temiz set KPI özeti |
| `kapali_gun_ozet.txt` | Kapalı gün analizi |

### Betimsel analiz (`Outputs/Analizler/01_descriptive/`)
| Dosya | İçerik |
|-------|--------|
| `betimsel_ozet.json` / `.txt` | Özet metrikler |
| `2_2_hava_yillik_trend.png` | Yıllık hava trendi |
| `2_2_yagis_kategori_yillik_stacked.png` | Yağış kategorisi yıllık dağılım |
| `2_3_gunluk_dagilim.png` | Günlük oturum dağılımı |
| `2_3_hafta_ici_sonu.png` | Hafta içi/sonu karşılaştırma |
| `2_3_mevsimsel_metrikler.png` | Mevsimsel süre/miktar |
| `2_3_saat_ogun_dagilim.png` | Saat/öğün yoğunluğu |
| `2_3_aylik_siparis_cizgi.png` | Aylık sipariş trendi |
| `2_4_masa_grup_yillik_trend.png` | Masa grubu yıllık pay |
| `2_4_hava_masa_heatmap.png` | Hava × masa ısı haritası |
| `sicaklik_masa_tercih_heatmap.png` | Sıcaklık × masa ısı haritası |
| `hava_oturum_suresi_boxplot.png` | Hava × oturum süresi boxplot |
| `betimsel_ozet_grafikleri.png` | Özet panel grafikleri |

### Zaman serisi (`Outputs/Analizler/02_zaman_serisi/`)
| Dosya | İçerik |
|-------|--------|
| `zaman_serisi_ozet.json` | ADF, decomposition, CCF özeti |
| `decompose_siparis_adedi.png` | Sipariş adedi ayrıştırma |
| `decompose_toplam_miktar.png` | Toplam miktar ayrıştırma |
| `decompose_oturum_suresi.png` | Oturum süresi ayrıştırma |
| `acf_pacf_siparis_adedi.png` | ACF/PACF grafikleri |
| `ccf_hava_masa_grubu.png` | Çapraz korelasyon grafikleri |

### Korelasyon ve regresyon (`Outputs/Analizler/03_korelasyon_regresyon/`)
| Dosya | İçerik |
|-------|--------|
| `korelasyon_regresyon_ozet.json` | Spearman, χ², MNLogit özeti |
| `korelasyon_heatmap_spearman.png` | Spearman ısı haritası |

### Birliktelik kuralları (`Outputs/Analizler/04_birliktelik_kurallari/`)
| Dosya | İçerik |
|-------|--------|
| `birliktelik_kurallari_ozet.json` / `.csv` | Apriori kural özeti |
| `hava_kume_ozet.csv` | Hava küme istatistikleri |
| `kume_karsilastirma_lift.csv` | Soğuk vs sıcak lift farkları |
| `network_soguk_yagisli.png` | Soğuk küme ağ grafiği |
| `network_sicak_gunesli.png` | Sıcak küme ağ grafiği |
| `heatmap_lift_karsilastirma.png` | Lift karşılaştırma ısı haritası |

---

## 10. Sonuç

2021 ve sonrası döneme ait IoB tabanlı analiz hattı, ham POS kayıtlarından başlayarak sistematik bir veri mühendisliği süreci (birleştirme, outlier ayrımı, enflasyon düzeltmesi, kategorik zenginleştirme) ve dört katmanlı istatistiksel analiz (betimsel → zaman serisi → regresyon → birliktelik) ile tamamlanmıştır.

**Ana sonuç:** Hava durumu, lokanta müşterilerinin **mekân tercihini** (masa grubu) ve **ürün birlikteliklerini** (sepet içeriği) istatistiksel olarak anlamlı biçimde etkilemektedir. Etki, sürekli davranış metriklerinde (süre, miktar) zayıf; kategorik tercihlerde (masa, ürün kombinasyonları) orta-güçlü düzeydedir. Bu bulgular, hava tahminine dayalı **dinamik menü planlaması**, **masa kapasitesi yönetimi** ve **cross-selling stratejileri** için bilimsel zemin sağlamaktadır.

---

*Bu rapor `00`–`04` notebook'larının çıktılarından (`Outputs/` klasörü) derlenmiştir. Güncel bulgu özeti için `rapor/bulgular.md` dosyasına bakınız.*

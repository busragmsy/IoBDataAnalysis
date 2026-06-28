# Bulgular (2021 ve Sonrasi)

## Veri hazirlama (tamamlandi)

### Outlier yonetimi

| Grup | Kayit | Kullanim |
|------|-------|----------|
| Temiz (`outlier_flag` = NaN) | 120.749 | Genel analizler (01-04) |
| `uzun_oturum` | 9.037 | Ayri uzun oturum-hava incelemesi |
| `sifir_sureli` | 3.461 | `oturum_sure_dk=0`, genel analizden haric |

Dosyalar:
- `Veriler/oturum_hava_temiz_2021_sonrasi.csv`
- `Veriler/oturum_hava_uzun_oturum_2021_sonrasi.csv`
- `Veriler/oturum_hava_tum_2021_sonrasi.csv`

### Enflasyon

Nominal medyan tutar (2021→2025): 74 TL → 820 TL  
Deflate medyan tutar (2021 baz): 90 TL → 195 TL (karsilastirilabilir)

Ana fiyat-bagimsiz hedefler: `toplam_miktar`, `urun_sayisi`

### Temiz set KPI

| Metrik | Deger |
|--------|-------|
| Donem | 2021-01-02 — 2025-05-28 |
| Ort. oturum suresi | 33,0 dk (medyan: 25 dk) |
| Ort. miktar | 5,91 |
| Ort. urun sayisi | 3,41 |
| Deflate ort. tutar | 172,9 TL |

## Analiz bulgulari

### 01 - Betimsel analiz (tamamlandi — 2.1–2.4)

**Masa grubu dagilimi:** İç Salon %39 · Bahçe %31 · Paket %30

**2.1 Sayisal ozet (bagimli degiskenler)**

| Degisken | Ort. | Medyan | Std | Min | Max | Q1 | Q3 |
|----------|------|--------|-----|-----|-----|----|----|
| toplam_miktar | 5,91 | 5,0 | 4,80 | 0,75 | 191 | 3,0 | 8,0 |
| urun_sayisi | 3,41 | 3,0 | 2,05 | 1,0 | 25 | 2,0 | 4,0 |
| oturum_sure_dk | 33,0 | 25,0 | 34,0 | 1,0 | 240 | 15,0 | 38,0 |

Hava degiskenleri ozeti: `2_1_hava_degisken_ozet.csv` · Tum birlestik tablo: `2_1_sayisal_ozet_tum.csv`

**2.2 Hava durumu — yillik (2021–2025)**

| Yil | Ort. sicaklik (°C) | Ort. nem (%) | Toplam yagis* | Ort. ruzgar | Ort. bulut | Oturum |
|-----|-------------------|--------------|---------------|-------------|------------|--------|
| 2021 | 17,0 | 65,7 | 2.831 | 16,2 | 53,8 | 28.138 |
| 2022 | 17,1 | 64,4 | 2.015 | 16,4 | 48,2 | 28.982 |
| 2023 | 18,3 | 66,0 | 2.486 | 16,9 | 54,8 | 28.794 |
| 2024 | 18,0 | 65,7 | 2.431 | 15,7 | 50,5 | 26.996 |
| 2025† | 12,5 | 65,2 | 593 | 11,9 | 54,6 | 7.839 |

*Oturum bazli `precipitation` toplami · †2025 yalnizca Ocak–Mayis

Yagis kategorisi payi (yillik ort.): Acik ~%52–58, Bulutlu ~%28–32, Yagmur ~%13–19. Kar payi 2025'te yukseldi (%2,4).

**2.3 Musteri davranisi — zaman dagilimi**

- **Yillik siparis (enflasyondan bagimsiz):** 2021: 28.138 → 2022: 28.982 (tepe) → 2024: 26.996 → 2025†: 7.839
- **Hafta sonu payi:** %17,7 (hafta sonu ort. miktar 6,47 vs hafta ici 5,79)
- **Gunluk:** En yogun Persembe (21.060), en dusuk Pazar (9.024); Pazar en yuksek ort. miktar (6,81)
- **Mevsimsel:** En uzun ort. sure Sonbahar (33,4 dk), en yuksek ort. miktar Sonbahar (6,0)
- **Saat / ogun:** Grafikler `2_3_saat_ogun_dagilim.png`, aylik cizgi `2_3_aylik_siparis_cizgi.png`

**2.4 Masa grubu davranisi**

| Yil | Bahce | Paket | Ic Salon |
|-----|-------|-------|----------|
| 2021 | %32,7 | %38,4 | %28,9 |
| 2024 | %28,9 | %29,0 | %42,1 |
| 2025† | %19,1 | %30,3 | %50,7 |

- Paket 2021'de baskin (%38), sonraki yillarda Ic Salon yukseldi (2025'te %51)
- Hava × masa capraz tablolar: `2_4_yagis_kategori_masa_crosstab_pct.csv`, `2_4_sicaklik_aralik_masa_crosstab_pct.csv`
- Heatmap: `2_4_hava_masa_heatmap.png`

**Ciktilar:** `Outputs/Analizler/01_descriptive/` (35 dosya: CSV, PNG, `betimsel_ozet.json`)

### Bölüm 3: Zaman Serisi Ayrıştırması ve Zamansal Dinamikler (02 — tamamlandi)

**Veri dönüşümü:** Oturum düzeyinden günlük frekansa (`D`) indirgeme; 1.608 takvim günü, 218 eksik gün zaman-esaslı interpolasyon ile tamamlandı. Günlük metrikler: sipariş adedi, toplam miktar (sum), ort. oturum süresi, ort. deflate tutar; CCF için sıcaklık anomalisi, bahçe/paket oranları.

**Seasonal decomposition (period=7, additive):**

| Seri | Trend (std) | Mevsimsellik (std) | Kalıntı (std) | Yorum |
|------|-------------|-------------------|---------------|-------|
| Sipariş adedi | 12,6 | 18,0 | 23,2 | Haftalık mevsimsellik belirgin; kalıntı varyansı en yüksek (dışsal şoklar) |
| Toplam miktar | 89,5 | 96,5 | 155,8 | Sepet hacminde güçlü haftalık ritim + yüksek oynaklık |
| Oturum süresi | 6,4 | 0,5 | 6,0 | Sürede haftalık döngü zayıf; trend/kalıntı baskın |

**Durağanlık (ADF — günlük sipariş adedi):** Test istatistiği = −5,92, p < 0,001 → seri %5 düzeyinde **durağan** kabul edilir; ACF/PACF (30 lag) haftalık bellek yapısını doğrular (`acf_pacf_siparis_adedi.png`).

**Çapraz korelasyon (CCF, ±14 gün):**

| İlişki | En güçlü lag | r | Yorum |
|--------|--------------|---|-------|
| Sıcaklık anomalisi ↔ Bahçe tercihi (%) | 0 gün | +0,20 | Eşzamanlı, zayıf-orta pozitif ilişki: anomalik ılıman günlerde bahçe payı artış eğilimi |
| Günlük yağış ↔ Paket oranı (%) | −4 gün | +0,07 | Gecikmeli etki istatistiksel olarak **zayıf**; yağış şokunun paket tercihine doğrudan güçlü lagged etkisi kanıtlanmadı |

**Çıktılar:** `Outputs/Analizler/02_zaman_serisi/` — `gunluk_zaman_serisi.csv`, decomposition grafikleri (×3), `acf_pacf_siparis_adedi.png`, `ccf_hava_masa_grubu.png`, `zaman_serisi_ozet.json`

### Bölüm 4: İstatistiksel Modelleme ve Davranışsal Korelasyon (03 — tamamlandi)

02'deki CCF bulgusu (lag ≈ 0) ile tutarlı olarak, bu analiz **oturum anındaki eşzamanlı** hava–tercih ilişkisine odaklanır.

**ADIM 1 — Korelasyon (Pearson / Spearman):**
- Shapiro-Wilk: tüm davranış değişkenlerinde normallik reddedildi → **birincil yorum Spearman ρ**
- İlişkiler istatistiksel olarak anlamlı olsa da **etki büyüklüğü zayıf** (|ρ| < 0,03); büyük örneklem (n = 120.749) küçük farkları anlamlı kılar
- En güçlü eşleşmeler: sıcaklık ↔ ürün sayısı (ρ = +0,026), nem ↔ ürün sayısı (ρ = −0,019)
- Heatmap: `korelasyon_heatmap_spearman.png`

**ADIM 2 — Ki-Kare (kategorik hava × masa tercihi):**

| Hava kategorisi | χ² | DoF | p | Cramér's V | Anlamlı |
|-----------------|-----|-----|---|------------|---------|
| nem_grubu | 4.076 | 6 | <0,001 | 0,130 | Evet |
| bulut_grubu | 2.529 | 4 | <0,001 | 0,102 | Evet |
| yagis_yogunlugu | 729 | 6 | <0,001 | 0,055 | Evet |
| ruzgar_seviyesi | 113 | 6 | <0,001 | 0,022 | Evet |

Tüm kategorik hava değişkenleri ile masa tercihi arasında **anlamlı ilişki** vardır; en güçlü kategorik etki **nem_grubu** (V = 0,13).

**ADIM 3 — Multinomial Logit (referans: İç Salon, pseudo-R² = 0,054):**

| Hedef | Değişken | OR | p | Yorum |
|-------|----------|----|---|-------|
| Bahçe | temperature_2m | **1,12** | <0,001 | 1°C artış → Bahçe odds'u %12 artar |
| Bahçe | precipitation | **0,88** | <0,001 | Yağış artışı Bahçe tercihini azaltır |
| Bahçe | windspeed_10m | 0,99 | <0,001 | Rüzgar artışı Bahçe odds'unu düşürür |
| Paket | temperature_2m | 1,02 | <0,001 | Hafif pozitif paket yönelimi |
| Paket | precipitation | 1,03 | 0,064 | Anlamlı değil (α = 0,05) |

**Somut bulgu:** Bahçe tercihini en güçlü **artıran** faktör **sıcaklık** (OR = 1,12); en güçlü **azaltan** faktör **yağış** (OR = 0,88). Bu, CCF'deki eşzamanlı sıcaklık–bahçe korelasyonunu (r = +0,20) regresyon düzeyinde doğrular.

**Çıktılar:** `Outputs/Analizler/03_korelasyon_regresyon/` — heatmap PNG, chi-kare/MNLogit CSV, `korelasyon_regresyon_ozet.json`

### Bölüm 5: Hava Durumu Odaklı Ürün Birliktelik Analizi (04 — tamamlandi)

**Kümeleme (sipariş bazlı transform):**

| Küme | Tanım | Oturum |
|------|-------|--------|
| Sıcak ve Güneşli | Temp > 15°C, yağışsız | 61.735 |
| Soğuk ve Yağışlı | Temp ≤ 15°C, yağışlı veya kapalı | 31.067 |
| Diğer | Ara koşullar | 27.947 |

**Apriori (min_support = 0,01, lift > 1):** Sıcak küme 400 kural · Soğuk küme 418 kural.

**Küme karşılaştırması — öne çıkan bulgu:**
> Soğuk ve Yağışlı günlerde **AZ CORBA → SU** birliktelik kuralının Lift değeri (**2,19**), Sıcak günlere göre **%23,9 daha yüksektir** (sıcak lift = 1,77). Benzer şekilde **CORBA → SU** kuralı soğukta lift **1,72** ile sıcağa göre **%13,8** daha güçlüdür.

**Soğuk/Yağışlı — Top 3 kural (lift):**

| Kural | Lift | Confidence |
|-------|------|------------|
| KAYMAK → AYRAN + KÜNEFE | 7,73 | 0,53 |
| AYRAN + KÜNEFE → KAYMAK | 7,73 | 0,20 |
| KÜNEFE + ŞİŞE AYRAN → DONDURMA | 7,63 | 0,27 |

**Sıcak/Güneşli — Top 3 kural (lift):**

| Kural | Lift | Confidence |
|-------|------|------------|
| KÜNEFE + SU → DONDURMA | 8,04 | 0,37 |
| DONDURMA → KÜNEFE + SU | 8,04 | 0,23 |
| KÜNEFE + ŞİŞE AYRAN → DONDURMA | 7,96 | 0,37 |

**Stratejik öneriler:**
1. **Soğuk/yağışlı günler:** CORBA + SU / AZ CORBA çapraz satış paketleri menü ve POS'ta öne çıkarılmalı (lift soğukta ~%14–24 daha yüksek — bilimsel kanıt).
2. **Sıcak/güneşli günler:** KÜNEFE + DONDURMA tatlı bundle'ı (lift > 7) yaz menüsü promosyonuna alınmalı.
3. **Genel:** AYRAN + FULL çekirdek birliktelik her iki kümede de güçlü; hava koşuluna göre yan ürün stratejisi değişir (soğukta CORBA, sıcakta DONDURMA).

**Çıktılar:** `Outputs/Analizler/04_birliktelik_kurallari/` — `birliktelik_kurallari_ozet.csv`, `network_soguk_yagisli.png`, `network_sicak_gunesli.png`, `heatmap_lift_karsilastirma.png`, `birliktelik_kurallari_ozet.json`

"""
KATMAN 5 — Dönem Karşılaştırması: Hava-Davranış İlişkisi Değişti mi?
======================================================================
Soru  : Pre-COVID, COVID ve Post-COVID dönemlerinde hava durumunun
        müşteri davranışı üzerindeki etkisi farklılaştı mı?
Çıktı : dönem bazlı model metrikleri, feature importance karşılaştırması,
        Cohen's d tablosu, görselleştirmeler
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings, os
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 0. AYARLAR
# ─────────────────────────────────────────
CSV_PATH   = "veri.csv"   # <── kendi dosya yolunu gir
OUTPUT_DIR = "katman5_cikti"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# 1. VERİ YÜKLE & HAZIRLA
# ─────────────────────────────────────────
print("Veri yükleniyor...")
df = pd.read_csv(CSV_PATH)

def get_donem(yil):
    if yil < 2020:    return 'Pre-COVID'
    elif yil <= 2022: return 'COVID'
    else:             return 'Post-COVID'

df['donem']    = df['yil'].apply(get_donem)
df['hafta_sonu'] = df['gun_adi'].isin(['Cumartesi', 'Pazar']).astype(int)
df['sicaklik_hissedilen_fark'] = df['temperature_2m'] - df['apparent_temperature']
df['yagis_var'] = (df['precipitation'] > 0).astype(int)
df['kar_var']   = (df['snowfall'] > 0).astype(int)

def oturum_dilimi(saat):
    if saat < 10:   return 'sabah'
    elif saat < 13: return 'ogle'
    elif saat < 17: return 'ikindi'
    else:           return 'aksam'
df['oturum_dilimi'] = df['saat'].apply(oturum_dilimi)

normal_mask = df['outlier_flag'].isna()

# Encoding (donem hariç — zaten ayrıştırıyoruz)
cat_cols = ['masa_grup', 'gun_adi', 'yagis_kategori', 'sicaklik_aralik', 'oturum_dilimi']
df_enc   = pd.get_dummies(df, columns=cat_cols, drop_first=False)
ohe_cols = [c for c in df_enc.columns if any(c.startswith(x+'_') for x in cat_cols)]

hava_f   = [
    'temperature_2m', 'apparent_temperature', 'relative_humidity_2m',
    'dewpoint_2m', 'precipitation', 'rain', 'snowfall', 'windspeed_10m',
    'cloudcover', 'pressure_msl', 'shortwave_radiation', 'is_day',
    'yagis_var', 'kar_var', 'sicaklik_hissedilen_fark'
]
zaman_f  = ['saat', 'ay', 'hafta_no', 'hafta_sonu']
all_features = hava_f + zaman_f + ohe_cols

# ─────────────────────────────────────────
# 2. COHEN'S D YARDIMCI FONKSİYON
# ─────────────────────────────────────────
def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1-1)*g1.std()**2 + (n2-1)*g2.std()**2) / (n1+n2-2))
    return (g1.mean() - g2.mean()) / pooled

def interpret_d(d):
    d = abs(d)
    if d < 0.2:   return 'çok küçük'
    elif d < 0.5: return 'küçük'
    elif d < 0.8: return 'orta'
    else:         return 'büyük'

# ─────────────────────────────────────────
# 3. DÖNEM BAZLI BETİMSEL İSTATİSTİK
# ─────────────────────────────────────────
print("\n" + "="*65)
print("DÖNEM BAZLI BETİMSEL İSTATİSTİK (Temiz Veri)")
print("="*65)

df_normal = df[normal_mask].copy()
hedefler  = ['oturum_sure_dk', 'toplam_tutar', 'toplam_miktar']

for hedef in hedefler:
    print(f"\n── {hedef} ──")
    tablo = df_normal.groupby('donem')[hedef].agg(['mean', 'median', 'std', 'count'])
    print(tablo.to_string())

# Kruskal-Wallis (dönem × oturum_sure_dk)
gruplar = [g['oturum_sure_dk'].values
           for _, g in df_normal.groupby('donem')]
stat, p = kruskal(*gruplar)
print(f"\nKruskal-Wallis (dönem × oturum_sure_dk): H={stat:.2f}, p={p:.6f}")

# ─────────────────────────────────────────
# 4. COHEN'S D — TÜM HEDEFLER × DÖNEM ÇİFTLERİ
# ─────────────────────────────────────────
print("\n" + "="*65)
print("COHEN'S D — ETKİ BÜYÜKLÜKLERİ")
print("="*65)

donemler = ['Pre-COVID', 'COVID', 'Post-COVID']
cifter   = [('Pre-COVID', 'COVID'),
            ('Pre-COVID', 'Post-COVID'),
            ('COVID', 'Post-COVID')]

for hedef in hedefler:
    print(f"\n── {hedef} ──")
    for d1, d2 in cifter:
        g1 = df_normal[df_normal['donem'] == d1][hedef].dropna()
        g2 = df_normal[df_normal['donem'] == d2][hedef].dropna()
        d  = cohens_d(g1, g2)
        _, p_mw = mannwhitneyu(g1, g2)
        print(f"  {d1} vs {d2:<20} d={d:+.3f}  ({interpret_d(d)})  p={p_mw:.4f}")

# ─────────────────────────────────────────
# 5. DÖNEM × YAĞIŞ ETKİLEŞİMİ
# ─────────────────────────────────────────
print("\n" + "="*65)
print("DÖNEM × YAĞIŞ ETKİLEŞİMİ — Günlük Oturum Sayısı (Katman 1 Tamamlayıcısı)")
print("="*65)

df_normal['tarih_dt'] = pd.to_datetime(df_normal['tarih'])
gunluk = df_normal.groupby(['tarih_dt', 'donem']).agg(
    gunluk_oturum=('MASANO', 'count'),
    yagis=('precipitation', 'mean')
).reset_index()
gunluk['yagis_var'] = (gunluk['yagis'] > 0).astype(int)

for donem in donemler:
    df_d = gunluk[gunluk['donem'] == donem]
    k    = df_d[df_d['yagis_var'] == 0]['gunluk_oturum']
    y_g  = df_d[df_d['yagis_var'] == 1]['gunluk_oturum']
    if len(y_g) > 5 and len(k) > 5:
        d = cohens_d(k, y_g)
        print(f"  {donem:<15} Kuru: {k.mean():.1f} | Yağışlı: {y_g.mean():.1f} | "
              f"Fark: {k.mean()-y_g.mean():+.1f} | d={d:.3f}")

# ─────────────────────────────────────────
# 6. DÖNEM BAZLI MODEL + FEATURE IMPORTANCE
# ─────────────────────────────────────────
print("\n" + "="*65)
print("DÖNEM BAZLI MODEL — Random Forest (oturum_sure_dk)")
print("="*65)

donem_sonuclar = {}

for donem in donemler:
    mask_d = normal_mask & (df['donem'] == donem)
    df_d   = df_enc[mask_d][all_features + ['oturum_sure_dk']].dropna()

    if len(df_d) < 500:
        print(f"\n{donem}: Yetersiz veri ({len(df_d)} satır), atlandı.")
        continue

    X_d = df_d[all_features]
    y_d = df_d['oturum_sure_dk']
    X_tr, X_te, y_tr, y_te = train_test_split(X_d, y_d, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)

    r2   = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    imp  = pd.Series(rf.feature_importances_, index=all_features).sort_values(ascending=False)

    hava_toplam = imp[imp.index.isin(hava_f)].sum()
    donem_sonuclar[donem] = {
        'n': len(df_d), 'r2': r2, 'rmse': rmse,
        'imp': imp, 'hava_toplam': hava_toplam
    }

    print(f"\n── {donem} (n={len(df_d):,}) ──")
    print(f"  R²: {r2:.4f}  |  RMSE: {rmse:.2f} dk")
    print(f"  Hava değişkenleri toplam katkısı: {hava_toplam:.4f} ({hava_toplam*100:.1f}%)")
    print(f"  Top 5 feature:")
    for feat, val in imp.head(5).items():
        print(f"    {feat:<35} {val:.4f}")

# ─────────────────────────────────────────
# 7. FEATURE IMPORTANCE KARŞILAŞTIRMA GRAFİĞİ
# ─────────────────────────────────────────
if len(donem_sonuclar) >= 2:
    fig, axes = plt.subplots(1, len(donem_sonuclar), figsize=(6*len(donem_sonuclar), 8),
                             sharey=False)
    colors = ['steelblue', 'tomato', 'seagreen']

    for ax, (donem, s), color in zip(axes, donem_sonuclar.items(), colors):
        s['imp'].head(15).sort_values().plot(kind='barh', ax=ax, color=color)
        ax.set_title(f"{donem}\nR²={s['r2']:.4f}", fontsize=11)
        ax.set_xlabel('Önem Skoru')

    plt.suptitle('Feature Importance Karşılaştırması — Dönem Bazlı', fontsize=13, y=1.01)
    plt.tight_layout()
    path_fig = os.path.join(OUTPUT_DIR, 'feature_importance_donem_karsilastirma.png')
    plt.savefig(path_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nGrafik kaydedildi: {path_fig}")

# ─────────────────────────────────────────
# 8. ÖZET TABLO
# ─────────────────────────────────────────
print("\n" + "="*65)
print("ÖZET — Dönem Bazlı Model Karşılaştırması")
print("="*65)
print(f"{'Dönem':<15} {'N':>8} {'R²':>8} {'RMSE':>8} {'Hava Katkısı':>14} {'En Önemli Hava Değişkeni':>28}")
print("-"*80)
for donem, s in donem_sonuclar.items():
    hava_imp_sorted = s['imp'][s['imp'].index.isin(hava_f)].sort_values(ascending=False)
    en_onemli = hava_imp_sorted.index[0] if len(hava_imp_sorted) > 0 else "-"
    print(f"{donem:<15} {s['n']:>8,} {s['r2']:>8.4f} {s['rmse']:>8.2f} "
          f"{s['hava_toplam']:>14.4f} {en_onemli:>28}")

print("\n✓ Katman 5 tamamlandı.")

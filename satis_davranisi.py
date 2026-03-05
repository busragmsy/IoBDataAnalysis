"""
KATMAN 3 — Satış Davranışı: toplam_tutar & toplam_miktar
=========================================================
Soru  : Hava koşulları müşterinin ne kadar harcadığını ve
        kaç ürün sipariş ettiğini etkiliyor mu?
Çıktı : model karşılaştırması, feature importance, hava kategorisi ortalamaları
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings, os
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 0. AYARLAR
# ─────────────────────────────────────────
CSV_PATH   = "veri.csv"   # <── kendi dosya yolunu gir
OUTPUT_DIR = "katman3_cikti"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# 1. VERİ YÜKLE & HAZIRLA
# ─────────────────────────────────────────
print("Veri yükleniyor...")
df = pd.read_csv(CSV_PATH)
print(f"  {df.shape[0]:,} satır")

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

# Temiz veri (outlier'sız)
normal_mask = df['outlier_flag'].isna()
df_clean = df[normal_mask].copy()
print(f"  Temiz veri: {len(df_clean):,} satır")

# Encoding
cat_cols = ['masa_grup', 'gun_adi', 'donem', 'yagis_kategori', 'sicaklik_aralik', 'oturum_dilimi']
df_enc   = pd.get_dummies(df_clean, columns=cat_cols, drop_first=False)

hava_f = [
    'temperature_2m', 'apparent_temperature', 'relative_humidity_2m',
    'dewpoint_2m', 'precipitation', 'rain', 'snowfall', 'windspeed_10m',
    'cloudcover', 'pressure_msl', 'shortwave_radiation', 'is_day',
    'yagis_var', 'kar_var', 'sicaklik_hissedilen_fark'
]
zaman_f = ['saat', 'ay', 'hafta_no', 'hafta_sonu']
ohe_f   = [c for c in df_enc.columns if any(c.startswith(x+'_') for x in cat_cols)]
all_features = hava_f + zaman_f + ohe_f

# ─────────────────────────────────────────
# 2. HEDEF DEĞİŞKENLER: Tanımlayıcı İstatistik
# ─────────────────────────────────────────
print("\n" + "="*60)
print("HEDEF DEĞİŞKEN İSTATİSTİKLERİ")
print("="*60)
for hedef in ['toplam_tutar', 'toplam_miktar']:
    print(f"\n{hedef}:")
    print(df_clean[hedef].describe().to_string())

# ─────────────────────────────────────────
# 3. HAVA KATEGORİSİNE GÖRE ORTALAMALAR
# ─────────────────────────────────────────
print("\n" + "="*60)
print("HAVA KATEGORİSİNE GÖRE ORTALAMALAR")
print("="*60)

for hedef in ['toplam_tutar', 'toplam_miktar']:
    print(f"\n── {hedef} ──")

    print("  Yağış kategorisine göre:")
    tablo = df_clean.groupby('yagis_kategori')[hedef].agg(['mean', 'median', 'count'])
    print(tablo.sort_values('mean', ascending=False).to_string())

    print("\n  Sıcaklık aralığına göre:")
    tablo2 = df_clean.groupby('sicaklik_aralik')[hedef].agg(['mean', 'median', 'count'])
    print(tablo2.sort_values('mean', ascending=False).to_string())

    print("\n  Masa grubuna göre:")
    tablo3 = df_clean.groupby('masa_grup')[hedef].agg(['mean', 'median', 'count'])
    print(tablo3.sort_values('mean', ascending=False).to_string())

    # Kruskal-Wallis
    gruplar = [g[hedef].values for _, g in df_clean.groupby('yagis_kategori')]
    stat, p = kruskal(*gruplar)
    print(f"\n  Kruskal-Wallis (yağış kategorisi): H={stat:.2f}, p={p:.6f}")

# ─────────────────────────────────────────
# 4. MODEL — Her hedef için
# ─────────────────────────────────────────
all_results = {}

for hedef in ['toplam_tutar', 'toplam_miktar']:
    print("\n" + "="*60)
    print(f"MODEL — {hedef}")
    print("="*60)

    df_m = df_enc[all_features + [hedef]].dropna()
    X    = df_m[all_features]
    y    = df_m[hedef]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression':  LinearRegression(),
        'Random Forest':      RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'Gradient Boosting':  GradientBoostingRegressor(n_estimators=200, random_state=42),
    }

    print(f"\n{'Model':<25} {'RMSE':>10} {'MAE':>10} {'R²':>8}")
    print("-"*57)

    best_r2, best_name, best_model = -999, None, None
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        mae  = mean_absolute_error(y_te, y_pred)
        r2   = r2_score(y_te, y_pred)
        print(f"{name:<25} {rmse:>10.2f} {mae:>10.2f} {r2:>8.4f}")
        if r2 > best_r2:
            best_r2, best_name, best_model = r2, name, model

    # CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(best_model, X, y, cv=kf, scoring='r2')
    print(f"\n  5-Fold CV ({best_name}): R²={cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        imp = pd.Series(best_model.feature_importances_, index=all_features).sort_values(ascending=False)

        hava_imp  = imp[imp.index.isin(hava_f)].sum()
        zaman_imp = imp[imp.index.isin(zaman_f)].sum()
        print(f"\n  Hava değişkenleri toplam katkısı: {hava_imp:.4f} ({hava_imp*100:.1f}%)")
        print(f"  Zaman değişkenleri toplam katkısı: {zaman_imp:.4f} ({zaman_imp*100:.1f}%)")

        print(f"\n  Top 15 feature ({best_name}):")
        print(imp.head(15).to_string())

        # Grafik
        fig, ax = plt.subplots(figsize=(10, 8))
        imp.head(20).sort_values().plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title(f'Feature Importance — {hedef} ({best_name})', fontsize=13)
        ax.set_xlabel('Önem Skoru')
        plt.tight_layout()
        path_fig = os.path.join(OUTPUT_DIR, f'feature_importance_{hedef}.png')
        plt.savefig(path_fig, dpi=150)
        plt.close()
        print(f"  Grafik kaydedildi: {path_fig}")

    all_results[hedef] = {'R2': best_r2, 'model': best_name}

# ─────────────────────────────────────────
# 5. ÖZET KARŞILAŞTIRMA
# ─────────────────────────────────────────
print("\n" + "="*60)
print("ÖZET — Katman 3 Model Karşılaştırması")
print("="*60)
print(f"  oturum_sure_dk (Katman 2) R²: 0.1780  ← referans")
for hedef, res in all_results.items():
    print(f"  {hedef:<20} R²: {res['R2']:.4f}  ({res['model']})")

print("\n✓ Katman 3 tamamlandı.")

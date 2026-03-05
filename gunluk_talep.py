"""
KATMAN 1 — Günlük Oturum Sayısı Tahmini
========================================
Soru  : Hava koşulları o gün kaç müşterinin geldiğini etkiliyor mu?
Çıktı : model_sonuclari, feature_importance_gunluk.png, katman1_ozet.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings, os
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 0. AYARLAR
# ─────────────────────────────────────────
CSV_PATH   = "veri.csv"          # <── kendi dosya yolunu gir
OUTPUT_DIR = "katman1_cikti"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# 1. VERİ YÜKLE
# ─────────────────────────────────────────
print("Veri yükleniyor...")
df = pd.read_csv(CSV_PATH)
print(f"  {df.shape[0]:,} satır, {df.shape[1]} kolon")

df['tarih_dt'] = pd.to_datetime(df['tarih'])

def get_donem(yil):
    if yil < 2020:   return 'Pre-COVID'
    elif yil <= 2022: return 'COVID'
    else:             return 'Post-COVID'

df['donem'] = df['yil'].apply(get_donem)

# ─────────────────────────────────────────
# 2. GÜNLÜK AGGREGATE
# ─────────────────────────────────────────
print("\nGünlük aggregate yapılıyor...")

gunluk_oturum = df.groupby('tarih_dt').size().reset_index(name='gunluk_oturum_sayisi')

hava_cols = [
    'temperature_2m', 'apparent_temperature', 'relative_humidity_2m',
    'dewpoint_2m', 'precipitation', 'rain', 'snowfall',
    'windspeed_10m', 'cloudcover', 'pressure_msl', 'shortwave_radiation'
]
gunluk_hava = df.groupby('tarih_dt')[hava_cols].mean().reset_index()
gunluk_hava['max_precipitation'] = df.groupby('tarih_dt')['precipitation'].max().values

gunluk_hava['gun_adi']    = gunluk_hava['tarih_dt'].dt.day_name()
gunluk_hava['ay']         = gunluk_hava['tarih_dt'].dt.month
gunluk_hava['yil']        = gunluk_hava['tarih_dt'].dt.year
gunluk_hava['hafta_no']   = gunluk_hava['tarih_dt'].dt.isocalendar().week.astype(int)
gunluk_hava['hafta_sonu'] = gunluk_hava['gun_adi'].isin(['Saturday', 'Sunday']).astype(int)
gunluk_hava['yagis_var']  = (gunluk_hava['precipitation'] > 0).astype(int)
gunluk_hava['kar_var']    = (gunluk_hava['snowfall'] > 0).astype(int)
gunluk_hava['sicaklik_hissedilen_fark'] = (
    gunluk_hava['temperature_2m'] - gunluk_hava['apparent_temperature']
)
gunluk_hava['donem'] = gunluk_hava['yil'].apply(get_donem)

df_gunluk = gunluk_oturum.merge(gunluk_hava, on='tarih_dt')

print(f"  {len(df_gunluk)} gün")
print(f"  Tarih: {df_gunluk['tarih_dt'].min().date()} → {df_gunluk['tarih_dt'].max().date()}")
print(f"\nGünlük oturum istatistikleri:")
print(df_gunluk['gunluk_oturum_sayisi'].describe().to_string())

# ─────────────────────────────────────────
# 3. KORELASYON ANALİZİ
# ─────────────────────────────────────────
print("\n" + "="*65)
print("KORELASYON: Hava Değişkenleri × Günlük Oturum Sayısı")
print("="*65)

hava_features_list = hava_cols + [
    'max_precipitation', 'yagis_var', 'kar_var', 'sicaklik_hissedilen_fark'
]

print(f"{'Değişken':<35} {'Pearson r':>10} {'p-değeri':>10} {'Spearman r':>11}")
print("-"*70)

corr_rows = []
for col in hava_features_list:
    r_p, p_p = pearsonr(df_gunluk[col], df_gunluk['gunluk_oturum_sayisi'])
    r_s, _   = spearmanr(df_gunluk[col], df_gunluk['gunluk_oturum_sayisi'])
    sig = "***" if p_p < 0.001 else ("**" if p_p < 0.01 else ("*" if p_p < 0.05 else "  "))
    print(f"{col:<35} {r_p:>+10.4f} {p_p:>10.4f} {r_s:>+11.4f}  {sig}")
    corr_rows.append({'feature': col, 'pearson': r_p, 'p_value': p_p, 'spearman': r_s})

print("\n* p<0.05  ** p<0.01  *** p<0.001")

# ─────────────────────────────────────────
# 4. YAĞIŞLI vs KURU GÜN
# ─────────────────────────────────────────
print("\n" + "="*65)
print("YAĞIŞLI vs KURU GÜN — Günlük Oturum Sayısı")
print("="*65)

def cohens_d(g1, g2):
    n1, n2   = len(g1), len(g2)
    pooled   = np.sqrt(((n1-1)*g1.std()**2 + (n2-1)*g2.std()**2) / (n1+n2-2))
    return (g1.mean() - g2.mean()) / pooled

kuru_g    = df_gunluk[df_gunluk['yagis_var'] == 0]['gunluk_oturum_sayisi']
yagisli_g = df_gunluk[df_gunluk['yagis_var'] == 1]['gunluk_oturum_sayisi']
_, p_mw   = mannwhitneyu(kuru_g, yagisli_g)
d_yagis   = cohens_d(kuru_g, yagisli_g)

print(f"  Kuru gün sayısı:           {len(kuru_g)}")
print(f"  Yağışlı gün sayısı:        {len(yagisli_g)}")
print(f"  Kuru ort. oturum/gün:      {kuru_g.mean():.1f}")
print(f"  Yağışlı ort. oturum/gün:   {yagisli_g.mean():.1f}")
print(f"  Fark:                      {kuru_g.mean() - yagisli_g.mean():+.1f} oturum/gün")
print(f"  Mann-Whitney p:            {p_mw:.6f}")
print(f"  Cohen's d:                 {d_yagis:.3f}")

print("\n--- Dönem Bazlı Yağış Etkisi ---")
for donem in ['Pre-COVID', 'COVID', 'Post-COVID']:
    df_d = df_gunluk[df_gunluk['donem'] == donem]
    k = df_d[df_d['yagis_var'] == 0]['gunluk_oturum_sayisi']
    y = df_d[df_d['yagis_var'] == 1]['gunluk_oturum_sayisi']
    if len(y) > 5:
        d = cohens_d(k, y)
        print(f"  {donem:<15} Kuru: {k.mean():.1f} | Yağışlı: {y.mean():.1f} | "
              f"Fark: {k.mean()-y.mean():+.1f} | d={d:.3f}")

# ─────────────────────────────────────────
# 5. MODEL
# ─────────────────────────────────────────
print("\n" + "="*65)
print("MODEL — Günlük Oturum Sayısı Tahmini")
print("="*65)

cat_cols_g = ['gun_adi', 'donem']
df_g_enc   = pd.get_dummies(df_gunluk, columns=cat_cols_g, drop_first=False)
ohe_g      = [c for c in df_g_enc.columns
               if any(c.startswith(x + '_') for x in cat_cols_g)]

sayisal = hava_cols + [
    'max_precipitation', 'yagis_var', 'kar_var', 'sicaklik_hissedilen_fark',
    'ay', 'hafta_no', 'hafta_sonu'
]
features_g = sayisal + ohe_g
target_g   = 'gunluk_oturum_sayisi'

X_g = df_g_enc[features_g]
y_g = df_g_enc[target_g]

X_tr, X_te, y_tr, y_te = train_test_split(X_g, y_g, test_size=0.2, random_state=42)

models = {
    'Linear Regression':  LinearRegression(),
    'Ridge Regression':   Ridge(alpha=1.0),
    'Random Forest':      RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting':  GradientBoostingRegressor(n_estimators=200, random_state=42),
}

print(f"\n{'Model':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-"*52)

results = {}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)
    r2   = r2_score(y_te, y_pred)
    print(f"{name:<25} {rmse:>8.2f} {mae:>8.2f} {r2:>8.4f}")
    results[name] = {'model': model, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

# ─────────────────────────────────────────
# 6. CROSS VALIDATION
# ─────────────────────────────────────────
print("\n--- 5-Fold Cross Validation ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_models = {k: v['model'] for k, v in results.items()
             if k in ['Linear Regression', 'Random Forest', 'Gradient Boosting']}

for name, model in cv_models.items():
    cv_r2   = cross_val_score(model, X_g, y_g, cv=kf, scoring='r2')
    cv_rmse = np.sqrt(-cross_val_score(model, X_g, y_g, cv=kf,
                                       scoring='neg_mean_squared_error'))
    print(f"  {name:<25} R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}   "
          f"RMSE: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")

# ─────────────────────────────────────────
# 7. FEATURE IMPORTANCE
# ─────────────────────────────────────────
rf_model = results['Random Forest']['model']
imp = pd.Series(rf_model.feature_importances_, index=features_g).sort_values(ascending=False)

print("\n--- Top 20 Feature Importance (Random Forest) ---")
print(imp.head(20).to_string())

# Grup katkıları
hava_imp  = imp[imp.index.isin(hava_cols + ['max_precipitation', 'yagis_var',
                                              'kar_var', 'sicaklik_hissedilen_fark'])]
zaman_imp = imp[imp.index.isin(['ay', 'hafta_no', 'hafta_sonu'])]
donem_imp = imp[[c for c in imp.index if 'donem' in c]]
gun_imp   = imp[[c for c in imp.index if 'gun_adi' in c]]

print("\n--- Değişken Grubu Katkıları ---")
print(f"  Hava değişkenleri:   {hava_imp.sum():.4f}  ({hava_imp.sum()*100:.1f}%)")
print(f"  Zaman (ay/hafta):    {zaman_imp.sum():.4f}  ({zaman_imp.sum()*100:.1f}%)")
print(f"  Dönem (COVID):       {donem_imp.sum():.4f}  ({donem_imp.sum()*100:.1f}%)")
print(f"  Gün adı:             {gun_imp.sum():.4f}  ({gun_imp.sum()*100:.1f}%)")

# Grafik
fig, ax = plt.subplots(figsize=(10, 8))
imp.head(20).sort_values().plot(kind='barh', ax=ax, color='darkorange')
ax.set_title('Feature Importance — Günlük Oturum Sayısı (Random Forest)', fontsize=13)
ax.set_xlabel('Önem Skoru')
plt.tight_layout()
path_fig = os.path.join(OUTPUT_DIR, 'feature_importance_gunluk.png')
plt.savefig(path_fig, dpi=150)
plt.show()
print(f"\nGrafik kaydedildi: {path_fig}")

print("\n✓ Katman 1 tamamlandı.")

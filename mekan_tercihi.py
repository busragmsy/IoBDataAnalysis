"""
KATMAN 4 — Mekan Tercihi Sınıflandırması: Bahçe vs İç Salon
=============================================================
Soru  : Hava koşullarından müşterinin Bahçe mi İç Salon mu
        seçeceği tahmin edilebilir mi?
Çıktı : classification_report, AUC-ROC, feature importance, confusion matrix
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, ConfusionMatrixDisplay)
import warnings, os
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 0. AYARLAR
# ─────────────────────────────────────────
CSV_PATH   = "veri.csv"   # <── kendi dosya yolunu gir
OUTPUT_DIR = "katman4_cikti"
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

# Sadece Bahçe ve İç Salon, outlier hariç
mask = df['outlier_flag'].isna() & df['masa_grup'].isin(['Bahçe', 'İç Salon'])
df_k4 = df[mask].copy()
df_k4['bahce_mi'] = (df_k4['masa_grup'] == 'Bahçe').astype(int)

print(f"  Katman 4 veri seti: {len(df_k4):,} oturum")
print(df_k4['masa_grup'].value_counts().to_string())
print(f"  Sınıf dengesi: {df_k4['bahce_mi'].mean():.3f} (Bahçe oranı)")

# ─────────────────────────────────────────
# 2. ENCODING & FEATURE SET
# ─────────────────────────────────────────
cat_cols = ['gun_adi', 'donem', 'yagis_kategori', 'sicaklik_aralik', 'oturum_dilimi']
df_enc   = pd.get_dummies(df_k4, columns=cat_cols, drop_first=False)
ohe_cols = [c for c in df_enc.columns if any(c.startswith(x+'_') for x in cat_cols)]

hava_f = [
    'temperature_2m', 'apparent_temperature', 'relative_humidity_2m',
    'dewpoint_2m', 'precipitation', 'rain', 'snowfall', 'windspeed_10m',
    'cloudcover', 'pressure_msl', 'shortwave_radiation', 'is_day',
    'yagis_var', 'kar_var', 'sicaklik_hissedilen_fark'
]
zaman_f  = ['saat', 'ay', 'hafta_no', 'hafta_sonu']
features = hava_f + zaman_f + ohe_cols
features = [f for f in features if f in df_enc.columns]

X = df_enc[features]
y = df_enc['bahce_mi']

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────
# 3. MODEL EĞİTİMİ
# ─────────────────────────────────────────
print("\n" + "="*60)
print("MODEL SONUÇLARI — Bahçe vs İç Salon")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=42),
}

print(f"\n{'Model':<25} {'Accuracy':>10} {'AUC-ROC':>10} {'F1-Bahçe':>10} {'F1-İçSalon':>12}")
print("-"*70)

results = {}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc     = (y_pred == y_te).mean()
    auc     = roc_auc_score(y_te, y_prob)
    report  = classification_report(y_te, y_pred, output_dict=True)
    f1_b    = report['1']['f1-score']
    f1_i    = report['0']['f1-score']

    print(f"{name:<25} {acc:>10.4f} {auc:>10.4f} {f1_b:>10.4f} {f1_i:>12.4f}")
    results[name] = {'model': model, 'acc': acc, 'auc': auc, 'f1_bahce': f1_b}

# ─────────────────────────────────────────
# 4. EN İYİ MODEL — Detaylı Rapor
# ─────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]['auc'])
best_model = results[best_name]['model']
y_pred_b   = best_model.predict(X_te)
y_prob_b   = best_model.predict_proba(X_te)[:, 1]

print(f"\n\n=== {best_name} — Sınıflandırma Raporu ===")
print(classification_report(y_te, y_pred_b, target_names=['İç Salon', 'Bahçe']))

# Confusion matrix
cm = confusion_matrix(y_te, y_pred_b)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['İç Salon', 'Bahçe'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion Matrix — {best_name}', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()

# ─────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    imp = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)

    print(f"\nTop 20 Feature Importance ({best_name}):")
    print(imp.head(20).to_string())

    hava_imp  = imp[imp.index.isin(hava_f)].sum()
    zaman_imp = imp[imp.index.isin(zaman_f)].sum()
    print(f"\nHava değişkenleri toplam katkısı:  {hava_imp:.4f} ({hava_imp*100:.1f}%)")
    print(f"Zaman değişkenleri toplam katkısı: {zaman_imp:.4f} ({zaman_imp*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(10, 8))
    imp.head(20).sort_values().plot(kind='barh', ax=ax, color='seagreen')
    ax.set_title(f'Feature Importance — Mekan Tercihi ({best_name})', fontsize=13)
    ax.set_xlabel('Önem Skoru')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_mekan.png'), dpi=150)
    plt.close()

# ─────────────────────────────────────────
# 6. HAVA KOŞULUNA GÖRE BAHÇE KULLANIM ORANI
# ─────────────────────────────────────────
print("\n" + "="*60)
print("HAVA KOŞULUNA GÖRE BAHÇE KULLANIM ORANI")
print("="*60)

print("\nYağış kategorisine göre:")
tablo = df_k4.groupby('yagis_kategori')['bahce_mi'].agg(['mean', 'count'])
tablo.columns = ['Bahçe Oranı', 'Oturum Sayısı']
print(tablo.sort_values('Bahçe Oranı', ascending=False).to_string())

print("\nSıcaklık aralığına göre:")
tablo2 = df_k4.groupby('sicaklik_aralik')['bahce_mi'].agg(['mean', 'count'])
tablo2.columns = ['Bahçe Oranı', 'Oturum Sayısı']
print(tablo2.sort_values('Bahçe Oranı', ascending=False).to_string())

print("\nOturum dilimine göre:")
tablo3 = df_k4.groupby('oturum_dilimi')['bahce_mi'].agg(['mean', 'count'])
tablo3.columns = ['Bahçe Oranı', 'Oturum Sayısı']
print(tablo3.sort_values('Bahçe Oranı', ascending=False).to_string())

# ─────────────────────────────────────────
# 7. CROSS VALIDATION
# ─────────────────────────────────────────
print("\n--- 5-Fold Stratified CV (Random Forest) ---")
skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

cv_acc = cross_val_score(rf_clf, X, y, cv=skf, scoring='accuracy')
cv_auc = cross_val_score(rf_clf, X, y, cv=skf, scoring='roc_auc')
print(f"  Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"  AUC-ROC:  {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

print("\n✓ Katman 4 tamamlandı.")

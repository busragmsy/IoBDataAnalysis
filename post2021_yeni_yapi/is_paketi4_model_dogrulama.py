"""
Is Paketi 4: Modelin Test Edilmesi ve Dogrulanmasi
====================================================

Bu script, sonuc raporunda eksik oldugu tespit edilen su analizleri kapatir:

1. Egitim/test ayrimi (zaman sirasina saygili) + zaman serisi capraz dogrulama
   (TimeSeriesSplit, k-fold yerine zaman kirilimlarina gore).
2. Hata olcutleri: RMSE, MAE, R2 (fold bazinda ortalama +/- std).
3. Onyargi / sapma (bias) analizi: residual (hata) dagilimi, kategorik
   gruplar arasi sapma (masa_grup, yagis_kategori, gun_adi, hafta_sonu).
4. Zaman serisi tanisal testleri: Durbin-Watson (otokorelasyon), ACF,
   ADF (residuallerin duraganligi).
5. Yeni model hedefi: `urun_sayisi` (satis/siparis adedi) -- H2 hipotezinin
   ("IoB + hava durumu entegrasyonu musteri davranisini yuksek dogrulukla
   ongorur") dogrudan test edildigi hedef degisken.
6. H2 hipotezi icin sayisal, otomatik uretilen bir degerlendirme raporu.

Mevcut `post2021_kapsamli_pipeline.py` dosyasini DEGISTIRMEZ; ona ek olarak
calisir ve ayni veri hazirlama / ozellik mantigini yeniden kullanir.

Calistirma (proje kokunden, ayni pipeline'daki gibi):

    python post2021_yeni_yapi/is_paketi4_model_dogrulama.py

Beklenen girdi: Veriler/oturum_hava_birlesik_2021_ve_sonrasi.csv
(pipeline ile ayni dosya; bu repoda .gitignore nedeniyle bulunmuyor,
kendi makinenizdeki veriyle calisir.)
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf, adfuller

sys.path.append(str(Path(__file__).resolve().parent))
from post2021_kapsamli_pipeline import (  # noqa: E402
    INPUT_PATH,
    OUTPUT_DIR,
    load_and_prepare_data,
)

# ---------------------------------------------------------------------------
# Yol / cikti ayarlari
# ---------------------------------------------------------------------------

VALIDATION_DIR = OUTPUT_DIR / "Model_Dogrulama"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = VALIDATION_DIR / "grafikler"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SQLITE_PATH = OUTPUT_DIR / "post2021_analiz_paketi.sqlite"
SUMMARY_JSON_PATH = VALIDATION_DIR / "is_paketi4_model_dogrulama_ozet.json"
H2_REPORT_PATH = VALIDATION_DIR / "H2_hipotez_degerlendirme.md"

TIME_COL = "tarih"
N_SPLITS = 5
HOLDOUT_RATIO = 0.20  # kronolojik son %20 -> nihai test seti (bias/residual icin)
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Ozellik listeleri (pipeline ile ayni mantik)
# ---------------------------------------------------------------------------

BASE_NUMERIC = [
    "saat",
    "ay",
    "hafta_no",
    "is_day",
    "merge_hour",
    "oturum_baslangic_saati",
    "oturum_bitis_saati",
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dewpoint_2m",
    "precipitation",
    "rain",
    "showers",
    "snowfall",
    "windspeed_10m",
    "winddirection_10m",
    "cloudcover",
    "pressure_msl",
    "shortwave_radiation",
    "hava_hissedilen_fark",
    "yagis_var",
    "ruzgar_yonu_sin",
    "ruzgar_yonu_cos",
]
BASE_CATEGORICAL = ["gun_adi", "yagis_kategori", "sicaklik_aralik", "outlier_flag"]
GROUP_COLS_FOR_BIAS = ["masa_grup", "yagis_kategori", "gun_adi"]

REG_TARGETS: Dict[str, Dict] = {
    "toplam_tutar": dict(
        numeric=BASE_NUMERIC + ["oturum_sure_dk", "toplam_miktar", "urun_sayisi"],
        categorical=BASE_CATEGORICAL + ["masa_grup"],
        label="Gelir davranisi (toplam_tutar)",
    ),
    "oturum_sure_dk": dict(
        numeric=BASE_NUMERIC + ["toplam_tutar", "toplam_miktar", "urun_sayisi"],
        categorical=BASE_CATEGORICAL + ["masa_grup"],
        label="Oturum suresi davranisi",
    ),
    "urun_sayisi": dict(
        numeric=BASE_NUMERIC + ["toplam_tutar", "toplam_miktar", "oturum_sure_dk"],
        categorical=BASE_CATEGORICAL + ["masa_grup"],
        label="Satis / siparis adedi (H2 ana hedefi)",
    ),
}


# ---------------------------------------------------------------------------
# Yardimci fonksiyonlar
# ---------------------------------------------------------------------------

def prepare_frame_with_time(
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    target: str,
    time_col: str = TIME_COL,
) -> pd.DataFrame:
    """prepare_model_frame ile ayni temizlik mantigi, ancak zaman kolonunu
    korur ve sonucu kronolojik olarak siralar (zaman serisi CV/split icin)."""

    extra = [time_col] if time_col in df.columns else []
    use_cols = list(dict.fromkeys(numeric_features + categorical_features + [target] + extra))
    use_cols = [c for c in use_cols if c in df.columns]
    model_df = df[use_cols].copy()

    for col in [c for c in numeric_features if c in model_df.columns]:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
        median_value = model_df[col].median()
        if pd.isna(median_value):
            median_value = 0.0
        model_df[col] = model_df[col].fillna(median_value)

    for col in [c for c in categorical_features if c in model_df.columns]:
        model_df[col] = model_df[col].fillna("unknown").astype(str)

    model_df = model_df.dropna(subset=[target])

    if time_col in model_df.columns:
        model_df = model_df.sort_values(time_col).reset_index(drop=True)

    return model_df


def build_pipeline(num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=350,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    min_samples_leaf=5,
                ),
            ),
        ]
    )


def time_series_cross_validate(
    model_df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    target: str,
    n_splits: int = N_SPLITS,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Zaman sirasina saygili k-katli capraz dogrulama (TimeSeriesSplit).

    Rastgele k-fold KULLANILMAZ, cunku veriler zaman serisi niteliginde
    olup ileri tarihli bilginin egitime sizmasi (data leakage) onlenir.
    """

    feature_cols = [c for c in num_cols + cat_cols if c in model_df.columns]
    X = model_df[feature_cols]
    y = model_df[target]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = build_pipeline(num_cols, cat_cols)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        fold_rows.append(
            {
                "target": target,
                "fold": fold_idx,
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }
        )

    cv_df = pd.DataFrame(fold_rows)
    summary = {
        "target": target,
        "cv_rmse_mean": float(cv_df["rmse"].mean()),
        "cv_rmse_std": float(cv_df["rmse"].std(ddof=0)),
        "cv_mae_mean": float(cv_df["mae"].mean()),
        "cv_mae_std": float(cv_df["mae"].std(ddof=0)),
        "cv_r2_mean": float(cv_df["r2"].mean()),
        "cv_r2_std": float(cv_df["r2"].std(ddof=0)),
        "n_splits": n_splits,
    }
    return cv_df, summary


def holdout_bias_diagnostics(
    model_df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    target: str,
    label: str,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """Kronolojik son dilimi nihai test seti olarak ayirip; residual,
    onyargi (bias), Durbin-Watson, ADF ve ACF diagnostiklerini uretir."""

    feature_cols = [c for c in num_cols + cat_cols if c in model_df.columns]
    n = len(model_df)
    split_point = int(n * (1 - HOLDOUT_RATIO))

    train_df = model_df.iloc[:split_point]
    test_df = model_df.iloc[split_point:]

    X_train, y_train = train_df[feature_cols], train_df[target]
    X_test, y_test = test_df[feature_cols], test_df[target]

    pipe = build_pipeline(num_cols, cat_cols)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    residuals = (y_test.to_numpy() - y_pred)

    # --- baseline karsilastirma (naif ortalama tahmini) ---
    baseline_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
    model_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    model_mae = float(mean_absolute_error(y_test, y_pred))
    model_r2 = float(r2_score(y_test, y_pred))
    improvement_pct = (
        float((baseline_rmse - model_rmse) / baseline_rmse * 100)
        if baseline_rmse > 0
        else np.nan
    )

    # --- bias / onyargi ---
    bias_overall = float(np.mean(residuals))
    bias_rows = []
    for col in GROUP_COLS_FOR_BIAS:
        if col not in test_df.columns:
            continue
        tmp = pd.DataFrame({col: test_df[col].to_numpy(), "residual": residuals})
        grouped = (
            tmp.groupby(col)["residual"]
            .agg(ortalama_hata="mean", ornek_sayisi="count")
            .reset_index()
        )
        grouped["target"] = target
        grouped["grup_degiskeni"] = col
        bias_rows.append(grouped)
    bias_by_group = (
        pd.concat(bias_rows, ignore_index=True) if bias_rows else pd.DataFrame()
    )

    # --- zaman serisi tanisal testleri (residual sirasi test setinin
    #     kronolojik sirasiyla ayni, cunku split zaman bazli yapildi) ---
    dw_stat = float(durbin_watson(residuals))

    try:
        adf_stat, adf_p, *_ = adfuller(residuals, autolag="AIC")
        adf_stat, adf_p = float(adf_stat), float(adf_p)
    except Exception:
        adf_stat, adf_p = float("nan"), float("nan")

    n_lags = min(20, max(1, len(residuals) // 3))
    try:
        acf_vals = acf(residuals, nlags=n_lags, fft=True)
    except Exception:
        acf_vals = np.array([])

    # --- grafikler ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_pred, residuals, alpha=0.4, s=10)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Tahmin Edilen Deger")
    ax.set_ylabel("Residual (Gercek - Tahmin)")
    ax.set_title(f"{label} - Residual vs Tahmin")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{target}_residual_vs_tahmin.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residual")
    ax.set_title(f"{label} - Residual Dagilimi (Bias Kontrolu)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{target}_residual_histogram.png", dpi=150)
    plt.close(fig)

    if acf_vals.size > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(len(acf_vals)), acf_vals, color="darkorange")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Gecikme (lag)")
        ax.set_ylabel("Otokorelasyon")
        ax.set_title(f"{label} - Residual ACF (Otokorelasyon)")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{target}_residual_acf.png", dpi=150)
        plt.close(fig)

    diagnostics = {
        "target": target,
        "label": label,
        "holdout_test_size": int(len(test_df)),
        "holdout_rmse": model_rmse,
        "holdout_mae": model_mae,
        "holdout_r2": model_r2,
        "baseline_rmse_mean_predictor": baseline_rmse,
        "rmse_improvement_over_baseline_pct": improvement_pct,
        "bias_overall_mean_residual": bias_overall,
        "durbin_watson": dw_stat,
        "durbin_watson_yorum": _yorumla_durbin_watson(dw_stat),
        "adf_statistic_on_residuals": adf_stat,
        "adf_pvalue_on_residuals": adf_p,
        "adf_yorum": _yorumla_adf(adf_p),
    }

    return diagnostics, bias_by_group, pd.DataFrame(
        {"target": target, "lag": range(len(acf_vals)), "acf": acf_vals}
    )


def _yorumla_durbin_watson(dw: float) -> str:
    if np.isnan(dw):
        return "hesaplanamadi"
    if 1.5 <= dw <= 2.5:
        return "onemli bir otokorelasyon belirtisi yok (~2'ye yakin)"
    if dw < 1.5:
        return "pozitif otokorelasyon belirtisi (model artik hatalari sistematik oruntu birakiyor olabilir)"
    return "negatif otokorelasyon belirtisi"


def _yorumla_adf(p_value: float) -> str:
    if np.isnan(p_value):
        return "hesaplanamadi"
    if p_value < 0.05:
        return "residualler duragan (p<0.05) -> modelin aciklayamadigi sistematik bir trend/kalinti yapi gorulmuyor"
    return "residuallerde duraganlik reddedilemedi (p>=0.05) -> modelin yakalayamadigi zamansal bir yapi olabilir"


def _yorumla_h2(cv_summary: Dict, diagnostics: Dict) -> str:
    r2 = cv_summary["cv_r2_mean"]
    improvement = diagnostics["rmse_improvement_over_baseline_pct"]

    if r2 >= 0.5 and improvement >= 20:
        seviye = "YUKSEK dogruluk destegi bulunmustur"
    elif r2 >= 0.2 or improvement >= 10:
        seviye = "ORTA duzeyde, sinirli dogruluk destegi bulunmustur"
    else:
        seviye = "H2'yi destekleyecek yeterli dogruluk bulunamamistir (dusuk R2 / dusuk iyilesme)"

    return (
        f"Capraz dogrulama ortalama R2 = {r2:.3f}, "
        f"naif ortalama tahminine gore RMSE iyilesmesi = %{improvement:.1f}. "
        f"Sonuc: {seviye}."
    )


# ---------------------------------------------------------------------------
# Ana akis
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Veri okunuyor: {INPUT_PATH}")
    df = load_and_prepare_data(INPUT_PATH)
    print(f"Toplam kayit: {len(df)}")

    all_cv_rows: List[pd.DataFrame] = []
    all_cv_summary: List[Dict] = []
    all_diagnostics: List[Dict] = []
    all_bias_rows: List[pd.DataFrame] = []
    all_acf_rows: List[pd.DataFrame] = []
    h2_lines: List[str] = []

    for target, cfg in REG_TARGETS.items():
        print(f"\n=== Hedef: {target} ({cfg['label']}) ===")
        model_df = prepare_frame_with_time(
            df, cfg["numeric"], cfg["categorical"], target, time_col=TIME_COL
        )
        if len(model_df) < 1000:
            print(f"  Yetersiz veri ({len(model_df)} satir), atlaniyor.")
            continue

        num_cols = [c for c in cfg["numeric"] if c in model_df.columns]
        cat_cols = [c for c in cfg["categorical"] if c in model_df.columns]

        cv_df, cv_summary = time_series_cross_validate(model_df, num_cols, cat_cols, target)
        print(
            f"  CV RMSE: {cv_summary['cv_rmse_mean']:.3f} (+/-{cv_summary['cv_rmse_std']:.3f}) | "
            f"MAE: {cv_summary['cv_mae_mean']:.3f} | R2: {cv_summary['cv_r2_mean']:.3f}"
        )
        all_cv_rows.append(cv_df)
        all_cv_summary.append(cv_summary)

        diagnostics, bias_by_group, acf_df = holdout_bias_diagnostics(
            model_df, num_cols, cat_cols, target, cfg["label"]
        )
        print(
            f"  Holdout RMSE: {diagnostics['holdout_rmse']:.3f} | "
            f"Baseline RMSE: {diagnostics['baseline_rmse_mean_predictor']:.3f} | "
            f"Iyilesme: %{diagnostics['rmse_improvement_over_baseline_pct']:.1f}"
        )
        print(f"  Durbin-Watson: {diagnostics['durbin_watson']:.3f} ({diagnostics['durbin_watson_yorum']})")
        print(f"  ADF (residual) p-degeri: {diagnostics['adf_pvalue_on_residuals']:.4f} ({diagnostics['adf_yorum']})")

        all_diagnostics.append(diagnostics)
        if not bias_by_group.empty:
            all_bias_rows.append(bias_by_group)
        all_acf_rows.append(acf_df)

        if target == "urun_sayisi":
            h2_lines.append(
                f"### H2 Hipotezi Degerlendirmesi (hedef: satis/siparis adedi - `urun_sayisi`)\n\n"
                + _yorumla_h2(cv_summary, diagnostics)
            )

    # --- birlestir ve kaydet ---
    cv_all_df = pd.concat(all_cv_rows, ignore_index=True) if all_cv_rows else pd.DataFrame()
    cv_summary_df = pd.DataFrame(all_cv_summary)
    diagnostics_df = pd.DataFrame(all_diagnostics)
    bias_all_df = pd.concat(all_bias_rows, ignore_index=True) if all_bias_rows else pd.DataFrame()
    acf_all_df = pd.concat(all_acf_rows, ignore_index=True) if all_acf_rows else pd.DataFrame()

    if SQLITE_PATH.exists():
        with sqlite3.connect(SQLITE_PATH) as conn:
            cv_all_df.to_sql("model_cv_fold_metrics", conn, if_exists="replace", index=False)
            cv_summary_df.to_sql("model_cv_summary", conn, if_exists="replace", index=False)
            diagnostics_df.to_sql("model_bias_diagnostics", conn, if_exists="replace", index=False)
            bias_all_df.to_sql("model_bias_by_group", conn, if_exists="replace", index=False)
            acf_all_df.to_sql("model_residual_acf", conn, if_exists="replace", index=False)
        print(f"\nYeni tablolar mevcut pakete eklendi: {SQLITE_PATH}")
    else:
        print(f"\nUYARI: {SQLITE_PATH} bulunamadi. Once ana pipeline'i calistirin, "
              "yoksa sonuclar yalnizca JSON/CSV olarak kaydedilecek.")
        cv_all_df.to_csv(VALIDATION_DIR / "model_cv_fold_metrics.csv", index=False)
        diagnostics_df.to_csv(VALIDATION_DIR / "model_bias_diagnostics.csv", index=False)
        bias_all_df.to_csv(VALIDATION_DIR / "model_bias_by_group.csv", index=False)

    summary_payload = {
        "aciklama": "Is Paketi 4 - Model test/dogrulama sonuclari",
        "cv_summary": cv_summary_df.to_dict(orient="records"),
        "holdout_diagnostics": diagnostics_df.to_dict(orient="records"),
    }
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
    print(f"Ozet JSON yazildi: {SUMMARY_JSON_PATH}")

    report_lines = [
        "# Is Paketi 4 - Modelin Test Edilmesi ve Dogrulanmasi\n",
        "Bu rapor `is_paketi4_model_dogrulama.py` script'i tarafindan otomatik uretilmistir.\n",
        "## Capraz Dogrulama Ozeti (TimeSeriesSplit, 5 kat)\n",
        cv_summary_df.to_markdown(index=False) if not cv_summary_df.empty else "_veri yok_",
        "\n## Holdout Seti Tanisal Sonuclari (Bias, Durbin-Watson, ADF)\n",
        diagnostics_df.to_markdown(index=False) if not diagnostics_df.empty else "_veri yok_",
        "\n" + "\n\n".join(h2_lines) if h2_lines else "",
        "\n\nGrafikler icin bkz: `Outputs/Post2021_YeniYapi/Model_Dogrulama/grafikler/`",
    ]
    with open(H2_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"H2 degerlendirme raporu yazildi: {H2_REPORT_PATH}")


if __name__ == "__main__":
    main()

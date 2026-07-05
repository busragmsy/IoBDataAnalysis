"""
Is Paketi 5: Hava Durumunun Marjinal Katkisi ve Ek Sizinti Kontrolu
====================================================================

Bu script, `post2021_kapsamli_pipeline.py` ve `is_paketi4_model_dogrulama.py`
dosyalarini DEGISTIRMEZ; ikisini de import edip ayni veri hazirlama, ayni
zaman-serisi metodolojisini (TimeSeriesSplit, Durbin-Watson, ADF, bias)
yeniden kullanir.

Kapsam:

1. H2'nin asil sorusu: hava durumu TEK BASINA ne kadar katki sagliyor?
   `masa_grup` hedefi icin 4 model karsilastirmasi:
     - weather_only   (sadece hava + zaman degiskenleri)
     - behavior_only  (hava haric, sadece davranissal/finansal degiskenler)
     - full_model     (hepsi bir arada)
     - naive_baseline (en sik sinif, kronolojik ayrimla)
   Zaman sirasina saygili (kronolojik) egitim/test ayrimi + zaman-serisi CV
   kullanilir; is_paketi4'teki rastgele-olmayan yaklasimla tutarlidir.

2. `oturum_sure_dk` ve `urun_sayisi` icin sizinti supheси testi: es-zamanli
   sonuc degiskenleri (toplam_tutar, toplam_miktar, urun_sayisi / oturum_sure_dk)
   modelden cikarilinca R^2 nasil degisiyor?

Calistirma (proje kokunden):

    python post2021_yeni_yapi/is_paketi5_hava_katkisi.py

Beklenen girdi: Veriler/oturum_hava_birlesik_2021_ve_sonrasi.csv
(pipeline ile ayni; .gitignore nedeniyle repoda yok, yerel veriyle calisir.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.append(str(Path(__file__).resolve().parent))

# --- Mevcut dosyalar DEGISTIRILMEDEN import ediliyor ---
from post2021_kapsamli_pipeline import (  # noqa: E402
    INPUT_PATH,
    OUTPUT_DIR,
    load_and_prepare_data,
)
from is_paketi4_model_dogrulama import (  # noqa: E402
    TIME_COL,
    RANDOM_STATE,
    N_SPLITS,
    HOLDOUT_RATIO,
    prepare_frame_with_time,
    time_series_cross_validate,
    holdout_bias_diagnostics,
)

# ---------------------------------------------------------------------------
# Yol / cikti ayarlari (is_paketi4 ile ayni ust klasor, ayri alt klasor)
# ---------------------------------------------------------------------------

VALIDATION_DIR = OUTPUT_DIR / "Model_Dogrulama"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_JSON_PATH = VALIDATION_DIR / "is_paketi4_hava_katkisi_ozet.json"
H2_MARGINAL_REPORT_PATH = VALIDATION_DIR / "H2_hava_marjinal_katki_raporu.md"

# ---------------------------------------------------------------------------
# Degisken gruplari
# ---------------------------------------------------------------------------

WEATHER_TIME_NUMERIC = [
    "saat", "ay", "hafta_no", "is_day", "merge_hour",
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "dewpoint_2m", "precipitation", "rain", "showers", "snowfall",
    "windspeed_10m", "winddirection_10m", "cloudcover", "pressure_msl",
    "shortwave_radiation", "hava_hissedilen_fark", "yagis_var",
    "ruzgar_yonu_sin", "ruzgar_yonu_cos",
]
WEATHER_TIME_CATEGORICAL = ["gun_adi", "yagis_kategori", "sicaklik_aralik", "outlier_flag"]

# masa_grup icin davranissal/finansal grup (hava haric)
BEHAVIOR_NUMERIC_FOR_MASAGRUP = ["oturum_sure_dk", "toplam_miktar", "toplam_tutar", "urun_sayisi"]

# Not: oturum_baslangic_saati / oturum_bitis_saati BILINCLI OLARAK burada yok;
# bunlar acilis/kapama saatinden turetildigi icin leakage yaratiyordu (bkz.
# onceki calisma notlari). masa_grup hedefinde leakage yaratmiyor olsalar da
# tutarlilik icin tum script boyunca disarida tutuluyor.
FULL_NUMERIC_MASAGRUP = WEATHER_TIME_NUMERIC + BEHAVIOR_NUMERIC_FOR_MASAGRUP
FULL_CATEGORICAL_MASAGRUP = WEATHER_TIME_CATEGORICAL


# ---------------------------------------------------------------------------
# Siniflandirma icin zaman-serisi tutarli yardimcilar
# (is_paketi4'teki regresyon fonksiyonlarinin siniflandirma karsiligi)
# ---------------------------------------------------------------------------

def build_classification_pipeline(num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[
        ("prep", preprocessor),
        ("rf", RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=5
        )),
    ])


def time_series_cross_validate_classification(
    model_df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    target: str,
    n_splits: int = N_SPLITS,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """masa_grup icin TimeSeriesSplit tabanli capraz dogrulama.

    is_paketi4.time_series_cross_validate ile ayni mantik, ancak
    accuracy/f1 metrikleri uzerinden (regresyon degil siniflandirma)."""

    feature_cols = [c for c in num_cols + cat_cols if c in model_df.columns]
    X = model_df[feature_cols]
    y = model_df[target]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Egitim setinde gorulmeyen bir sinif test setinde varsa fold atlanir
        if not set(y_test.unique()).issubset(set(y_train.unique())):
            continue

        pipe = build_classification_pipeline(num_cols, cat_cols)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="macro"))

        fold_rows.append({
            "target": target,
            "fold": fold_idx,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "accuracy": acc,
            "f1_macro": f1,
        })

    cv_df = pd.DataFrame(fold_rows)
    if cv_df.empty:
        summary = {
            "target": target, "cv_accuracy_mean": np.nan, "cv_accuracy_std": np.nan,
            "cv_f1_macro_mean": np.nan, "cv_f1_macro_std": np.nan, "n_splits": 0,
        }
    else:
        summary = {
            "target": target,
            "cv_accuracy_mean": float(cv_df["accuracy"].mean()),
            "cv_accuracy_std": float(cv_df["accuracy"].std(ddof=0)),
            "cv_f1_macro_mean": float(cv_df["f1_macro"].mean()),
            "cv_f1_macro_std": float(cv_df["f1_macro"].std(ddof=0)),
            "n_splits": int(len(cv_df)),
        }
    return cv_df, summary


def holdout_classification_diagnostics(
    model_df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    target: str,
    label: str,
    use_dummy: bool = False,
) -> Dict:
    """Kronolojik son dilimi holdout test seti olarak ayirip accuracy/F1/AUC
    ve naif (en sik sinif) baseline karsilastirmasi uretir.

    use_dummy=True ise gercek ozellikler yerine DummyClassifier(most_frequent)
    egitilir -- naive_baseline varyanti icin kullanilir.
    """

    feature_cols = [c for c in num_cols + cat_cols if c in model_df.columns]
    n = len(model_df)
    split_point = int(n * (1 - HOLDOUT_RATIO))

    train_df = model_df.iloc[:split_point]
    test_df = model_df.iloc[split_point:]

    y_train, y_test = train_df[target], test_df[target]

    if use_dummy:
        dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
        dummy.fit(train_df[[target]], y_train)  # gercek ozellik kullanilmiyor
        y_pred = dummy.predict(test_df[[target]])
        y_prob = None
    else:
        X_train, X_test = train_df[feature_cols], test_df[feature_cols]
        pipe = build_classification_pipeline(num_cols, cat_cols)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        try:
            y_prob = pipe.predict_proba(X_test)
        except Exception:
            y_prob = None

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro"))

    auc = np.nan
    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            auc = np.nan

    return {
        "target": target,
        "label": label,
        "holdout_test_size": int(len(test_df)),
        "holdout_accuracy": acc,
        "holdout_f1_macro": f1,
        "holdout_auc_ovr_macro": auc,
    }


# ---------------------------------------------------------------------------
# Ana akis
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Veri okunuyor: {INPUT_PATH}")
    df = load_and_prepare_data(INPUT_PATH)
    print(f"Toplam kayit: {len(df)}")

    all_cv_summaries: List[Dict] = []
    all_holdout: List[Dict] = []
    marginal_notes: List[str] = []

    # =======================================================================
    # BOLUM 1: masa_grup icin 4 model karsilastirmasi (H2'nin asil sorusu)
    # =======================================================================
    print("\n=== masa_grup: weather_only / behavior_only / full_model / naive_baseline ===")

    cls_target = "masa_grup"
    variants = {
        "weather_only": (WEATHER_TIME_NUMERIC, WEATHER_TIME_CATEGORICAL),
        "behavior_only": (BEHAVIOR_NUMERIC_FOR_MASAGRUP, []),
        "full_model": (FULL_NUMERIC_MASAGRUP, FULL_CATEGORICAL_MASAGRUP),
    }

    results_by_variant: Dict[str, Dict] = {}

    for variant_name, (num_feats, cat_feats) in variants.items():
        model_df = prepare_frame_with_time(df, num_feats, cat_feats, cls_target, time_col=TIME_COL)
        if len(model_df) < 1000 or model_df[cls_target].nunique() < 2:
            print(f"  [atlandi] {variant_name}: yetersiz veri veya tek sinif")
            continue

        num_cols = [c for c in num_feats if c in model_df.columns]
        cat_cols = [c for c in cat_feats if c in model_df.columns]

        cv_df, cv_summary = time_series_cross_validate_classification(
            model_df, num_cols, cat_cols, cls_target
        )
        cv_summary["variant"] = variant_name
        print(f"  [{variant_name}] CV accuracy={cv_summary['cv_accuracy_mean']:.3f} "
              f"(+/-{cv_summary['cv_accuracy_std']:.3f})")

        holdout = holdout_classification_diagnostics(
            model_df, num_cols, cat_cols, cls_target, label=f"masa_grup ({variant_name})"
        )
        holdout["variant"] = variant_name
        print(f"  [{variant_name}] Holdout accuracy={holdout['holdout_accuracy']:.3f} "
              f"F1_macro={holdout['holdout_f1_macro']:.3f} AUC={holdout['holdout_auc_ovr_macro']}")

        all_cv_summaries.append(cv_summary)
        all_holdout.append(holdout)
        results_by_variant[variant_name] = holdout

    # naive_baseline: full_model ile ayni veri cercevesi, DummyClassifier
    baseline_model_df = prepare_frame_with_time(
        df, FULL_NUMERIC_MASAGRUP, FULL_CATEGORICAL_MASAGRUP, cls_target, time_col=TIME_COL
    )
    if len(baseline_model_df) >= 1000 and baseline_model_df[cls_target].nunique() >= 2:
        baseline_holdout = holdout_classification_diagnostics(
            baseline_model_df, [], [], cls_target, label="masa_grup (naive_baseline)", use_dummy=True
        )
        baseline_holdout["variant"] = "naive_baseline"
        print(f"  [naive_baseline] Holdout accuracy={baseline_holdout['holdout_accuracy']:.3f} "
              f"F1_macro={baseline_holdout['holdout_f1_macro']:.3f}")
        all_holdout.append(baseline_holdout)
        results_by_variant["naive_baseline"] = baseline_holdout

    # Marjinal katki notlari
    if "full_model" in results_by_variant and "behavior_only" in results_by_variant:
        delta = (results_by_variant["full_model"]["holdout_accuracy"]
                 - results_by_variant["behavior_only"]["holdout_accuracy"])
        marginal_notes.append(
            f"- **Hava/zaman degiskenlerinin davranissal modele EK katkisi**: "
            f"full_model ({results_by_variant['full_model']['holdout_accuracy']:.3f}) - "
            f"behavior_only ({results_by_variant['behavior_only']['holdout_accuracy']:.3f}) "
            f"= **{delta:+.3f}** (holdout accuracy puani)."
        )
    if "weather_only" in results_by_variant and "naive_baseline" in results_by_variant:
        delta_wb = (results_by_variant["weather_only"]["holdout_accuracy"]
                    - results_by_variant["naive_baseline"]["holdout_accuracy"])
        marginal_notes.append(
            f"- **Hava durumunun TEK BASINA naif taban cizgisine gore katkisi**: "
            f"weather_only ({results_by_variant['weather_only']['holdout_accuracy']:.3f}) - "
            f"naive_baseline ({results_by_variant['naive_baseline']['holdout_accuracy']:.3f}) "
            f"= **{delta_wb:+.3f}**."
        )
    if "behavior_only" in results_by_variant and "naive_baseline" in results_by_variant:
        delta_bb = (results_by_variant["behavior_only"]["holdout_accuracy"]
                    - results_by_variant["naive_baseline"]["holdout_accuracy"])
        marginal_notes.append(
            f"- **Davranissal/finansal degiskenlerin taban cizgisine gore katkisi**: "
            f"behavior_only ({results_by_variant['behavior_only']['holdout_accuracy']:.3f}) - "
            f"naive_baseline ({results_by_variant['naive_baseline']['holdout_accuracy']:.3f}) "
            f"= **{delta_bb:+.3f}**."
        )

    # =======================================================================
    # BOLUM 2: es-zamanli sonuc degiskenleri sizinti testi
    #          (oturum_sure_dk, urun_sayisi, toplam_tutar icin ayni mantik)
    # =======================================================================
    #
    # NEDEN BU TEST GEREKLI: is_paketi4_model_dogrulama.py'deki REG_TARGETS
    # her 3 hedef icin de birbirini "es-zamanli sonuc degiskeni" olarak
    # ozellik listesine ekliyor (orn. urun_sayisi tahmininde toplam_tutar
    # VE toplam_miktar kullaniliyor). Bu degiskenler hedefle ayni oturumun
    # SONUNDA olusuyor ve cogu zaman hedefle mekanik/neredeyse deterministik
    # iliskili (toplam_miktar ~ urun_sayisi; toplam_tutar = miktar x fiyat).
    # Bu yuzden is_paketi4'un urun_sayisi (H2 ana hedefi) icin raporladigi
    # R^2/RMSE sonuclari da, oturum_sure_dk'de bulunan sizintiyla AYNI TURDEN
    # bir sizinti tasiyabilir. Asagidaki blok, her hedef icin bu degiskenler
    # VARKEN ve YOKKEN performansi karsilastirir; H2'nin nihai degerlendirmesi
    # icin "yokken" (yalnizca hava + zaman + masa_grup ile) elde edilen sonuc
    # esas alinmalidir.

    CONCURRENT_OUTCOME_COLS = {
        "oturum_sure_dk": ["toplam_tutar", "toplam_miktar", "urun_sayisi"],
        "urun_sayisi": ["toplam_tutar", "toplam_miktar", "oturum_sure_dk"],
        "toplam_tutar": ["toplam_miktar", "urun_sayisi", "oturum_sure_dk"],
    }

    def sizinti_karsilastirma(target: str, concurrent_cols: List[str]) -> Dict[str, Dict]:
        print(f"\n=== {target}: es-zamanli sonuc degiskenleri "
              f"({', '.join(concurrent_cols)}) CIKARILINCA ne oluyor? ===")

        target_variants = {
            "with_concurrent_outcomes": dict(
                numeric=WEATHER_TIME_NUMERIC + concurrent_cols,
                categorical=WEATHER_TIME_CATEGORICAL + ["masa_grup"],
                label=f"{target} ({'/'.join(concurrent_cols)} DAHIL)",
            ),
            "without_concurrent_outcomes": dict(
                numeric=WEATHER_TIME_NUMERIC,
                categorical=WEATHER_TIME_CATEGORICAL + ["masa_grup"],
                label=f"{target} ({'/'.join(concurrent_cols)} HARIC -- sadece hava+zaman+masa_grup)",
            ),
        }

        target_results: Dict[str, Dict] = {}

        for variant_name, cfg in target_variants.items():
            model_df = prepare_frame_with_time(
                df, cfg["numeric"], cfg["categorical"], target, time_col=TIME_COL
            )
            if len(model_df) < 1000:
                print(f"  [atlandi] {variant_name}: yetersiz veri")
                continue

            num_cols = [c for c in cfg["numeric"] if c in model_df.columns]
            cat_cols = [c for c in cfg["categorical"] if c in model_df.columns]

            cv_df, cv_summary = time_series_cross_validate(model_df, num_cols, cat_cols, target)
            cv_summary["variant"] = variant_name
            print(f"  [{variant_name}] CV R2={cv_summary['cv_r2_mean']:.3f} "
                  f"(+/-{cv_summary['cv_r2_std']:.3f}) RMSE={cv_summary['cv_rmse_mean']:.3f}")

            diagnostics, _bias_by_group, _acf_df = holdout_bias_diagnostics(
                model_df, num_cols, cat_cols, target, cfg["label"]
            )
            diagnostics["variant"] = variant_name
            print(f"  [{variant_name}] Holdout R2={diagnostics['holdout_r2']:.3f} "
                  f"RMSE={diagnostics['holdout_rmse']:.3f} DW={diagnostics['durbin_watson']:.3f} "
                  f"({diagnostics['durbin_watson_yorum']})")

            all_cv_summaries.append(cv_summary)
            all_holdout.append(diagnostics)
            target_results[variant_name] = diagnostics

        if "with_concurrent_outcomes" in target_results and "without_concurrent_outcomes" in target_results:
            r2_with = target_results["with_concurrent_outcomes"]["holdout_r2"]
            r2_without = target_results["without_concurrent_outcomes"]["holdout_r2"]
            fark = r2_with - r2_without
            marginal_notes.append(
                f"- **{target} icin sizinti testi**: es-zamanli sonuc degiskenleri "
                f"({', '.join(concurrent_cols)}) modelde iken holdout R^2={r2_with:.3f}; "
                f"bu degiskenler CIKARILINCA (sadece hava+zaman+masa_grup) "
                f"holdout R^2={r2_without:.3f} (fark: {fark:+.3f}). "
                + (
                    "Buyuk dusus, onceki yuksek R^2'nin buyuk olcude es-zamanli "
                    "sonuc degiskenlerinden kaynaklandigini gosterir; hava "
                    "durumunun kendisi bu hedefi yuksek dogrulukla aciklamiyor. "
                    "H2 degerlendirmesinde bu hedef icin 'without_concurrent_outcomes' "
                    "sonucu esas alinmalidir."
                    if fark > 0.15
                    else "Fark buyuk degilse, es-zamanli degiskenlerin katkisi "
                    "sinirlidir ve hava+zaman degiskenleri tek basina da benzer "
                    "bir aciklama gucune sahiptir."
                )
            )

        return target_results

    all_target_results: Dict[str, Dict[str, Dict]] = {}
    for target, concurrent_cols in CONCURRENT_OUTCOME_COLS.items():
        all_target_results[target] = sizinti_karsilastirma(target, concurrent_cols)

    # =======================================================================
    # Ciktilarin kaydedilmesi
    # =======================================================================
    cv_summary_df = pd.DataFrame(all_cv_summaries)
    holdout_df = pd.DataFrame(all_holdout)

    summary_payload = {
        "aciklama": "Is Paketi 5 - Hava durumu marjinal katkisi ve ek sizinti kontrolu",
        "cv_summary": json.loads(cv_summary_df.to_json(orient="records", force_ascii=False)) if not cv_summary_df.empty else [],
        "holdout_results": json.loads(holdout_df.to_json(orient="records", force_ascii=False)) if not holdout_df.empty else [],
        "marginal_katki_notlari": marginal_notes,
    }
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
    print(f"\nOzet JSON yazildi: {SUMMARY_JSON_PATH}")

    # --- H2 nihai (sizinti-duzeltilmis) verdiği: urun_sayisi, es-zamanli
    #     sonuc degiskenleri OLMADAN (yalnizca hava+zaman+masa_grup) ---
    h2_final_verdict_lines = []
    urun_sayisi_clean = all_target_results.get("urun_sayisi", {}).get("without_concurrent_outcomes")
    if urun_sayisi_clean:
        r2 = urun_sayisi_clean["holdout_r2"]
        improvement = urun_sayisi_clean["rmse_improvement_over_baseline_pct"]
        if r2 >= 0.5 and improvement >= 20:
            seviye = "YUKSEK dogruluk destegi bulunmustur"
        elif r2 >= 0.2 or improvement >= 10:
            seviye = "ORTA duzeyde, sinirli dogruluk destegi bulunmustur"
        else:
            seviye = "H2'yi destekleyecek yeterli dogruluk bulunamamistir (dusuk R2 / dusuk iyilesme)"
        h2_final_verdict_lines = [
            "\n## H2 Hipotezi -- Nihai (Sizinti-Duzeltilmis) Degerlendirme\n",
            (
                "`is_paketi4_model_dogrulama.py`'nin `urun_sayisi` icin raporladigi H2 "
                "sonucu, es-zamanli sonuc degiskenlerini (toplam_tutar, toplam_miktar, "
                "oturum_sure_dk) ozellik olarak icerdigi icin sizintili olabilir "
                "(bkz. yukaridaki sizinti testi). Bu nedenle H2'nin nihai degerlendirmesi "
                "icin, **yalnizca hava durumu + zaman + masa_grup degiskenleriyle** "
                "(es-zamanli sonuc degiskenleri OLMADAN) elde edilen sonuc esas alinmistir:\n"
            ),
            (
                f"- Holdout R^2 = {r2:.3f}\n"
                f"- Naif ortalama tahminine gore RMSE iyilesmesi = %{improvement:.1f}\n"
                f"- Sonuc: **{seviye}**\n"
            ),
            (
                "Onemli: bu sonuc, `is_paketi4`'un kendi H2 raporundaki (sizintili) "
                "sayilardan farkli olabilir. Rapor yazilirken bu bolumdeki sayilar "
                "esas alinmali, `is_paketi4`'un urun_sayisi icin verdigi orijinal "
                "R2/RMSE sayilari 'sizinti supheli' olarak isaretlenmelidir."
            ),
        ]

    def _target_table(target: str, cols: List[str]) -> str:
        if holdout_df.empty or "target" not in holdout_df.columns:
            return "_veri yok_"
        sub = holdout_df[(holdout_df["target"] == target) & (holdout_df["variant"].isin(
            ["with_concurrent_outcomes", "without_concurrent_outcomes"]
        ))]
        if sub.empty:
            return "_veri yok_"
        return sub[cols].to_markdown(index=False)

    reg_cols = ["variant", "holdout_r2", "holdout_rmse", "holdout_mae",
                "rmse_improvement_over_baseline_pct", "durbin_watson", "durbin_watson_yorum"]

    report_lines = [
        "# Is Paketi 5 - Hava Durumunun Marjinal Katkisi ve Ek Sizinti Kontrolu\n",
        "Bu rapor `is_paketi5_hava_katkisi.py` script'i tarafindan otomatik uretilmistir.\n",
        "## masa_grup: 4 Model Karsilastirmasi (zaman sirasina saygili holdout)\n",
        (holdout_df[holdout_df["target"] == "masa_grup"][
            ["variant", "holdout_accuracy", "holdout_f1_macro", "holdout_auc_ovr_macro", "holdout_test_size"]
        ].to_markdown(index=False)
         if not holdout_df.empty and "target" in holdout_df.columns else "_veri yok_"),
        "\n## oturum_sure_dk: Es-zamanli Sonuc Degiskenleri Testi\n",
        _target_table("oturum_sure_dk", reg_cols),
        "\n## urun_sayisi: Es-zamanli Sonuc Degiskenleri Testi (H2 ana hedefi)\n",
        _target_table("urun_sayisi", reg_cols),
        "\n## toplam_tutar: Es-zamanli Sonuc Degiskenleri Testi (tutarlilik kontrolu)\n",
        _target_table("toplam_tutar", reg_cols),
        "\n## Marjinal Katki ve Sizinti Testi Notlari\n",
        "\n".join(marginal_notes) if marginal_notes else "_yeterli veri yok_",
        "\n".join(h2_final_verdict_lines),
    ]
    with open(H2_MARGINAL_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"H2 marjinal katki raporu yazildi: {H2_MARGINAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
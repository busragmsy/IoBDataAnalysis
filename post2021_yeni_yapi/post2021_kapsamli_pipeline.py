"""
Post-2021 IoB comprehensive analysis pipeline.

Goal:
- Build a new end-to-end structure for 2021+ data.
- Use all available columns in meaningful analyses.
- Produce a single analysis package file (SQLite) plus one JSON executive summary.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "Veriler" / "oturum_hava_birlesik_2021_ve_sonrasi.csv"
OUTPUT_DIR = PROJECT_ROOT / "Outputs" / "Post2021_YeniYapi"
SQLITE_PATH = OUTPUT_DIR / "post2021_analiz_paketi.sqlite"
SUMMARY_JSON_PATH = OUTPUT_DIR / "post2021_analiz_ozet.json"


ORIGINAL_COLUMNS = [
    "MASANO",
    "CEKNO",
    "acilis_datetime",
    "kapama_datetime",
    "oturum_sure_dk",
    "toplam_miktar",
    "toplam_tutar",
    "urun_sayisi",
    "urun_listesi",
    "masa_grup",
    "tarih",
    "saat",
    "gun_adi",
    "ay",
    "yil",
    "hafta_no",
    "merge_saati",
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
    "is_day",
    "shortwave_radiation",
    "yagis_kategori",
    "sicaklik_aralik",
    "outlier_flag",
]


COLUMN_USAGE_MAP: Dict[str, Tuple[str, str, str]] = {
    "MASANO": (
        "session_identity",
        "table-level usage density and repeated seating analysis",
        "daily_summary,kpi_summary",
    ),
    "CEKNO": (
        "session_identity",
        "check/session uniqueness, duplicate control, session counting",
        "daily_summary,data_quality,kpi_summary",
    ),
    "acilis_datetime": (
        "time_anchor",
        "session start timing and start-hour behavior",
        "daily_summary,parameter_behavior_long",
    ),
    "kapama_datetime": (
        "time_anchor",
        "session end timing and end-hour behavior",
        "daily_summary,parameter_behavior_long",
    ),
    "oturum_sure_dk": (
        "behavior_target",
        "dwell-time analysis, kruskal tests, regression target",
        "parameter_behavior_long,kruskal_scores,model_metrics",
    ),
    "toplam_miktar": (
        "basket_target",
        "basket quantity behavior and explanatory modeling",
        "parameter_behavior_long,kruskal_scores,model_metrics",
    ),
    "toplam_tutar": (
        "revenue_target",
        "revenue behavior and explanatory modeling",
        "parameter_behavior_long,kruskal_scores,model_metrics",
    ),
    "urun_sayisi": (
        "basket_target",
        "basket breadth and product diversity patterns",
        "parameter_behavior_long,kruskal_scores,model_metrics",
    ),
    "urun_listesi": (
        "basket_text",
        "product tokenization and context-aware top-product analysis",
        "product_top_overall,product_top_context",
    ),
    "masa_grup": (
        "preference_target",
        "core preference target for grouping and classification",
        "parameter_behavior_long,association_scores,model_metrics",
    ),
    "tarih": (
        "time_anchor",
        "daily aggregation and open/closed-day coverage",
        "daily_summary,kpi_summary",
    ),
    "saat": (
        "time_feature",
        "intra-day preference changes and model feature",
        "parameter_behavior_long,model_metrics",
    ),
    "gun_adi": (
        "time_feature",
        "weekday/weekend preference structure",
        "parameter_behavior_long,association_scores,model_metrics",
    ),
    "ay": (
        "time_feature",
        "seasonality and month-level preference drift",
        "parameter_behavior_long,model_metrics",
    ),
    "yil": (
        "scope_feature",
        "post-2021 scope validation",
        "kpi_summary,data_quality",
    ),
    "hafta_no": (
        "time_feature",
        "weekly rhythm and demand oscillation",
        "parameter_behavior_long,model_metrics",
    ),
    "merge_saati": (
        "integration_quality",
        "weather-join timestamp quality and merge hour effects",
        "data_quality,parameter_behavior_long",
    ),
    "temperature_2m": (
        "weather_numeric",
        "raw temperature effect and quantile-bin behavior",
        "parameter_behavior_long,association_scores,model_metrics",
    ),
    "apparent_temperature": (
        "weather_numeric",
        "felt-temperature effect and thermal comfort gap",
        "parameter_behavior_long,model_metrics",
    ),
    "relative_humidity_2m": (
        "weather_numeric",
        "humidity effect and comfort interaction",
        "parameter_behavior_long,association_scores,model_metrics",
    ),
    "dewpoint_2m": (
        "weather_numeric",
        "air moisture comfort and preference association",
        "parameter_behavior_long,association_scores,model_metrics",
    ),
    "precipitation": (
        "weather_numeric",
        "rain volume sensitivity and preference shifts",
        "parameter_behavior_long,association_scores,model_metrics",
    ),
    "rain": (
        "weather_numeric",
        "rain-specific intensity behavior",
        "parameter_behavior_long,association_scores,model_metrics",
    ),
    "showers": (
        "weather_numeric",
        "shower intensity behavior",
        "parameter_behavior_long,association_scores,model_metrics",
    ),
    "snowfall": (
        "weather_numeric",
        "snow impact on channel/space preference",
        "parameter_behavior_long,association_scores,model_metrics",
    ),
    "windspeed_10m": (
        "weather_numeric",
        "wind-speed sensitivity in seating preference",
        "parameter_behavior_long,association_scores,model_metrics",
    ),
    "winddirection_10m": (
        "weather_numeric",
        "wind-direction encoded effects (sin/cos)",
        "parameter_behavior_long,model_metrics",
    ),
    "cloudcover": (
        "weather_numeric",
        "cloudiness comfort and daylight substitution behavior",
        "parameter_behavior_long,association_scores,model_metrics",
    ),
    "pressure_msl": (
        "weather_numeric",
        "pressure-driven condition differences",
        "parameter_behavior_long,association_scores,model_metrics",
    ),
    "is_day": (
        "weather_time_flag",
        "daytime vs nighttime preference profile",
        "parameter_behavior_long,association_scores,model_metrics",
    ),
    "shortwave_radiation": (
        "weather_numeric",
        "solar radiation and outdoor preference relation",
        "parameter_behavior_long,association_scores,model_metrics",
    ),
    "yagis_kategori": (
        "weather_categorical",
        "interpretable precipitation category effect",
        "parameter_behavior_long,association_scores,model_metrics",
    ),
    "sicaklik_aralik": (
        "weather_categorical",
        "interpretable temperature-band effect",
        "parameter_behavior_long,association_scores,model_metrics",
    ),
    "outlier_flag": (
        "data_quality_flag",
        "normal vs outlier behavior comparison and robustness",
        "data_quality,parameter_behavior_long,model_metrics",
    ),
}


def quantile_bin(series: pd.Series, q: int = 5) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype="object")

    if valid.nunique() < 2:
        out = pd.Series(np.nan, index=series.index, dtype="object")
        out.loc[series.notna()] = "single_bin"
        return out

    try:
        binned = pd.qcut(series, q=q, duplicates="drop")
    except ValueError:
        ranked = series.rank(method="first")
        binned = pd.qcut(ranked, q=q, duplicates="drop")

    out = binned.astype("string").replace("<NA>", np.nan)
    return out.astype("object")


def load_and_prepare_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    datetime_cols = ["acilis_datetime", "kapama_datetime", "tarih", "merge_saati"]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    numeric_cols = [
        "oturum_sure_dk",
        "toplam_miktar",
        "toplam_tutar",
        "urun_sayisi",
        "saat",
        "ay",
        "yil",
        "hafta_no",
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
        "is_day",
        "shortwave_radiation",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "yil" in df.columns:
        df = df[df["yil"] >= 2021].copy()

    if "tarih" in df.columns:
        df = df[df["tarih"].notna()].copy()

    if "outlier_flag" not in df.columns:
        df["outlier_flag"] = "normal"
    df["outlier_flag"] = df["outlier_flag"].fillna("normal").astype(str)

    for col in ["masa_grup", "gun_adi", "yagis_kategori", "sicaklik_aralik", "urun_listesi"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)

    df["hava_hissedilen_fark"] = df["temperature_2m"] - df["apparent_temperature"]
    df["yagis_var"] = (df["precipitation"].fillna(0) > 0).astype(int)

    wind_rad = np.deg2rad(df["winddirection_10m"].fillna(0))
    df["ruzgar_yonu_sin"] = np.sin(wind_rad)
    df["ruzgar_yonu_cos"] = np.cos(wind_rad)

    if "acilis_datetime" in df.columns:
        df["oturum_baslangic_saati"] = df["acilis_datetime"].dt.hour
    else:
        df["oturum_baslangic_saati"] = np.nan

    if "kapama_datetime" in df.columns:
        df["oturum_bitis_saati"] = df["kapama_datetime"].dt.hour
    else:
        df["oturum_bitis_saati"] = np.nan

    if "merge_saati" in df.columns:
        df["merge_hour"] = df["merge_saati"].dt.hour
    else:
        df["merge_hour"] = np.nan

    weekend_values = {"Saturday", "Sunday", "Cumartesi", "Pazar"}
    df["hafta_sonu"] = df["gun_adi"].isin(weekend_values).astype(int)

    weather_cols_for_bins = [
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
    ]
    for col in weather_cols_for_bins:
        if col in df.columns:
            df[f"{col}_qbin"] = quantile_bin(df[col], q=5)

    return df


def build_column_usage(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []

    for col in ORIGINAL_COLUMNS:
        role, usage, output = COLUMN_USAGE_MAP.get(
            col, ("unspecified", "general descriptive use", "data_quality")
        )
        rows.append(
            {
                "column_name": col,
                "role": role,
                "usage_method": usage,
                "output_tables": output,
            }
        )

    derived_cols = [col for col in df.columns if col not in ORIGINAL_COLUMNS]
    for col in derived_cols:
        rows.append(
            {
                "column_name": col,
                "role": "derived_feature",
                "usage_method": "feature engineering for richer pattern detection",
                "output_tables": "parameter_behavior_long,association_scores,model_metrics",
            }
        )

    return pd.DataFrame(rows)


def build_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    n = len(df)

    for col in df.columns:
        s = df[col]
        non_null = int(s.notna().sum())
        null_count = int(s.isna().sum())
        null_rate = (100.0 * null_count / n) if n else 0.0
        unique_count = int(s.nunique(dropna=True))

        if s.dropna().empty:
            sample = ""
        else:
            sample = str(s.dropna().iloc[0])

        rows.append(
            {
                "column_name": col,
                "dtype": str(s.dtype),
                "row_count": n,
                "non_null_count": non_null,
                "null_count": null_count,
                "null_rate_pct": round(null_rate, 4),
                "unique_count": unique_count,
                "sample_value": sample,
            }
        )

    return pd.DataFrame(rows)


def safe_mode(series: pd.Series):
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else np.nan


def build_kpi_summary(df: pd.DataFrame) -> pd.DataFrame:
    start_date = df["tarih"].min().normalize()
    end_date = df["tarih"].max().normalize()

    all_days = pd.date_range(start_date, end_date, freq="D")
    open_days = int(df["tarih"].dt.normalize().nunique())
    closed_days = int(len(all_days) - open_days)

    unique_checks = int(df["CEKNO"].nunique(dropna=True))
    unique_tables = int(df["MASANO"].nunique(dropna=True))

    outlier_rate = float((df["outlier_flag"].astype(str) != "normal").mean() * 100)

    one_row = {
        "analysis_start": str(start_date.date()),
        "analysis_end": str(end_date.date()),
        "row_count": int(len(df)),
        "open_day_count": open_days,
        "closed_day_count": closed_days,
        "unique_check_count": unique_checks,
        "unique_table_count": unique_tables,
        "check_uniqueness_ratio": round(unique_checks / len(df), 6) if len(df) else 0,
        "mean_oturum_sure_dk": round(float(df["oturum_sure_dk"].mean()), 4),
        "median_oturum_sure_dk": round(float(df["oturum_sure_dk"].median()), 4),
        "mean_toplam_tutar": round(float(df["toplam_tutar"].mean()), 4),
        "median_toplam_tutar": round(float(df["toplam_tutar"].median()), 4),
        "mean_toplam_miktar": round(float(df["toplam_miktar"].mean()), 4),
        "mean_urun_sayisi": round(float(df["urun_sayisi"].mean()), 4),
        "outlier_rate_pct": round(outlier_rate, 4),
    }

    return pd.DataFrame([one_row])


def build_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby(df["tarih"].dt.normalize())
        .agg(
            row_count=("CEKNO", "size"),
            unique_check_count=("CEKNO", "nunique"),
            unique_table_count=("MASANO", "nunique"),
            mean_oturum_sure_dk=("oturum_sure_dk", "mean"),
            mean_toplam_tutar=("toplam_tutar", "mean"),
            sum_toplam_tutar=("toplam_tutar", "sum"),
            sum_toplam_miktar=("toplam_miktar", "sum"),
            mean_temperature_2m=("temperature_2m", "mean"),
            mean_precipitation=("precipitation", "mean"),
            mean_windspeed_10m=("windspeed_10m", "mean"),
            dominant_yagis_kategori=("yagis_kategori", safe_mode),
            dominant_sicaklik_aralik=("sicaklik_aralik", safe_mode),
        )
        .reset_index()
        .rename(columns={"tarih": "date"})
    )

    return daily


def build_parameter_behavior(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    parameter_cols = [
        "yagis_kategori",
        "sicaklik_aralik",
        "yagis_var",
        "is_day",
        "gun_adi",
        "ay",
        "hafta_no",
        "saat",
        "merge_hour",
        "hafta_sonu",
        "outlier_flag",
        "temperature_2m_qbin",
        "apparent_temperature_qbin",
        "relative_humidity_2m_qbin",
        "dewpoint_2m_qbin",
        "precipitation_qbin",
        "rain_qbin",
        "showers_qbin",
        "snowfall_qbin",
        "windspeed_10m_qbin",
        "winddirection_10m_qbin",
        "cloudcover_qbin",
        "pressure_msl_qbin",
        "shortwave_radiation_qbin",
    ]

    available_params = [p for p in parameter_cols if p in df.columns]

    long_frames: List[pd.DataFrame] = []
    for param in available_params:
        tmp = df[
            [param, "masa_grup", "oturum_sure_dk", "toplam_tutar", "toplam_miktar", "urun_sayisi", "CEKNO"]
        ].dropna(subset=[param, "masa_grup"])

        if len(tmp) < 100:
            continue

        grouped = (
            tmp.groupby([param, "masa_grup"])
            .agg(
                session_rows=("CEKNO", "size"),
                unique_checks=("CEKNO", "nunique"),
                median_oturum_sure_dk=("oturum_sure_dk", "median"),
                mean_toplam_tutar=("toplam_tutar", "mean"),
                mean_toplam_miktar=("toplam_miktar", "mean"),
                mean_urun_sayisi=("urun_sayisi", "mean"),
            )
            .reset_index()
        )

        totals = grouped.groupby(param)["session_rows"].sum().rename("parameter_total_rows")
        grouped = grouped.merge(totals, on=param, how="left")
        grouped["preference_share_pct"] = (
            100 * grouped["session_rows"] / grouped["parameter_total_rows"]
        ).round(2)

        grouped.insert(0, "parameter", param)
        grouped = grouped.rename(columns={param: "parameter_value", "masa_grup": "table_group"})
        long_frames.append(grouped)

    if long_frames:
        parameter_behavior_long = pd.concat(long_frames, ignore_index=True)
    else:
        parameter_behavior_long = pd.DataFrame(
            columns=[
                "parameter",
                "parameter_value",
                "table_group",
                "session_rows",
                "unique_checks",
                "median_oturum_sure_dk",
                "mean_toplam_tutar",
                "mean_toplam_miktar",
                "mean_urun_sayisi",
                "parameter_total_rows",
                "preference_share_pct",
            ]
        )

    dominant = (
        parameter_behavior_long.sort_values(
            ["parameter", "parameter_value", "preference_share_pct", "session_rows"],
            ascending=[True, True, False, False],
        )
        .groupby(["parameter", "parameter_value"], as_index=False)
        .first()
    )

    if not dominant.empty:
        dominant = dominant[
            [
                "parameter",
                "parameter_value",
                "table_group",
                "preference_share_pct",
                "session_rows",
                "mean_toplam_tutar",
                "median_oturum_sure_dk",
            ]
        ].rename(
            columns={
                "table_group": "dominant_table_group",
                "preference_share_pct": "dominant_share_pct",
            }
        )

    return parameter_behavior_long, dominant, available_params


def build_association_scores(df: pd.DataFrame, params: Iterable[str]) -> pd.DataFrame:
    rows: List[dict] = []

    for param in params:
        tmp = df[[param, "masa_grup"]].dropna()
        if len(tmp) < 100:
            continue

        ct = pd.crosstab(tmp[param], tmp["masa_grup"])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue

        chi2, p_value, _, _ = chi2_contingency(ct)
        n = ct.values.sum()
        min_dim = min(ct.shape) - 1
        cramers_v = np.sqrt((chi2 / n) / min_dim) if min_dim > 0 else np.nan

        rows.append(
            {
                "parameter": param,
                "sample_size": int(n),
                "category_count": int(ct.shape[0]),
                "table_group_count": int(ct.shape[1]),
                "min_group_size": int(ct.sum(axis=1).min()),
                "chi2_stat": float(chi2),
                "p_value": float(p_value),
                "cramers_v": float(cramers_v),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["cramers_v", "p_value"], ascending=[False, True]).reset_index(drop=True)
    return out


def build_kruskal_scores(df: pd.DataFrame, params: Iterable[str]) -> pd.DataFrame:
    outcomes = ["oturum_sure_dk", "toplam_tutar", "toplam_miktar", "urun_sayisi"]
    rows: List[dict] = []

    for param in params:
        for outcome in outcomes:
            if outcome not in df.columns:
                continue

            tmp = df[[param, outcome]].dropna()
            if len(tmp) < 150:
                continue

            groups = [
                g[outcome].values
                for _, g in tmp.groupby(param)
                if len(g) >= 30 and np.isfinite(g[outcome]).all()
            ]
            if len(groups) < 2:
                continue

            stat, p_value = kruskal(*groups)
            rows.append(
                {
                    "parameter": param,
                    "outcome": outcome,
                    "group_count": len(groups),
                    "kruskal_stat": float(stat),
                    "p_value": float(p_value),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["p_value", "kruskal_stat"], ascending=[True, False]).reset_index(drop=True)
    return out


def extract_products(raw_value) -> List[str]:
    if pd.isna(raw_value):
        return []

    text = str(raw_value).strip()
    if text in {"", "[]", "nan", "None"}:
        return []

    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", text)
    if quoted:
        values = []
        for left, right in quoted:
            token = (left or right).strip()
            if token:
                values.append(token)
        return values

    clean = text.strip("[]")
    values = []
    for part in clean.split(","):
        token = part.strip().strip("'\"")
        if token:
            values.append(token)
    return values


def build_product_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tmp = df[["masa_grup", "yagis_kategori", "sicaklik_aralik", "urun_listesi"]].copy()
    tmp["product_tokens"] = tmp["urun_listesi"].apply(extract_products)
    tmp = tmp.explode("product_tokens")
    tmp = tmp.dropna(subset=["product_tokens"])

    if tmp.empty:
        return pd.DataFrame(), pd.DataFrame()

    top_overall = (
        tmp["product_tokens"].value_counts().rename_axis("product_name").reset_index(name="row_count")
    )
    total_rows = top_overall["row_count"].sum()
    top_overall["share_pct"] = (100 * top_overall["row_count"] / total_rows).round(4)
    top_overall = top_overall.head(200)

    top_context = (
        tmp.groupby(["masa_grup", "yagis_kategori", "sicaklik_aralik", "product_tokens"]) 
        .size()
        .reset_index(name="row_count")
        .sort_values("row_count", ascending=False)
        .head(500)
        .rename(columns={"product_tokens": "product_name"})
    )

    return top_overall, top_context


def prepare_model_frame(
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    target: str,
) -> pd.DataFrame:
    use_cols = [c for c in numeric_features + categorical_features + [target] if c in df.columns]
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
    return model_df


def run_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: List[dict] = []
    fi_frames: List[pd.DataFrame] = []

    base_numeric = [
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
    base_categorical = ["gun_adi", "yagis_kategori", "sicaklik_aralik", "outlier_flag"]

    # Classification: masa_grup
    cls_numeric = base_numeric + ["oturum_sure_dk", "toplam_miktar", "toplam_tutar", "urun_sayisi"]
    cls_target = "masa_grup"
    cls_df = prepare_model_frame(df, cls_numeric, base_categorical, cls_target)

    if len(cls_df) >= 1000 and cls_df[cls_target].nunique() >= 2:
        X = cls_df[[c for c in cls_numeric + base_categorical if c in cls_df.columns]]
        y = cls_df[cls_target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        num_cols = [c for c in cls_numeric if c in X.columns]
        cat_cols = [c for c in base_categorical if c in X.columns]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", "passthrough", num_cols),
            ],
            remainder="drop",
        )

        cls_pipe = Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                        n_jobs=-1,
                        min_samples_leaf=5,
                    ),
                ),
            ]
        )

        cls_pipe.fit(X_train, y_train)
        y_pred = cls_pipe.predict(X_test)

        try:
            y_prob = cls_pipe.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        except Exception:
            auc = np.nan

        metric_rows.append(
            {
                "task": "classification",
                "target": cls_target,
                "model": "RandomForestClassifier",
                "sample_size": int(len(cls_df)),
                "metric_accuracy": float(accuracy_score(y_test, y_pred)),
                "metric_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
                "metric_auc_ovr_macro": float(auc) if pd.notna(auc) else np.nan,
                "metric_r2": np.nan,
                "metric_mae": np.nan,
                "metric_rmse": np.nan,
            }
        )

        fi = pd.DataFrame(
            {
                "task": "classification",
                "target": cls_target,
                "model": "RandomForestClassifier",
                "feature_name": cls_pipe.named_steps["prep"].get_feature_names_out(),
                "importance": cls_pipe.named_steps["rf"].feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        fi_frames.append(fi.head(200))

    # Regression helper
    def run_regression(target: str, numeric_features: List[str], categorical_features: List[str]):
        reg_df = prepare_model_frame(df, numeric_features, categorical_features, target)
        if len(reg_df) < 1000:
            return

        feature_cols = [c for c in numeric_features + categorical_features if c in reg_df.columns]
        X_reg = reg_df[feature_cols]
        y_reg = reg_df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X_reg, y_reg, test_size=0.25, random_state=42
        )

        num_cols = [c for c in numeric_features if c in X_reg.columns]
        cat_cols = [c for c in categorical_features if c in X_reg.columns]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", "passthrough", num_cols),
            ],
            remainder="drop",
        )

        reg_pipe = Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=350,
                        random_state=42,
                        n_jobs=-1,
                        min_samples_leaf=5,
                    ),
                ),
            ]
        )

        reg_pipe.fit(X_train, y_train)
        y_pred = reg_pipe.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        metric_rows.append(
            {
                "task": "regression",
                "target": target,
                "model": "RandomForestRegressor",
                "sample_size": int(len(reg_df)),
                "metric_accuracy": np.nan,
                "metric_f1_macro": np.nan,
                "metric_auc_ovr_macro": np.nan,
                "metric_r2": float(r2_score(y_test, y_pred)),
                "metric_mae": float(mean_absolute_error(y_test, y_pred)),
                "metric_rmse": float(rmse),
            }
        )

        fi = pd.DataFrame(
            {
                "task": "regression",
                "target": target,
                "model": "RandomForestRegressor",
                "feature_name": reg_pipe.named_steps["prep"].get_feature_names_out(),
                "importance": reg_pipe.named_steps["rf"].feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        fi_frames.append(fi.head(200))

    reg_cat = base_categorical + ["masa_grup"]

    run_regression(
        target="toplam_tutar",
        numeric_features=base_numeric + ["oturum_sure_dk", "toplam_miktar", "urun_sayisi"],
        categorical_features=reg_cat,
    )

    run_regression(
        target="oturum_sure_dk",
        numeric_features=base_numeric + ["toplam_tutar", "toplam_miktar", "urun_sayisi"],
        categorical_features=reg_cat,
    )

    metrics_df = pd.DataFrame(metric_rows)
    if fi_frames:
        fi_df = pd.concat(fi_frames, ignore_index=True)
    else:
        fi_df = pd.DataFrame(
            columns=["task", "target", "model", "feature_name", "importance"]
        )

    return metrics_df, fi_df


def normalize_for_sql(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        elif pd.api.types.is_categorical_dtype(out[col]):
            out[col] = out[col].astype(str)

    return out


def write_sqlite_tables(tables: Dict[str, pd.DataFrame], sqlite_path: Path) -> None:
    if sqlite_path.exists():
        sqlite_path.unlink()

    with sqlite3.connect(sqlite_path) as conn:
        for table_name, table_df in tables.items():
            normalize_for_sql(table_df).to_sql(table_name, conn, if_exists="replace", index=False)


def to_serializable_records(df: pd.DataFrame, top_n: int = 20) -> List[dict]:
    if df.empty:
        return []
    sample = normalize_for_sql(df.head(top_n)).copy()
    return json.loads(sample.to_json(orient="records", force_ascii=False))


def write_summary_json(
    kpi_summary: pd.DataFrame,
    association_scores: pd.DataFrame,
    model_metrics: pd.DataFrame,
    dominant_preferences: pd.DataFrame,
    output_path: Path,
    sqlite_path: Path,
) -> None:
    payload = {
        "analysis_scope": "Post-2021 IoB restaurant behavior analysis",
        "input_file": str(INPUT_PATH),
        "output_sqlite": str(sqlite_path),
        "kpi_summary": to_serializable_records(kpi_summary, top_n=1),
        "top_association_drivers": to_serializable_records(association_scores, top_n=10),
        "model_metrics": to_serializable_records(model_metrics, top_n=20),
        "dominant_preferences": to_serializable_records(dominant_preferences, top_n=20),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    print("[1/8] Loading and preparing data...")
    df = load_and_prepare_data(INPUT_PATH)

    print("[2/8] Building usage and quality tables...")
    column_usage = build_column_usage(df)
    data_quality = build_data_quality(df)

    print("[3/8] Building KPI and daily summaries...")
    kpi_summary = build_kpi_summary(df)
    daily_summary = build_daily_summary(df)

    print("[4/8] Building parameter-based behavior tables...")
    parameter_behavior_long, dominant_preferences, available_params = build_parameter_behavior(df)

    print("[5/8] Running statistical association tests...")
    association_scores = build_association_scores(df, available_params)
    kruskal_scores = build_kruskal_scores(df, available_params)

    print("[6/8] Building product-context tables...")
    product_top_overall, product_top_context = build_product_tables(df)

    print("[7/8] Training models and extracting metrics...")
    model_metrics, model_feature_importance = run_models(df)

    print("[8/8] Writing consolidated outputs...")
    tables = {
        "column_usage": column_usage,
        "data_quality": data_quality,
        "kpi_summary": kpi_summary,
        "daily_summary": daily_summary,
        "parameter_behavior_long": parameter_behavior_long,
        "dominant_preferences": dominant_preferences,
        "association_scores": association_scores,
        "kruskal_scores": kruskal_scores,
        "model_metrics": model_metrics,
        "model_feature_importance": model_feature_importance,
        "product_top_overall": product_top_overall,
        "product_top_context": product_top_context,
    }
    write_sqlite_tables(tables, SQLITE_PATH)

    write_summary_json(
        kpi_summary=kpi_summary,
        association_scores=association_scores,
        model_metrics=model_metrics,
        dominant_preferences=dominant_preferences,
        output_path=SUMMARY_JSON_PATH,
        sqlite_path=SQLITE_PATH,
    )

    print("Done.")
    print(f"SQLite package: {SQLITE_PATH}")
    print(f"Executive summary: {SUMMARY_JSON_PATH}")


if __name__ == "__main__":
    main()

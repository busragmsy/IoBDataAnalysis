"""
Post-2021 closure analysis
========================================
Question : How many closed days are there after 2021, and what are the
           consecutive closure date ranges?
Output   : closed day list, closure streak ranges, yearly summary, text report
"""

import os
import pandas as pd


# -----------------------------------------
# 0. SETTINGS
# -----------------------------------------
CSV_PATH = r"C:/Users/BUSRA/Documents/GitHub/IoBDataAnalysis/Veriler/oturum_hava_birlesik.csv"
OUTPUT_DIR = r"C:/Users/BUSRA/Documents/GitHub/IoBDataAnalysis/Outputs/Kapali_gun_analizi_cikti"
MIN_YEAR = 2021

os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_closure_streaks(closed_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Group closed days into consecutive streaks."""
    if len(closed_days) == 0:
        return pd.DataFrame(columns=["baslangic_tarihi", "bitis_tarihi", "gun_sayisi"])

    closed_df = pd.DataFrame({"kapali_tarih": closed_days}).sort_values("kapali_tarih")
    closed_df["streak_id"] = closed_df["kapali_tarih"].diff().dt.days.ne(1).cumsum()

    streaks = (
        closed_df.groupby("streak_id")
        .agg(
            baslangic_tarihi=("kapali_tarih", "min"),
            bitis_tarihi=("kapali_tarih", "max"),
            gun_sayisi=("kapali_tarih", "size"),
        )
        .reset_index(drop=True)
        .sort_values(["gun_sayisi", "baslangic_tarihi"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return streaks


def main() -> None:
    print("Veri yukleniyor...")
    df = pd.read_csv(CSV_PATH)

    if "tarih" not in df.columns:
        raise ValueError("CSV dosyasinda 'tarih' kolonu bulunamadi.")

    df["tarih_dt"] = pd.to_datetime(df["tarih"], errors="coerce")
    df = df.dropna(subset=["tarih_dt"]).copy()

    df_post_2021 = df[df["tarih_dt"].dt.year >= MIN_YEAR].copy()
    if df_post_2021.empty:
        raise ValueError(f"{MIN_YEAR} ve sonrasi icin kayit bulunamadi.")

    open_days = pd.DatetimeIndex(df_post_2021["tarih_dt"].dt.normalize().unique()).sort_values()
    period_start = pd.Timestamp(f"{MIN_YEAR}-01-01")
    period_end = open_days.max()
    full_calendar = pd.date_range(period_start, period_end, freq="D")

    closed_days = full_calendar.difference(open_days)
    closed_days_df = pd.DataFrame({"kapali_tarih": closed_days})
    streaks = build_closure_streaks(closed_days)
    streaks_chrono = streaks.sort_values("baslangic_tarihi").reset_index(drop=True)

    yearly_total_days = pd.Series(full_calendar).dt.year.value_counts().sort_index()
    yearly_open_days = pd.Series(open_days).dt.year.value_counts().sort_index()

    yearly_summary = pd.DataFrame(
        {
            "toplam_gun": yearly_total_days,
            "acik_gun": yearly_open_days.reindex(yearly_total_days.index, fill_value=0),
        }
    )
    yearly_summary["kapali_gun"] = yearly_summary["toplam_gun"] - yearly_summary["acik_gun"]
    yearly_summary["kapali_oran_yuzde"] = (
        100 * yearly_summary["kapali_gun"] / yearly_summary["toplam_gun"]
    ).round(2)
    yearly_summary.index.name = "yil"
    yearly_summary = yearly_summary.reset_index()

    total_days = len(full_calendar)
    open_day_count = len(open_days)
    closed_day_count = len(closed_days)
    closed_ratio = (100 * closed_day_count / total_days) if total_days else 0.0

    total_streak_count = len(streaks)
    if total_streak_count > 0:
        longest_streak = streaks.iloc[0]
        avg_streak = streaks["gun_sayisi"].mean()
    else:
        longest_streak = None
        avg_streak = 0.0

    # -----------------------------------------
    # Save outputs
    # -----------------------------------------
    closed_days_out = closed_days_df.copy()
    if not closed_days_out.empty:
        closed_days_out["kapali_tarih"] = closed_days_out["kapali_tarih"].dt.strftime("%Y-%m-%d")

    streaks_out = streaks_chrono.copy()
    if not streaks_out.empty:
        streaks_out["baslangic_tarihi"] = streaks_out["baslangic_tarihi"].dt.strftime("%Y-%m-%d")
        streaks_out["bitis_tarihi"] = streaks_out["bitis_tarihi"].dt.strftime("%Y-%m-%d")

    closed_days_path = os.path.join(OUTPUT_DIR, "kapali_gunler_2021_ve_sonrasi.csv")
    streaks_path = os.path.join(OUTPUT_DIR, "kapali_kalma_donemleri.csv")
    yearly_path = os.path.join(OUTPUT_DIR, "kapali_gun_yillik_ozet.csv")
    summary_path = os.path.join(OUTPUT_DIR, "kapali_gun_analizi_ozet.txt")

    closed_days_out.to_csv(closed_days_path, index=False, encoding="utf-8")
    streaks_out.to_csv(streaks_path, index=False, encoding="utf-8")
    yearly_summary.to_csv(yearly_path, index=False, encoding="utf-8")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("KAPALI GUN ANALIZI (2021 VE SONRASI)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Analiz donemi: {period_start.date()} - {period_end.date()}\n")
        f.write(f"Toplam gun: {total_days}\n")
        f.write(f"Acik gun (veri olan): {open_day_count}\n")
        f.write(f"Kapali gun (veri olmayan): {closed_day_count}\n")
        f.write(f"Kapali gun orani: %{closed_ratio:.2f}\n")
        f.write(f"Kesintisiz kapali donem sayisi: {total_streak_count}\n")
        f.write(f"Ortalama kapali donem uzunlugu: {avg_streak:.2f} gun\n")

        if longest_streak is not None:
            f.write(
                "En uzun kapali donem: "
                f"{int(longest_streak['gun_sayisi'])} gun "
                f"({longest_streak['baslangic_tarihi'].date()} - "
                f"{longest_streak['bitis_tarihi'].date()})\n"
            )
        else:
            f.write("En uzun kapali donem: Bulunamadi\n")

        f.write("\nEn uzun 10 kapali donem:\n")
        if streaks.empty:
            f.write("Kapali donem yok.\n")
        else:
            top10 = streaks.head(10)
            for i, row in top10.iterrows():
                f.write(
                    f"{i + 1:>2}. {int(row['gun_sayisi'])} gun | "
                    f"{row['baslangic_tarihi'].date()} - {row['bitis_tarihi'].date()}\n"
                )

    # -----------------------------------------
    # Console summary
    # -----------------------------------------
    print("\n" + "=" * 60)
    print("KAPALI GUN ANALIZI (2021 VE SONRASI)")
    print("=" * 60)
    print(f"Analiz donemi: {period_start.date()} - {period_end.date()}")
    print(f"Toplam gun: {total_days}")
    print(f"Acik gun: {open_day_count}")
    print(f"Kapali gun: {closed_day_count}")
    print(f"Kapali gun orani: %{closed_ratio:.2f}")
    print(f"Kesintisiz kapali donem sayisi: {total_streak_count}")
    print(f"Ortalama kapali donem uzunlugu: {avg_streak:.2f} gun")

    if longest_streak is not None:
        print(
            "En uzun kapali donem: "
            f"{int(longest_streak['gun_sayisi'])} gun "
            f"({longest_streak['baslangic_tarihi'].date()} - {longest_streak['bitis_tarihi'].date()})"
        )

    print("\nKaydedilen dosyalar:")
    print(f"- {closed_days_path}")
    print(f"- {streaks_path}")
    print(f"- {yearly_path}")
    print(f"- {summary_path}")
    print("\nAnaliz tamamlandi.")


if __name__ == "__main__":
    main()

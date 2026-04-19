import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# PATHS
# =========================================================
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT_TABLES = ROOT / "outputs" / "tables"
OUTPUT_FIGURES = ROOT / "outputs" / "figures"


# =========================================================
# LOAD DATA
# =========================================================
def load_master_dataset():

    df = pd.read_csv(PROCESSED / "master_dataset.csv", parse_dates=["date"])

    # Tarihe göre sırala
    df = df.sort_values("date").reset_index(drop=True)

    return df


# =========================================================
# BASIC METRICS
# =========================================================
def negative_return_frequency(series):
    
    return (series < 0).mean()


def downside_std(series):
    
    negative_part = series[series < 0]

    if len(negative_part) == 0:
        return np.nan

    return negative_part.std()


def cumulative_return(series):
    
    return (1 + series).cumprod() - 1


# =========================================================
# ASSET SUMMARY
# =========================================================
def summarize_asset(series, asset_name):
    
    summary = {
        f"{asset_name}_count": series.count(),
        f"{asset_name}_mean": series.mean(),
        f"{asset_name}_std": series.std(),
        f"{asset_name}_min": series.min(),
        f"{asset_name}_p5": series.quantile(0.05),
        f"{asset_name}_median": series.median(),
        f"{asset_name}_p95": series.quantile(0.95),
        f"{asset_name}_max": series.max(),
        f"{asset_name}_neg_freq": negative_return_frequency(series),
        f"{asset_name}_downside_std": downside_std(series),
    }
    return summary


# =========================================================
# SAMPLE / REGIME SUMMARY
# =========================================================
def summarize_regime(df, regime_name):
    
    row = {"regime": regime_name}

    # Rejim tarih aralığı
    row["start_date"] = df["date"].min()
    row["end_date"] = df["date"].max()

    # BIST summary
    row.update(summarize_asset(df["bist_ret"], "bist"))

    # GOLD summary
    row.update(summarize_asset(df["gold_ret"], "gold"))

    # Temel korelasyonlar
    row["corr_bist_gold"] = df["bist_ret"].corr(df["gold_ret"])
    row["corr_bist_usdtry"] = df["bist_ret"].corr(df["usdtry_ret"])
    row["corr_gold_usdtry"] = df["gold_ret"].corr(df["usdtry_ret"])

    return pd.DataFrame([row])


# =========================================================
# REGIME SPLITS
# =========================================================
def split_regimes(df):
    
    regimes = {
        "full_sample": df.copy(),

        
        "high_inflation": df[df["D_high_inflation"] == 1].copy(),

        # FX stress: Ağustos 2018 - Eylül 2018
        "fx_stress": df[df["D_fx_stress"] == 1].copy(),

        # Tight policy: Haziran 2023 sonrası tanımlanan policy regime
        "tight_policy": df[df["D_tight_policy"] == 1].copy(),
    }

    return regimes


# =========================================================
# PERCENTAGE TABLE
# =========================================================
def create_percentage_table(summary_table):
    
    pct_table = summary_table.copy()

    pct_cols = [
        "bist_mean", "bist_std", "bist_p5", "bist_p95",
        "gold_mean", "gold_std", "gold_p5", "gold_p95"
    ]

    for col in pct_cols:
        if col in pct_table.columns:
            pct_table[col] = pct_table[col] * 100

    rename_map = {
        "bist_mean": "bist_mean_pct",
        "bist_std": "bist_std_pct",
        "bist_p5": "bist_p5_pct",
        "bist_p95": "bist_p95_pct",
        "gold_mean": "gold_mean_pct",
        "gold_std": "gold_std_pct",
        "gold_p5": "gold_p5_pct",
        "gold_p95": "gold_p95_pct",
    }

    pct_table = pct_table.rename(columns=rename_map)

    output_path = OUTPUT_TABLES / "descriptive_summary_pct.csv"
    pct_table.to_csv(output_path, index=False)

    print("\nSaved: descriptive_summary_pct.csv")

    return pct_table


# =========================================================
# BUILD SUMMARY TABLE
# =========================================================
def build_descriptive_summary(df):
    regimes = split_regimes(df)

    summary_rows = []

    for regime_name, regime_df in regimes.items():
        if len(regime_df) == 0:
            continue

        summary_rows.append(summarize_regime(regime_df, regime_name))

    summary_table = pd.concat(summary_rows, ignore_index=True)

    
    output_path = OUTPUT_TABLES / "descriptive_summary.csv"
    summary_table.to_csv(output_path, index=False)

    print("\nSaved: descriptive_summary.csv")
    print("\n=== DESCRIPTIVE SUMMARY ===")
    print(summary_table)

    
    create_percentage_table(summary_table)

    return summary_table


# =========================================================
# REGIME COUNTS
# =========================================================
def build_regime_counts(df):
    counts = pd.DataFrame({
        "regime": [
            "full_sample",
            "high_inflation",
            "fx_stress",
            "tight_policy"
        ],
        "n_obs": [
            len(df),
            int(df["D_high_inflation"].sum()),
            int(df["D_fx_stress"].sum()),
            int(df["D_tight_policy"].sum())
        ]
    })

    output_path = OUTPUT_TABLES / "regime_counts.csv"
    counts.to_csv(output_path, index=False)

    print("\nSaved: regime_counts.csv")
    print("\n=== REGIME COUNTS ===")
    print(counts)

    return counts


# =========================================================
# CUMULATIVE RETURN PLOTS
# =========================================================
def plot_cumulative_returns(df, title, file_name):
    plot_df = df.copy().sort_values("date")

    plot_df["bist_cumret"] = cumulative_return(plot_df["bist_ret"])
    plot_df["gold_cumret"] = cumulative_return(plot_df["gold_ret"])

    plt.figure(figsize=(10, 5))
    plt.plot(plot_df["date"], plot_df["bist_cumret"], label="BIST")
    plt.plot(plot_df["date"], plot_df["gold_cumret"], label="Gold (TRY)")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_FIGURES / file_name
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {file_name}")


def make_regime_cumulative_plots(df):
    regimes = split_regimes(df)

    
    if len(regimes["full_sample"]) > 1:
        plot_cumulative_returns(
            regimes["full_sample"],
            "Cumulative Returns - Full Sample",
            "cumret_full_sample.png"
        )

    if len(regimes["high_inflation"]) > 1:
        plot_cumulative_returns(
            regimes["high_inflation"],
            "Cumulative Returns - High Inflation",
            "cumret_high_inflation.png"
        )

    if len(regimes["fx_stress"]) > 1:
        plot_cumulative_returns(
            regimes["fx_stress"],
            "Cumulative Returns - FX Stress",
            "cumret_fx_stress.png"
        )

    if len(regimes["tight_policy"]) > 1:
        plot_cumulative_returns(
            regimes["tight_policy"],
            "Cumulative Returns - Tight Policy",
            "cumret_tight_policy.png"
        )


# =========================================================
# ROLLING CORRELATION
# =========================================================
def plot_rolling_correlation(df, window, file_name):
    plot_df = df.copy().sort_values("date")

    plot_df["rolling_corr_bist_gold"] = (
        plot_df["bist_ret"].rolling(window=window).corr(plot_df["gold_ret"])
    )

    plt.figure(figsize=(10, 5))
    plt.plot(plot_df["date"], plot_df["rolling_corr_bist_gold"])
    plt.axhline(0, linestyle="--")

    plt.title(f"Rolling Correlation: BIST vs Gold ({window}-day window)")
    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.tight_layout()

    output_path = OUTPUT_FIGURES / file_name
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {file_name}")


def make_rolling_correlation_plots(df):
    
    plot_rolling_correlation(df, 30, "rolling_corr_bist_gold_30d.png")
    plot_rolling_correlation(df, 60, "rolling_corr_bist_gold_60d.png")


# =========================================================
# MAIN RUNNER
# =========================================================
def run_descriptive_analysis():
    
    df = load_master_dataset()

    
    build_descriptive_summary(df)

    
    build_regime_counts(df)

   
    make_regime_cumulative_plots(df)

    
    make_rolling_correlation_plots(df)

    print("\nDescriptive analysis completed.")


if __name__ == "__main__":
    run_descriptive_analysis()
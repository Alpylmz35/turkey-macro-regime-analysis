import pandas as pd
import statsmodels.api as sm
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

    
    df = df.sort_values("date").reset_index(drop=True)

    return df


# =========================================================
# MODEL HELPERS
# =========================================================
def fit_ols_hac(y, X, maxlags=5):
    
    X = sm.add_constant(X)

    # OLS + Newey-West HAC
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

    return model


def extract_model_results(model, model_name, asset_name, sample_name):
    
    result = pd.DataFrame({
        "variable": model.params.index,
        "coef": model.params.values,
        "std_err": model.bse.values,
        "t_value": model.tvalues.values,
        "p_value": model.pvalues.values,
    })

    result["model"] = model_name
    result["asset"] = asset_name
    result["sample"] = sample_name
    result["n_obs"] = int(model.nobs)
    result["r_squared"] = model.rsquared

    return result


def significance_stars(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    else:
        return ""


# =========================================================
# SAVE CLEAN OUTPUTS
# =========================================================
def save_model_summary(model, model_name, asset_name):
    
    txt_path = OUTPUT_TABLES / f"summary_{model_name}_{asset_name}.txt"
    html_path = OUTPUT_TABLES / f"summary_{model_name}_{asset_name}.html"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(str(model.summary()))

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(model.summary().as_html())

    print(f"Saved: {txt_path.name}")
    print(f"Saved: {html_path.name}")


def save_simple_table(df, base_name):
    csv_path = OUTPUT_TABLES / f"{base_name}.csv"
    md_path = OUTPUT_TABLES / f"{base_name}.md"
    html_path = OUTPUT_TABLES / f"{base_name}.html"

    df.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))

    df.to_html(html_path, index=False)

    print(f"Saved: {csv_path.name}")
    print(f"Saved: {md_path.name}")
    print(f"Saved: {html_path.name}")


# =========================================================
# FIGURES
# =========================================================
def plot_coefficients(model, model_name, asset_name):
    coef_df = pd.DataFrame({
        "variable": model.params.index,
        "coef": model.params.values
    })

    coef_df = coef_df[coef_df["variable"] != "const"].copy()

    plt.figure(figsize=(9, 5))
    plt.bar(coef_df["variable"], coef_df["coef"])
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Coefficient Plot - {model_name} - {asset_name.upper()}")
    plt.ylabel("Coefficient")
    plt.tight_layout()

    output_path = OUTPUT_FIGURES / f"coef_{model_name}_{asset_name}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path.name}")


def plot_actual_vs_fitted(model, y, model_name, asset_name):
    fitted = model.fittedvalues

    plt.figure(figsize=(6, 6))
    plt.scatter(fitted, y, alpha=0.4)
    plt.xlabel("Fitted")
    plt.ylabel("Actual")
    plt.title(f"Actual vs Fitted - {model_name} - {asset_name.upper()}")
    plt.tight_layout()

    output_path = OUTPUT_FIGURES / f"actual_vs_fitted_{model_name}_{asset_name}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path.name}")


def plot_residual_histogram(model, model_name, asset_name):
    resid = model.resid

    plt.figure(figsize=(8, 5))
    plt.hist(resid, bins=40)
    plt.title(f"Residual Histogram - {model_name} - {asset_name.upper()}")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()

    output_path = OUTPUT_FIGURES / f"residual_hist_{model_name}_{asset_name}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path.name}")


def plot_interaction_effect(results_df, regime_name):
    interaction_var = f"usdtry_x_{regime_name}"

    sub = results_df[
        results_df["variable"].isin(["usdtry_ret", interaction_var])
    ].copy()

    if sub.empty:
        return

    for asset_name in sub["asset"].unique():
        asset_sub = sub[sub["asset"] == asset_name].copy()

        if asset_sub.empty:
            continue

        plt.figure(figsize=(6, 4))
        plt.bar(asset_sub["variable"], asset_sub["coef"])
        plt.title(f"USDTRY Effect Components - {regime_name} - {asset_name.upper()}")
        plt.ylabel("Coefficient")
        plt.tight_layout()

        output_path = OUTPUT_FIGURES / f"interaction_effect_{regime_name}_{asset_name}.png"
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Saved: {output_path.name}")


# =========================================================
# PUBLICATION TABLE
# =========================================================
def build_publication_table(results_df):
    variables_order = [
        "const",
        "usdtry_ret",
        "usdtry_x_high_inflation",
        "usdtry_x_fx_stress",
        "usdtry_x_tight_policy",
        "cds5y_change",
        "bond2y_change_bps",
        "fx_vol_5d",
        "bist_ret_l1",
        "gold_ret_l1"
    ]

    df = results_df.copy()
    df = df[df["variable"].isin(variables_order)]

    df["stars"] = df["p_value"].apply(significance_stars)

    df["formatted"] = (
        df["coef"].round(3).astype(str)
        + df["stars"]
        + "\n(" + df["std_err"].round(3).astype(str) + ")"
    )

    table = df.pivot_table(
        index="variable",
        columns=["model", "asset"],
        values="formatted",
        aggfunc="first"
    )

    table = table.reindex(variables_order)

    
    model_info = (
        df.groupby(["model", "asset"])
        .agg(
            n_obs=("n_obs", "first"),
            r2=("r_squared", "first")
        )
    )

    n_row = model_info["n_obs"].astype(int)
    n_row.name = "N"

    r2_row = model_info["r2"].round(3)
    r2_row.name = "R_squared"

    table = pd.concat([table, n_row.to_frame().T, r2_row.to_frame().T])

    table = table.reset_index()

    return table


def build_model_summary(results_df):
    summary = (
        results_df.groupby(["model", "asset", "sample"], as_index=False)
        .agg(
            n_obs=("n_obs", "first"),
            r_squared=("r_squared", "first")
        )
    )

    return summary


# =========================================================
# BASELINE MODEL
# =========================================================
def run_baseline_model(df, asset_name):
    if asset_name == "bist":
        y = df["bist_ret"]
        lag_var = "bist_ret_l1"
    elif asset_name == "gold":
        y = df["gold_ret"]
        lag_var = "gold_ret_l1"
    else:
        raise ValueError("asset_name must be 'bist' or 'gold'")

    X = df[
        [
            "usdtry_ret",
            "fx_vol_5d",
            "bond2y_change_bps",
            "cds5y_change",
            lag_var,
        ]
    ]

    model = fit_ols_hac(y, X, maxlags=5)

    print("\n" + "=" * 80)
    print(f"BASELINE MODEL - {asset_name.upper()}")
    print("=" * 80)
    print(model.summary())

    save_model_summary(model, "baseline", asset_name)
    plot_coefficients(model, "baseline", asset_name)
    plot_actual_vs_fitted(model, y, "baseline", asset_name)
    plot_residual_histogram(model, "baseline", asset_name)

    results = extract_model_results(
        model=model,
        model_name="baseline",
        asset_name=asset_name,
        sample_name="full_sample"
    )

    return model, results


# =========================================================
# INTERACTION MODELS
# =========================================================
def run_interaction_model(df, asset_name, regime_name):
    data = df.copy()

    if asset_name == "bist":
        y = data["bist_ret"]
        lag_var = "bist_ret_l1"
    elif asset_name == "gold":
        y = data["gold_ret"]
        lag_var = "gold_ret_l1"
    else:
        raise ValueError("asset_name must be 'bist' or 'gold'")

    regime_map = {
        "high_inflation": "D_high_inflation",
        "fx_stress": "D_fx_stress",
        "tight_policy": "D_tight_policy"
    }

    regime_var = regime_map[regime_name]

   
    interaction_var = f"usdtry_x_{regime_name}"
    data[interaction_var] = data["usdtry_ret"] * data[regime_var]

    X = data[
        [
            "usdtry_ret",
            "fx_vol_5d",
            "bond2y_change_bps",
            "cds5y_change",
            lag_var,
            regime_var,
            interaction_var
        ]
    ]

    model = fit_ols_hac(y, X, maxlags=5)

    print("\n" + "=" * 80)
    print(f"INTERACTION MODEL - {asset_name.upper()} - {regime_name.upper()}")
    print("=" * 80)
    print(model.summary())

    save_model_summary(model, f"interaction_{regime_name}", asset_name)
    plot_coefficients(model, f"interaction_{regime_name}", asset_name)
    plot_actual_vs_fitted(model, y, f"interaction_{regime_name}", asset_name)
    plot_residual_histogram(model, f"interaction_{regime_name}", asset_name)

    results = extract_model_results(
        model=model,
        model_name=f"interaction_{regime_name}",
        asset_name=asset_name,
        sample_name="full_sample"
    )

    return model, results


# =========================================================
# SUBSAMPLE MODELS
# =========================================================
def run_subsample_model(df, asset_name, regime_name):
    regime_map = {
        "high_inflation": "D_high_inflation",
        "fx_stress": "D_fx_stress",
        "tight_policy": "D_tight_policy"
    }

    sub = df[df[regime_map[regime_name]] == 1].copy()

    if asset_name == "bist":
        y = sub["bist_ret"]
        lag_var = "bist_ret_l1"
    elif asset_name == "gold":
        y = sub["gold_ret"]
        lag_var = "gold_ret_l1"
    else:
        raise ValueError("asset_name must be 'bist' or 'gold'")

    X = sub[
        [
            "usdtry_ret",
            "fx_vol_5d",
            "bond2y_change_bps",
            "cds5y_change",
            lag_var,
        ]
    ]

    model = fit_ols_hac(y, X, maxlags=5)

    print("\n" + "=" * 80)
    print(f"SUBSAMPLE MODEL - {asset_name.upper()} - {regime_name.upper()}")
    print("=" * 80)
    print(model.summary())

    save_model_summary(model, f"subsample_{regime_name}", asset_name)
    plot_coefficients(model, f"subsample_{regime_name}", asset_name)
    plot_actual_vs_fitted(model, y, f"subsample_{regime_name}", asset_name)
    plot_residual_histogram(model, f"subsample_{regime_name}", asset_name)

    results = extract_model_results(
        model=model,
        model_name=f"subsample_{regime_name}",
        asset_name=asset_name,
        sample_name=regime_name
    )

    return model, results


# =========================================================
# RUN ALL MODELS
# =========================================================
def run_all_models():
    df = load_master_dataset()

    all_results = []

    
    for asset_name in ["bist", "gold"]:
        _, results = run_baseline_model(df, asset_name)
        all_results.append(results)

    
    for regime_name in ["high_inflation", "fx_stress", "tight_policy"]:
        for asset_name in ["bist", "gold"]:
            _, results = run_interaction_model(df, asset_name, regime_name)
            all_results.append(results)

    # Subsample models
    for regime_name in ["high_inflation", "fx_stress", "tight_policy"]:
        for asset_name in ["bist", "gold"]:
            _, results = run_subsample_model(df, asset_name, regime_name)
            all_results.append(results)

    
    results_df = pd.concat(all_results, ignore_index=True)

    publication_table = build_publication_table(results_df)
    save_simple_table(publication_table, "regression_publication_table")

    model_summary = build_model_summary(results_df)
    save_simple_table(model_summary, "regression_model_summary")

    #  Interaction figures
    for regime_name in ["high_inflation", "fx_stress", "tight_policy"]:
        plot_interaction_effect(results_df, regime_name)

    print("\nRegression analysis completed.")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    run_all_models()
"""
Build a monthly sector ETF panel dataset for sector outperformance prediction.

This script prepares the data used in Jackson Wang's Gradient Boosting model.
It downloads ETF and SPY prices from Yahoo Finance, joins local FRED macro CSVs,
creates rolling momentum/volatility/drawdown features, constructs the target,
and saves chronological train/validation/test splits.

Expected local files:
    data/DGS10.csv
    data/DGS2.csv
    data/DFF.csv
    data/VIXCLS.csv
"""

from pathlib import Path

import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

START_DATE = "2015-01-01"
END_DATE = "2026-01-01"

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE"]
BENCHMARK = "SPY"

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_fred_csv(file_path: Path, series_name: str) -> pd.DataFrame:
    """Load a FRED CSV file and standardize it to Date / series_name columns."""
    df = pd.read_csv(file_path)

    if "DATE" in df.columns:
        date_col = "DATE"
    elif "observation_date" in df.columns:
        date_col = "observation_date"
    else:
        date_col = df.columns[0]

    value_col = [col for col in df.columns if col != date_col][0]

    df = df[[date_col, value_col]].copy()
    df.columns = ["Date", series_name]
    df["Date"] = pd.to_datetime(df["Date"])
    df[series_name] = pd.to_numeric(df[series_name], errors="coerce")
    return df


def build_panel() -> tuple[pd.DataFrame, list[str]]:
    """Build the unified sector ETF panel and return the cleaned panel plus feature names."""
    tickers = SECTOR_ETFS + [BENCHMARK]

    raw_prices = yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
    )

    if isinstance(raw_prices.columns, pd.MultiIndex):
        close_daily = raw_prices["Adj Close"].copy()
    else:
        close_daily = raw_prices[["Adj Close"]].copy()
        close_daily.columns = tickers

    monthly_close = close_daily.resample("M").last()
    monthly_ret = monthly_close.pct_change()

    dgs10 = load_fred_csv(DATA_DIR / "DGS10.csv", "DGS10")
    dgs2 = load_fred_csv(DATA_DIR / "DGS2.csv", "DGS2")
    dff = load_fred_csv(DATA_DIR / "DFF.csv", "DFF")
    vix = load_fred_csv(DATA_DIR / "VIXCLS.csv", "VIXCLS")

    macro = dgs10.merge(dgs2, on="Date", how="outer")
    macro = macro.merge(dff, on="Date", how="outer")
    macro = macro.merge(vix, on="Date", how="outer")
    macro = macro.sort_values("Date").ffill()

    macro_monthly = macro.set_index("Date").resample("M").last()
    macro_monthly["yield_spread_10y_2y"] = macro_monthly["DGS10"] - macro_monthly["DGS2"]

    rows = []
    spy_ret = monthly_ret[BENCHMARK]
    spy_price = monthly_close[BENCHMARK]

    for etf in SECTOR_ETFS:
        etf_ret = monthly_ret[etf]
        etf_price = monthly_close[etf]

        temp = pd.DataFrame(
            {
                "Date": monthly_ret.index,
                "ETF": etf,
                "etf_ret": etf_ret.values,
                "spy_ret": spy_ret.values,
                "next_etf_ret": etf_ret.shift(-1).values,
                "next_spy_ret": spy_ret.shift(-1).values,
                "etf_mom_1m": etf_ret.values,
                "etf_mom_3m": etf_price.pct_change(3).values,
                "etf_mom_6m": etf_price.pct_change(6).values,
                "etf_mom_12m": etf_price.pct_change(12).values,
                "etf_vol_3m": etf_ret.rolling(3).std().values,
                "etf_vol_6m": etf_ret.rolling(6).std().values,
                "etf_vol_12m": etf_ret.rolling(12).std().values,
                "etf_drawdown_12m": (etf_price / etf_price.rolling(12).max() - 1).values,
                "spy_mom_1m": spy_ret.values,
                "spy_mom_3m": spy_price.pct_change(3).values,
                "spy_mom_6m": spy_price.pct_change(6).values,
                "spy_mom_12m": spy_price.pct_change(12).values,
                "spy_vol_3m": spy_ret.rolling(3).std().values,
                "spy_vol_6m": spy_ret.rolling(6).std().values,
                "spy_vol_12m": spy_ret.rolling(12).std().values,
                "spy_drawdown_12m": (spy_price / spy_price.rolling(12).max() - 1).values,
            }
        )

        temp["target"] = (temp["next_etf_ret"] > temp["next_spy_ret"]).astype(int)
        rows.append(temp)

    panel = pd.concat(rows, ignore_index=True)

    macro_monthly_reset = macro_monthly.reset_index()
    macro_monthly_reset["YearMonth"] = macro_monthly_reset["Date"].dt.to_period("M")
    panel["YearMonth"] = panel["Date"].dt.to_period("M")

    panel = panel.merge(
        macro_monthly_reset.drop(columns=["Date"]),
        on="YearMonth",
        how="left",
    ).drop(columns=["YearMonth"])

    panel = panel.sort_values(["ETF", "Date"]).reset_index(drop=True)

    macro_cols = ["DGS10", "DGS2", "DFF", "VIXCLS", "yield_spread_10y_2y"]
    for col in macro_cols:
        panel[f"{col}_change_1m"] = panel.groupby("ETF")[col].diff()

    panel = pd.get_dummies(panel, columns=["ETF"], drop_first=True)
    panel_clean = panel.dropna().reset_index(drop=True)

    drop_cols = ["Date", "target", "next_etf_ret", "next_spy_ret"]
    feature_cols = [col for col in panel_clean.columns if col not in drop_cols]

    return panel_clean, feature_cols


def save_splits(panel_clean: pd.DataFrame, feature_cols: list[str]) -> None:
    """Save clean panel data and scaled chronological model splits."""
    train_mask = (panel_clean["Date"] >= "2015-01-01") & (panel_clean["Date"] <= "2020-12-31")
    valid_mask = (panel_clean["Date"] >= "2021-01-01") & (panel_clean["Date"] <= "2022-12-31")
    test_mask = (panel_clean["Date"] >= "2023-01-01") & (panel_clean["Date"] <= "2025-12-31")

    X_train = panel_clean.loc[train_mask, feature_cols].copy()
    X_valid = panel_clean.loc[valid_mask, feature_cols].copy()
    X_test = panel_clean.loc[test_mask, feature_cols].copy()

    y_train = panel_clean.loc[train_mask, "target"].astype(int).copy()
    y_valid = panel_clean.loc[valid_mask, "target"].astype(int).copy()
    y_test = panel_clean.loc[test_mask, "target"].astype(int).copy()

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_valid_scaled = pd.DataFrame(scaler.transform(X_valid), columns=feature_cols, index=X_valid.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    panel_clean.to_csv(OUTPUT_DIR / "sector_etf_unified_panel_2015_2025.csv", index=False)
    X_train_scaled.to_csv(OUTPUT_DIR / "X_train_scaled.csv", index=False)
    X_valid_scaled.to_csv(OUTPUT_DIR / "X_valid_scaled.csv", index=False)
    X_test_scaled.to_csv(OUTPUT_DIR / "X_test_scaled.csv", index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_valid.to_csv(OUTPUT_DIR / "y_valid.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    print("Saved cleaned panel and model splits.")
    print("Panel shape:", panel_clean.shape)
    print("Date range:", panel_clean["Date"].min(), "to", panel_clean["Date"].max())
    print("Train / validation / test target means:", y_train.mean(), y_valid.mean(), y_test.mean())


if __name__ == "__main__":
    panel, features = build_panel()
    save_splits(panel, features)

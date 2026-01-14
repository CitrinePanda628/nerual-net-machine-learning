from defeatbeta_api.data.ticker import Ticker
import pandas_ta as ta
import pandas as pd

STOCKS = {
    "AAPL": 1,
    "MSFT": 2,
    "GOOGL": 3,
    "AMZN": 4,
    "META": 5,
    "NVDA": 6,
    "TSLA": 7
}

all_data = []

for symbol, symbol_id in STOCKS.items():
    print(f"Fetching {symbol}...")

    ticker = Ticker(symbol)
    df = ticker.price().sort_values("report_date")

    df["symbol"] = symbol
    df["symbol_id"] = symbol_id

    # --- RETURNS ---
    df["return"] = df["close"].pct_change()
    df["return_3d"]  = df["close"].pct_change(3)
    df["return_5d"]  = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)

    # --- MOMENTUM ---
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"])
    df["macd_hist"] = macd["MACDh_12_26_9"]

    # --- TREND ---
    df["sma_20"] = ta.sma(df["close"], length=20)
    df["price_vs_sma20"] = (df["close"] - df["sma_20"]) / df["sma_20"]

    # --- VOLATILITY ---
    df["volatility_5d"] = df["return"].rolling(5, min_periods=5).std()
    df["volatility_10d"] = df["return"].rolling(10, min_periods=10).std()
    df["volatility_ratio"] = df["volatility_5d"] / df["volatility_10d"]

    bb = ta.bbands(df["close"], length=20)
    upper = bb.filter(like="BBU").iloc[:, 0]
    lower = bb.filter(like="BBL").iloc[:, 0]
    df["bb_width"] = (upper - lower) / df["close"]

    # --- VOLUME ---
    df["volume_sma_5"] = ta.sma(df["volume"], length=5)
    df["volume_ratio"] = df["volume"] / df["volume_sma_5"]

    # --- TREND STRENGTH ---
    adx = ta.adx(df["high"], df["low"], df["close"])
    df["adx"] = adx["ADX_14"]

    all_data.append(df)

# =========================
# FINAL DATASET
# =========================
data = pd.concat(all_data, ignore_index=True)

# --- TARGET ---
data["target_return"] = data.groupby("symbol_id")["return"].shift(-1)

features = [
    "return",
    "return_3d",
    "return_5d",
    "return_10d",
    "rsi_14",
    "macd_hist",
    "price_vs_sma20",
    "volatility_5d",
    "volatility_ratio",
    "bb_width",
    "volume_ratio",
    "adx",
    "symbol_id"
]

final_df = data[["report_date", "symbol"] + features + ["target_return"]]

# Drop rows where any feature or target is NaN
final_df = final_df.dropna(subset=features + ["target_return"])

print("Dataset created successfully")
print(final_df.head())
print("Rows:", len(final_df))

final_df.to_csv(
    "/mnt/c/Users/fayzm/Downloads/magnificent7_ml_dataset_v2.csv",
    index=False
)
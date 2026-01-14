import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Optional: disable oneDNN optimizations at the OS level, not here:
#   set TF_ENABLE_ONEDNN_OPTS=0  (in your shell)

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "btc_hourly_ohclv_ta.csv"

HISTORY_HOURS = 72      # how many past hours to look at
AHEAD_HOURS = 1         # predict 1h ahead
OFFSET_STEPS = AHEAD_HOURS  # since data is hourly

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv(CSV_PATH)

# Clean column names
df.columns = df.columns.str.strip().str.upper()

# We expect these columns from your CSV:
# UNIX_TIMESTAMP, DATETIME, OPEN, HIGH, CLOSE, LOW, VOLUME, SMA_20, EMA_12, EMA_26,
# MACD, MACD_SIGNAL, MACD_DIFF, RSI, BB_HIGH, BB_LOW, BB_WIDTH, STOCH_K, STOCH_D,
# VOLUME_SMA, MFI, ATR, PRICE_CHANGE, HIGH_LOW_RATIO, CLOSE_OPEN_RATIO,
# VOLATILITY_30D, PRICE_VOLATILITY_30D, HL_VOLATILITY_30D

# Drop DATETIME if present (we'll use UNIX_TIMESTAMP)
if "DATETIME" in df.columns:
    df = df.drop(columns=["DATETIME"])

# Sort and create datetime index
df = df.sort_values("UNIX_TIMESTAMP")
df["dt"] = pd.to_datetime(df["UNIX_TIMESTAMP"], unit="s")
df = df.set_index("dt").sort_index()

# ============================================================
# 2. ADD TIME + RETURN/VOL FEATURES
# ============================================================
df["hour"] = df.index.hour
df["day"] = df.index.day
df["month"] = df.index.month
df["year"] = df.index.year
df["dayofweek"] = df.index.dayofweek
df["dayofyear"] = df.index.dayofyear
df["weekofyear"] = df.index.isocalendar().week.astype(int)
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Short-term returns and volatility (more signal, less laggy than indicators)
df["RET_1H"] = df["CLOSE"].pct_change()
df["RET_6H"] = df["CLOSE"].pct_change(6)
df["RET_24H"] = df["CLOSE"].pct_change(24)

df["VOL_6H"] = df["RET_1H"].rolling(6).std()
df["VOL_24H"] = df["RET_1H"].rolling(24).std()

# Target: 1-hour-ahead return
df["RET_1H_FWD"] = df["CLOSE"].shift(-AHEAD_HOURS) / df["CLOSE"] - 1.0

# ============================================================
# 3. FEATURE SELECTION
# ============================================================
indicator_cols = [
    "OPEN","HIGH","CLOSE","LOW","VOLUME",
    "SMA_20","EMA_12","EMA_26",
    "MACD","MACD_SIGNAL","MACD_DIFF",
    "RSI",
    "BB_HIGH","BB_LOW","BB_WIDTH",
    "STOCH_K","STOCH_D",
    "VOLUME_SMA","MFI","ATR",
    "PRICE_CHANGE","HIGH_LOW_RATIO","CLOSE_OPEN_RATIO",
    "VOLATILITY_30D","PRICE_VOLATILITY_30D","HL_VOLATILITY_30D",
]

time_cols = [
    "hour","day","month","year","dayofweek","dayofyear","weekofyear",
    "is_weekend",
    "hour_sin","hour_cos",
    "dow_sin","dow_cos",
    "month_sin","month_cos"
]

return_vol_cols = [
    "RET_1H","RET_6H","RET_24H",
    "VOL_6H","VOL_24H"
]

feature_cols = indicator_cols + time_cols + return_vol_cols
target_col = "RET_1H_FWD"

# Drop rows where target is NaN (because of shift) or CLOSE missing
df = df.dropna(subset=[target_col, "CLOSE"])

# Also fill any residual NaNs in features
df[feature_cols] = df[feature_cols].fillna(method="bfill").fillna(method="ffill")

# ============================================================
# 4. TRAIN / VAL / TEST SPLIT (70 / 20 / 10)
# ============================================================
N = len(df)
train_end = int(N * 0.70)
val_end = int(N * 0.90)

df_train = df.iloc[:train_end]
df_val = df.iloc[train_end:val_end]
df_test = df.iloc[val_end:]

# ============================================================
# 5. SCALING
# ============================================================
scaler_X = StandardScaler()
scaler_y = StandardScaler()  # we’re predicting returns, so standard scaling is fine

scaler_X.fit(df_train[feature_cols])
scaler_y.fit(df_train[[target_col]])

def scale_part(df_part):
    X_scaled = pd.DataFrame(
        scaler_X.transform(df_part[feature_cols]),
        index=df_part.index,
        columns=feature_cols
    )
    y_scaled = scaler_y.transform(df_part[[target_col]]).flatten()
    return X_scaled, y_scaled

X_train_df, y_train = scale_part(df_train)
X_val_df, y_val = scale_part(df_val)
X_test_df, y_test = scale_part(df_test)

# ============================================================
# 6. SEQUENCE CREATION
# ============================================================
def create_sequences(X_df, y_arr, history, offset):
    X, y, idxs = [], [], []
    data = X_df.values
    index = X_df.index

    for i in range(len(X_df) - history - offset):
        X.append(data[i:i+history])
        y.append(y_arr[i+history+offset-1])
        idxs.append(index[i+history+offset-1])

    return np.array(X), np.array(y), np.array(idxs)

X_train, y_train_seq, idx_train = create_sequences(X_train_df, y_train, HISTORY_HOURS, OFFSET_STEPS)
X_val, y_val_seq, idx_val = create_sequences(X_val_df, y_val, HISTORY_HOURS, OFFSET_STEPS)
X_test, y_test_seq, idx_test = create_sequences(X_test_df, y_test, HISTORY_HOURS, OFFSET_STEPS)

# ============================================================
# 7. MODEL (SANE, NOT OVERKILL)
# ============================================================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(HISTORY_HOURS, len(feature_cols))),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)

model.fit(
    X_train, y_train_seq,
    validation_data=(X_val, y_val_seq),
    epochs=5,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# ============================================================
# 8. PREDICT ON LAST 10% (TEST SET)
# ============================================================
y_pred_test_scaled = model.predict(X_test)
y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled).flatten()
y_test_actual = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

# Build a DataFrame with predicted forward returns and reconstruct price
pred_df = pd.DataFrame({
    "RET_1H_FWD_ACTUAL": y_test_actual,
    "RET_1H_FWD_PRED": y_pred_test
}, index=idx_test)

# Reconstruct price from returns: P_{t+1} = P_t * (1 + ret)
# We align with the CLOSE price at the previous step
base_close = df.loc[pred_df.index, "CLOSE"].shift( -AHEAD_HOURS )  # align so that fwd ret applies to that base
# If shift misaligns at the end, fill from original CLOSE
base_close = base_close.fillna(method="ffill").fillna(method="bfill")

base_close_aligned = base_close.values[:len(pred_df)]

pred_df["CLOSE_ACTUAL"] = base_close_aligned * (1 + pred_df["RET_1H_FWD_ACTUAL"].values)
pred_df["CLOSE_PRED"] = base_close_aligned * (1 + pred_df["RET_1H_FWD_PRED"].values)
# ============================================================
# 9. PLOTTING
# ============================================================
plt.figure(figsize=(18, 8))

# Full BTC for context
plt.plot(df.index, df["CLOSE"], label="BTC CLOSE (Full)", color="black", alpha=0.3)

# Last 10% actual vs predicted (reconstructed)
plt.plot(pred_df.index, pred_df["CLOSE_ACTUAL"], label="BTC CLOSE Actual (Last 10%)", color="blue")
plt.plot(pred_df.index, pred_df["CLOSE_PRED"], label="BTC CLOSE Predicted (Last 10%)", color="red")

plt.title("BTC — Actual vs Predicted (1h Returns → Reconstructed Price, Last 10%)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: also inspect the returns directly
plt.figure(figsize=(18, 5))
plt.plot(pred_df.index, pred_df["RET_1H_FWD_ACTUAL"], label="Actual 1h Forward Return", color="blue", alpha=0.7)
plt.plot(pred_df.index, pred_df["RET_1H_FWD_PRED"], label="Predicted 1h Forward Return", color="red", alpha=0.7)
plt.axhline(0, color="black", linewidth=1)
plt.title("BTC — Actual vs Predicted 1h Forward Returns (Last 10%)")
plt.xlabel("Time")
plt.ylabel("Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
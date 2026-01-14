import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CSV_PATH = "btc_hourly_ohclv_ta.csv"
df = pd.read_csv(CSV_PATH)

df["dt"] = pd.to_datetime(df["UNIX_TIMESTAMP"], unit="s")
df = df.sort_values("dt").reset_index(drop=True)

df["log_close"] = np.log(df["CLOSE"])

feature_cols = [
    "OPEN","HIGH","LOW","CLOSE","VOLUME",
    "SMA_20","EMA_12","EMA_26","MACD","MACD_SIGNAL","MACD_DIFF",
    "RSI","BB_HIGH","BB_LOW","BB_WIDTH",
    "STOCH_K","STOCH_D","VOLUME_SMA","MFI",
    "ATR","PRICE_CHANGE","HIGH_LOW_RATIO","CLOSE_OPEN_RATIO",
    "VOLATILITY_30D","PRICE_VOLATILITY_30D","HL_VOLATILITY_30D"
]

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

features = df[feature_cols].astype(float).values
targets_log = df["log_close"].values
timestamps = df["dt"].values

def create_sequences_multivariate(X, y, history, horizon):

    Xs, ys = [], []
    n = len(X)
    for i in range(n - history - horizon + 1):
        Xs.append(X[i:i+history])
        ys.append(y[i+history+horizon-1])
    return np.array(Xs), np.array(ys)

history = 72
horizons = [6,12,24,48]   
results = []

for horizon in horizons:
    print(f"\nRunning experiment: {history}h -> {horizon}h")

    X_seq_raw, y_seq = create_sequences_multivariate(features, targets_log, history, horizon)

    n = len(X_seq_raw)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train_raw = X_seq_raw[:train_end]  
    X_val_raw   = X_seq_raw[train_end:val_end]
    X_test_raw  = X_seq_raw[val_end:]
    y_train = y_seq[:train_end]
    y_val   = y_seq[train_end:val_end]
    y_test  = y_seq[val_end:]

    n_features = X_train_raw.shape[2]
    scaler = MinMaxScaler(feature_range=(0,1))
    X_train_flat_for_scaler = X_train_raw.reshape(-1, n_features) 
    scaler.fit(X_train_flat_for_scaler)

    def scale_sequences(X_raw):
        n_samples = X_raw.shape[0]
        X_flat = X_raw.reshape(-1, n_features)
        X_scaled_flat = scaler.transform(X_flat)
        return X_scaled_flat.reshape(n_samples, history, n_features)

    X_train = scale_sequences(X_train_raw)
    X_val   = scale_sequences(X_val_raw)
    X_test  = scale_sequences(X_test_raw)
    X_all   = scale_sequences(X_seq_raw)   


    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat   = X_val.reshape(X_val.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0], -1)
    X_all_flat   = X_all.reshape(X_all.shape[0], -1)

    def build_lstm(input_shape):
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1)
        ])
        model.compile(optimizer=optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
        return model

    lstm = build_lstm((history, n_features))
    es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    lstm.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1,
        batch_size=64,
        callbacks=[es],
        verbose=1
    )

    y_pred_log_lstm = lstm.predict(X_all, verbose=1).flatten()
    y_pred_price_lstm = np.exp(y_pred_log_lstm)
    y_true_price = np.exp(y_seq)

    mae_lstm = mean_absolute_error(y_true_price, y_pred_price_lstm)
    rmse_lstm = np.sqrt(mean_squared_error(y_true_price, y_pred_price_lstm))
    dir_true = np.sign(np.diff(y_true_price))
    dir_pred_lstm = np.sign(np.diff(y_pred_price_lstm))
    diracc_lstm = (dir_true == dir_pred_lstm).mean()

    results.append({
        "model": "LSTM",
        "horizon": horizon,
        "mae": mae_lstm,
        "rmse": rmse_lstm,
        "diracc": diracc_lstm,
        "y_pred": y_pred_price_lstm,
        "y_true": y_true_price,
        "timestamps": timestamps[history+horizon-1:history+horizon-1+len(y_true_price)]
    })


rows = []
for r in results:
    rows.append([r["model"], r["horizon"], r["mae"], r["rmse"], r["diracc"]])
comp_df = pd.DataFrame(rows, columns=["Model","Horizon_h","MAE","RMSE","DirAcc"])
print("\n=== Model Comparison ===")
print(comp_df.sort_values(["Horizon_h","Model"]).to_string(index=False))

for horizon in sorted(set(r["horizon"] for r in results)):
    plt.figure(figsize=(16,6))
    subset = [r for r in results if r["horizon"]==horizon]
    plt.plot(subset[0]["timestamps"], subset[0]["y_true"], color="black", label="Actual", alpha=0.6)
    for r in subset:
        plt.plot(r["timestamps"], r["y_pred"], label=f"{r['model']} Pred", alpha=0.8)
    plt.title(f"{history}h -> {horizon}h predictions")
    plt.xlabel("Time")
    plt.ylabel("BTC Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
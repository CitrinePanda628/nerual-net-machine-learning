"""
btc_lstm_gru_pipeline.py

Requirements:
- pandas
- numpy
- scikit-learn
- matplotlib
- tensorflow (>=2.x)

Install if needed:
pip install pandas numpy scikit-learn matplotlib tensorflow
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# ---------------------------
# Reproducibility
# ---------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------
# User parameters
# ---------------------------
CSV_PATH = "btc_hourly_ohclv_ta.csv"   # update path if needed
SEQ_LEN = 24                          # past 24 hours as input
PRED_HORIZON = 2                      # predict close price 2 hours ahead
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 1e-3

# ---------------------------
# 1) Load and basic cleanup
# ---------------------------
df = pd.read_csv(CSV_PATH)

# Drop DATETIME (redundant) and keep UNIX_TIMESTAMP only if needed
if "DATETIME" in df.columns:
    df = df.drop(columns=["DATETIME"])

# Keep a copy of raw close for plotting later
df["CLOSE_raw"] = df["CLOSE"].astype(float)

# ---------------------------
# 2) Create target: CLOSE shifted -PRED_HORIZON
# ---------------------------
df["target_close_future"] = df["CLOSE"].shift(-PRED_HORIZON)
# Drop rows with NaN targets (end of series)
df = df.dropna().reset_index(drop=True)

# ---------------------------
# 3) Remove unnecessary columns
# ---------------------------
# We will drop UNIX_TIMESTAMP (time index) from features to avoid leakage.
# Keep CLOSE in features because model can learn price dynamics; we'll scale it.
drop_cols = []
if "UNIX_TIMESTAMP" in df.columns:
    drop_cols.append("UNIX_TIMESTAMP")
# Also drop the raw copy from features (we keep CLOSE in features)
if "CLOSE_raw" in df.columns:
    df = df.drop(columns=["CLOSE_raw"])

# Final feature set (all numeric columns except target)
feature_cols = [c for c in df.columns if c not in ["target_close_future"] + drop_cols]

# ---------------------------
# 4) Train/Val/Test split by time (no shuffle)
#    Fit scalers on training set only
# ---------------------------
n = len(df)
train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

df_train = df.iloc[:train_end].copy()
df_val   = df.iloc[train_end:val_end].copy()
df_test  = df.iloc[val_end:].copy()

# ---------------------------
# 5) Scaling (MinMax 0-1)
#    - Fit scaler_X on X_train
#    - Fit scaler_y on y_train (target close)
# ---------------------------
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_train_raw = df_train[feature_cols].astype(float).values
X_val_raw   = df_val[feature_cols].astype(float).values
X_test_raw  = df_test[feature_cols].astype(float).values

y_train_raw = df_train[["target_close_future"]].astype(float).values
y_val_raw   = df_val[["target_close_future"]].astype(float).values
y_test_raw  = df_test[["target_close_future"]].astype(float).values

# Fit scalers only on training data
scaler_X.fit(X_train_raw)
scaler_y.fit(y_train_raw)

X_train_scaled = scaler_X.transform(X_train_raw)
X_val_scaled   = scaler_X.transform(X_val_raw)
X_test_scaled  = scaler_X.transform(X_test_raw)

y_train_scaled = scaler_y.transform(y_train_raw)
y_val_scaled   = scaler_y.transform(y_val_raw)
y_test_scaled  = scaler_y.transform(y_test_raw)


# ---------------------------
def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len + 1):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len-1])  # align so sequence end corresponds to time t, target is t+PRED_HORIZON already shifted
    return np.array(Xs), np.array(ys)

# Build sequences separately for train/val/test using their own arrays
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQ_LEN)
X_val_seq,   y_val_seq   = create_sequences(X_val_scaled,   y_val_scaled,   SEQ_LEN)
X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test_scaled,  SEQ_LEN)

print("Shapes:")
print("X_train_seq:", X_train_seq.shape, "y_train_seq:", y_train_seq.shape)
print("X_val_seq:  ", X_val_seq.shape,   "y_val_seq:  ", y_val_seq.shape)
print("X_test_seq: ", X_test_seq.shape,  "y_test_seq: ", y_test_seq.shape)

# ---------------------------
# 7) Model builders
# ---------------------------
def build_lstm(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear")
    ])
    opt = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model


input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])

# ---------------------------
# 8) Train LSTM
# ---------------------------
lstm = build_lstm(input_shape)
es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
history_lstm = lstm.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es],
    verbose=2
)


# ---------------------------
# 10) Evaluate on test set
# ---------------------------
def evaluate_model(model, X_seq, y_seq, scaler_y, label="model"):
    # Predict scaled values
    y_pred_scaled = model.predict(X_seq, batch_size=1024)
    # Inverse transform to original price scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_seq.reshape(-1, 1)).flatten()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Directional accuracy: compare sign of (pred - last input close) vs (true - last input close)
    # Need last input close in original scale: take last timestep CLOSE from input sequence (feature index of CLOSE)
    # Find index of CLOSE in feature_cols
    try:
        close_idx = feature_cols.index("CLOSE")
    except ValueError:
        # fallback: if CLOSE not present, use first column
        close_idx = 0

    # Extract last input close (original scale)
    last_input_scaled = X_seq[:, -1, close_idx].reshape(-1, 1)
    last_input = scaler_X.inverse_transform(
        np.concatenate([np.zeros((len(last_input_scaled), X_seq.shape[2])),], axis=1)
    )  # dummy inverse not used; instead we will reconstruct properly below

    # Simpler approach: reconstruct last input CLOSE by inverse transforming a vector where only CLOSE is set
    # Build array of zeros and set CLOSE column to last_input_scaled, then inverse_transform
    zeros = np.zeros((len(last_input_scaled), X_seq.shape[2]))
    zeros[:, close_idx] = last_input_scaled.flatten()
    last_input_unscaled = scaler_X.inverse_transform(zeros)[:, close_idx]

    # Compute directions
    dir_pred = (y_pred - last_input_unscaled) > 0
    dir_true = (y_true - last_input_unscaled) > 0
    dir_acc = np.mean(dir_pred == dir_true)
    mistakes = np.sum(dir_pred != dir_true)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": dir_acc,
        "directional_mistakes": int(mistakes)
    }

print("\nEvaluating LSTM on test set...")
res_lstm = evaluate_model(lstm, X_test_seq, y_test_seq, scaler_y, label="LSTM")
print("LSTM MAE:", res_lstm["mae"], "RMSE:", res_lstm["rmse"], "Dir Acc:", res_lstm["directional_accuracy"], "Mistakes:", res_lstm["directional_mistakes"])


# ---------------------------
# 11) Plot training history (loss & mae)
# ---------------------------
def plot_history(hist, title="Training history"):
    plt.figure(figsize=(10, 4))
    plt.plot(hist.history["loss"], label="train_loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    if "mae" in hist.history:
        plt.plot(hist.history["mae"], label="train_mae", linestyle="--")
        plt.plot(hist.history["val_mae"], label="val_mae", linestyle="--")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_history(history_lstm, "LSTM Training History")

# ---------------------------
# 12) Plot predicted vs actual for test (last 10% region)
# ---------------------------
def plot_pred_vs_true(y_true, y_pred, title="Predicted vs Actual (Test)"):
    plt.figure(figsize=(14, 5))
    plt.plot(y_true, label="Actual", linewidth=1)
    plt.plot(y_pred, label="Predicted", linewidth=1)
    plt.title(title)
    plt.xlabel("Test sample index")
    plt.ylabel("BTC Close Price")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot first N points and also a zoomed window
plot_pred_vs_true(res_lstm["y_true"], res_lstm["y_pred"], title="LSTM: Predicted vs Actual (Test)")

# Zoomed view (last 500 points)
zoom_n = min(500, len(res_lstm["y_true"]))
if zoom_n > 0:
    plot_pred_vs_true(res_lstm["y_true"][-zoom_n:], res_lstm["y_pred"][-zoom_n:], title=f"LSTM Zoom Last {zoom_n} Test Points")

# ---------------------------
# 13) Print summary statistics
# ---------------------------
def print_summary(name, res):
    print(f"\n=== {name} Summary ===")
    print(f"MAE: {res['mae']:.2f}")
    print(f"RMSE: {res['rmse']:.2f}")
    print(f"Directional accuracy: {res['directional_accuracy']*100:.2f}%")
    print(f"Directional mistakes (count): {res['directional_mistakes']}")
    # Additional: percent error relative to price
    mean_price = np.mean(res["y_true"])
    print(f"MAE as % of mean price: {res['mae'] / mean_price * 100:.3f}%")

print_summary("LSTM", res_lstm)


# ---------------------------
# 14) Save models and scalers (optional)
# ---------------------------
# Uncomment to save
# lstm.save("lstm_btc_model.h5")
# gru.save("gru_btc_model.h5")
# import joblib
# joblib.dump(scaler_X, "scaler_X.save")
# joblib.dump(scaler_y, "scaler_y.save")

# ---------------------------
# Notes and suggestions
# ---------------------------
# - This script predicts the CLOSE price 2 hours ahead using the past 24 hours.
# - If training is slow, reduce EPOCHS or BATCH_SIZE, or use a GPU.
# - You can change the target horizon by modifying PRED_HORIZON and recomputing the target.
# - For classification (direction only), convert the target to binary and use 'sigmoid' output with binary_crossentropy.
# - Consider adding more features (log returns, cyclical time features) for improved performance.
# - Walk-forward retraining: for production, retrain periodically with newest data and evaluate with walk-forward CV.
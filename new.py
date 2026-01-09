import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.stats import skew, kurtosis

print("Starting full training pipeline...")


SEQ_LEN = 30
PRED_HORIZON = 1
CLIP_TARGET = 0.05
N_WALK_FOLDS = 4
MIN_TRAIN_YEARS = 5
RANDOM_STATE = 42

SAVE_FOLDER = "trained_models"


if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)
    print(f"Created folder: {SAVE_FOLDER}")
else:
    print(f"Folder already exists: {SAVE_FOLDER}")

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved JSON: {path}")

def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved pickle: {path}")

def progress(msg):
    print(f"[INFO] {msg}")


progress("Loading dataset...")

df = pd.read_csv("magnificent7_ml_dataset_v2.csv")
df["report_date"] = pd.to_datetime(df["report_date"])
df = df.sort_values(["symbol", "report_date"]).reset_index(drop=True)

required_cols = [
    "report_date","symbol","return","return_3d","return_5d","return_10d",
    "rsi_14","macd_hist","price_vs_sma20","volatility_5d","volatility_ratio",
    "bb_width","volume_ratio","adx","symbol_id","target_return"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")


progress("Adding engineered features...")

def add_features(g):
    g = g.sort_values("report_date").reset_index(drop=True)

    g["ret_20d"] = (1 + g["return"]).rolling(20).apply(np.prod, raw=True) - 1
    g["ret_60d"] = (1 + g["return"]).rolling(60).apply(np.prod, raw=True) - 1

    m20 = g["return"].rolling(20).mean()
    s20 = g["return"].rolling(20).std()
    g["rolling_sharpe_20"] = m20 / (s20 + 1e-9)

    g["rolling_skew_60"] = g["return"].rolling(60).apply(lambda x: skew(x, bias=False), raw=False)
    g["rolling_kurt_60"] = g["return"].rolling(60).apply(lambda x: kurtosis(x, fisher=True, bias=False), raw=False)

    g["vol_5d_rolling_mean_20"] = g["volatility_5d"].rolling(20).mean()
    g["vol_5d_rolling_std_20"] = g["volatility_5d"].rolling(20).std()

    g["vol_ratio_rolling_mean_20"] = g["volatility_ratio"].rolling(20).mean()
    g["vol_ratio_rolling_std_20"] = g["volatility_ratio"].rolling(20).std()

    g["vol_ratio_rolling_mean_60"] = g["volume_ratio"].rolling(60).mean()
    g["vol_ratio_rolling_std_60"] = g["volume_ratio"].rolling(60).std()

    g["vol_ratio_mom_5"] = g["volume_ratio"].pct_change(5)

    return g

df = df.groupby("symbol", group_keys=False).apply(add_features)

df["target_return_clipped"] = df["target_return"].clip(-CLIP_TARGET, CLIP_TARGET)

feature_cols = [
    "return","return_3d","return_5d","return_10d","rsi_14","macd_hist",
    "price_vs_sma20","volatility_5d","volatility_ratio","bb_width",
    "volume_ratio","adx","ret_20d","ret_60d","rolling_sharpe_20",
    "rolling_skew_60","rolling_kurt_60","vol_5d_rolling_mean_20",
    "vol_5d_rolling_std_20","vol_ratio_rolling_mean_20",
    "vol_ratio_rolling_std_20","vol_ratio_rolling_mean_60",
    "vol_ratio_rolling_std_60","vol_ratio_mom_5"
]


def build_sequences(g):
    g = g.sort_values("report_date").reset_index(drop=True)
    X, y, dates = [], [], []

    for i in range(SEQ_LEN, len(g) - PRED_HORIZON + 1):
        window = g.iloc[i-SEQ_LEN:i]
        target = g.iloc[i+PRED_HORIZON-1]["target_return_clipped"]

        if window[feature_cols].isna().any().any(): 
            continue
        if pd.isna(target):
            continue

        X.append(window[feature_cols].values)
        y.append(target)
        dates.append(g.iloc[i+PRED_HORIZON-1]["report_date"])

    if not X:
        return None, None, None

    return np.array(X), np.array(y), np.array(dates)

symbol_groups = dict(list(df.groupby("symbol")))


def build_lstm(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x2 = layers.LSTM(64, return_sequences=False)(x)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.2)(x2)
    last = layers.Lambda(lambda t: t[:, -1, :])(inp)
    proj = layers.Dense(64, activation="relu")(last)
    x = layers.Add()([x2, proj])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1)(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.Huber(),
                  metrics=["mae"])
    return model


progress("Starting walk-forward validation...")

start_year = df["report_date"].dt.year.min()
end_year = df["report_date"].dt.year.max()
years = np.arange(start_year + MIN_TRAIN_YEARS, end_year)

if len(years) < N_WALK_FOLDS:
    split_years = years
else:
    split_years = np.linspace(years[0], years[-1], N_WALK_FOLDS, dtype=int)

walk_results = []
all_preds_hist = []
all_trues_hist = []

fold_idx = 0

for split_year in split_years:
    fold_idx += 1
    progress(f"Fold {fold_idx}/{len(split_years)} — split year {split_year}")

    fold_preds = []
    fold_trues = []

    for stock_i, (symbol, g) in enumerate(symbol_groups.items(), start=1):
        progress(f"Training stock {symbol} ({stock_i}/7)")

        train_mask = g["report_date"].dt.year < split_year
        test_mask = g["report_date"].dt.year == split_year

        if train_mask.sum() < SEQ_LEN + 50:
            continue

        X_train, y_train, _ = build_sequences(g.loc[train_mask])
        X_test, y_test, _ = build_sequences(g.loc[test_mask])

        if X_train is None or X_test is None:
            continue

        n_features = X_train.shape[2]

        feat_scaler = StandardScaler()
        X_train_2d = X_train.reshape(-1, n_features)
        X_test_2d = X_test.reshape(-1, n_features)
        feat_scaler.fit(X_train_2d)
        X_train_s = feat_scaler.transform(X_train_2d).reshape(X_train.shape)
        X_test_s = feat_scaler.transform(X_test_2d).reshape(X_test.shape)

        y_scaler = StandardScaler()
        y_train_s = y_scaler.fit_transform(y_train.reshape(-1,1)).flatten()

        progress("Training LSTM...")
        lstm = build_lstm((SEQ_LEN, n_features))
        lstm.fit(X_train_s, y_train_s, epochs=30, batch_size=64, verbose=0)
        y_lstm_s = lstm.predict(X_test_s, verbose=0).flatten()
        y_lstm = y_scaler.inverse_transform(y_lstm_s.reshape(-1,1)).flatten()

        progress("Training XGBoost...")
        X_train_tab = X_train_s[:, -1, :]
        X_test_tab = X_test_s[:, -1, :]
        xgb = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, objective="reg:squarederror"
        )
        xgb.fit(X_train_tab, y_train)
        y_xgb = xgb.predict(X_test_tab)

        progress("Training RandomForest...")
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=6,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        rf.fit(X_train_tab, y_train)
        y_rf = rf.predict(X_test_tab)

        progress("Training LinearRegression...")
        lin = LinearRegression()
        lin.fit(X_train_tab, y_train)
        y_lin = lin.predict(X_test_tab)

        y_ens = (y_lstm + y_xgb + y_rf + y_lin) / 4.0

        fold_preds.append(y_ens)
        fold_trues.append(y_test)

    if fold_preds:
        fold_preds = np.concatenate(fold_preds)
        fold_trues = np.concatenate(fold_trues)

        all_preds_hist.append(fold_preds)
        all_trues_hist.append(fold_trues)

        mae = np.mean(np.abs(fold_preds - fold_trues))
        mse = np.mean((fold_preds - fold_trues)**2)
        acc = np.mean(np.sign(fold_preds) == np.sign(fold_trues))

        walk_results.append({
            "year": int(split_year),
            "mae": float(mae),
            "mse": float(mse),
            "direction_acc": float(acc)
        })

        progress(f"Fold results — MAE: {mae:.6f}, MSE: {mse:.6f}, ACC: {acc:.3f}")

save_json(os.path.join(SAVE_FOLDER, "walkforward_results.json"), walk_results)


progress("Starting future-day prediction...")

tmr = {
    'AAPL': np.float64(0.5437378301921518),
    'MSFT': np.float64(0.4008430100417367),
    'GOOGL': np.float64(1.553924485882446),
    'AMZN': np.float64(0.2601869818310123),
    'META': np.float64(-0.8548423508164742),
    'NVDA': np.float64(3.9336166302974753),
    'TSLA': np.float64(-0.44893146037197507)
}

future_preds = {}

for symbol, g in symbol_groups.items():
    progress(f"Training full-history model for {symbol}")

    X_seq, y_seq, _ = build_sequences(g)
    if X_seq is None:
        continue

    n_features = X_seq.shape[2]

    feat_scaler = StandardScaler()
    X_2d = X_seq.reshape(-1, n_features)
    feat_scaler.fit(X_2d)
    X_s = feat_scaler.transform(X_2d).reshape(X_seq.shape)

    y_scaler = StandardScaler()
    y_s = y_scaler.fit_transform(y_seq.reshape(-1,1)).flatten()

    lstm = build_lstm((SEQ_LEN, n_features))
    lstm.fit(X_s, y_s, epochs=30, batch_size=64, verbose=0)
    y_lstm_s = lstm.predict(X_s[-1:].reshape(1,SEQ_LEN,n_features), verbose=0).flatten()
    y_lstm = y_scaler.inverse_transform(y_lstm_s.reshape(-1,1)).flatten()[0]

    X_tab = X_s[:, -1, :]

    xgb = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, objective="reg:squarederror"
    )
    xgb.fit(X_tab, y_seq)
    y_xgb = xgb.predict(X_tab[-1:].reshape(1,-1))[0]

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=6,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_tab, y_seq)
    y_rf = rf.predict(X_tab[-1:].reshape(1,-1))[0]

    lin = LinearRegression()
    lin.fit(X_tab, y_seq)
    y_lin = lin.predict(X_tab[-1:].reshape(1,-1))[0]

    y_ens = (y_lstm + y_xgb + y_rf + y_lin) / 4.0
    future_preds[symbol] = float(y_ens)

    # Save models
    stock_folder = os.path.join(SAVE_FOLDER, symbol)
    os.makedirs(stock_folder, exist_ok=True)

    lstm.save(os.path.join(stock_folder, "lstm.keras"))
    save_pickle(os.path.join(stock_folder, "xgb.pkl"), xgb)
    save_pickle(os.path.join(stock_folder, "rf.pkl"), rf)
    save_pickle(os.path.join(stock_folder, "linear.pkl"), lin)
    save_pickle(os.path.join(stock_folder, "feature_scaler.pkl"), feat_scaler)
    save_pickle(os.path.join(stock_folder, "target_scaler.pkl"), y_scaler)

pred_vals = np.array([future_preds[s] for s in tmr])
actual_vals = np.array([tmr[s] for s in tmr])
actual_clipped = np.clip(actual_vals, -CLIP_TARGET, CLIP_TARGET)

mae_f = float(np.mean(np.abs(pred_vals - actual_clipped)))
mse_f = float(np.mean((pred_vals - actual_clipped)**2))
acc_f = float(np.mean(np.sign(pred_vals) == np.sign(actual_vals)))

future_results = {
    "predictions": future_preds,
    "actual": {k: float(v) for k,v in tmr.items()},
    "metrics": {
        "mae_vs_clipped": mae_f,
        "mse_vs_clipped": mse_f,
        "directional_accuracy": acc_f
    }
}

save_json(os.path.join(SAVE_FOLDER, "future_day_results.json"), future_results)

ensemble_config = {
    "weights": {
        "lstm": 0.25,
        "xgb": 0.25,
        "rf": 0.25,
        "linear": 0.25
    }
}
save_json(os.path.join(SAVE_FOLDER, "ensemble_config.json"), ensemble_config)



progress("Training complete.")
print("\nWalk-forward results:")
print(walk_results)

print("\nFuture-day prediction results:")
print(future_results)
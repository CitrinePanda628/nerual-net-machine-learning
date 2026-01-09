import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

print("tsal.py")

tmr = {'AAPL': np.float64(0.5437378301921518), 
       'MSFT': np.float64(0.4008430100417367), 
       'GOOGL': np.float64(1.553924485882446), 
       'AMZN': np.float64(0.2601869818310123), 
       'META': np.float64(-0.8548423508164742), 
       'NVDA': np.float64(3.9336166302974753), 
       'TSLA': np.float64(-0.44893146037197507)}


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

INPUT_CSV = "magnificent7_ml_dataset.csv"
SAVED_MODEL_DIR = "stock_nn_tf_model"
SCALER_PATH = "feature_scaler.pkl"
METRICS_CSV = "company_eval_test_metrics.csv"
TOMORROW_CSV = "tomorrow_predictions.csv"

FEATURES_TO_SCALE = [
    "return",
    "rsi_14",
    "macd_hist",
    "volume_change",
    "volatility_5d"
]
CATEGORICAL = ["symbol_id"]
TARGET = "target_return"
SPLIT_DATE = "2023-01-01"

df = pd.read_csv(INPUT_CSV, dtype={"symbol": str, "symbol_id": str})

df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
bad_dates = df["report_date"].isna().sum()
if bad_dates:
    raise SystemExit(f"ERROR: {bad_dates} unparsable report_date rows. Fix CSV first.")

dups = df.duplicated(subset=["symbol", "report_date"]).sum()
if dups:
    print(f"Warning: dropping {dups} duplicate (symbol, report_date) rows")
    df = df.drop_duplicates(subset=["symbol", "report_date"], keep="first")

essential = FEATURES_TO_SCALE + CATEGORICAL + [TARGET]
missing_before = df[essential].isna().sum()
print("Missing values per essential column (before drop):")
print(missing_before)
df = df.dropna(subset=essential)
print("Rows after dropping essential-NaN rows:", len(df))


df = df.sort_values(["symbol_id", "report_date"]).reset_index(drop=True)


train_mask = df["report_date"] < pd.to_datetime(SPLIT_DATE)
train_df = df.loc[train_mask].copy()
test_df = df.loc[~train_mask].copy()

print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")
print("Data range (all):", df['report_date'].min().date(), "→", df['report_date'].max().date())
print("Train range:", train_df['report_date'].min().date(), "→", train_df['report_date'].max().date())
print("Test range:", test_df['report_date'].min().date(), "→", test_df['report_date'].max().date())


scaler = StandardScaler()
train_num = train_df[FEATURES_TO_SCALE].astype(float)
test_num = test_df[FEATURES_TO_SCALE].astype(float)

train_num_scaled = pd.DataFrame(scaler.fit_transform(train_num), columns=FEATURES_TO_SCALE, index=train_df.index)
test_num_scaled = pd.DataFrame(scaler.transform(test_num), columns=FEATURES_TO_SCALE, index=test_df.index)

train_dummies = pd.get_dummies(train_df["symbol_id"], prefix="sym")
test_dummies = pd.get_dummies(test_df["symbol_id"], prefix="sym")

test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)

X_train = pd.concat([train_num_scaled.reset_index(drop=True), train_dummies.reset_index(drop=True)], axis=1)
X_test  = pd.concat([test_num_scaled.reset_index(drop=True), test_dummies.reset_index(drop=True)], axis=1)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)


FEATURE_COLUMNS = X_train.columns.tolist()
print("Final feature columns count:", len(FEATURE_COLUMNS))

y_train = train_df[TARGET].astype(float).reset_index(drop=True)
y_test = test_df[TARGET].astype(float).reset_index(drop=True)


xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',   
    random_state=SEED
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=1
)


y_pred_xgb = xgb_model.predict(X_test)

test_eval_xgb = test_df.reset_index(drop=True).copy()
test_eval_xgb["pred_return"] = y_pred_xgb
test_eval_xgb["abs_error"] = (test_eval_xgb["target_return"] - test_eval_xgb["pred_return"]).abs()
test_eval_xgb["sign_correct"] = np.sign(test_eval_xgb["target_return"]) == np.sign(test_eval_xgb["pred_return"])

company_metrics_xgb = test_eval_xgb.groupby("symbol").agg(
    rows=("symbol","size"),
    actual_mean=("target_return","mean"),
    predicted_mean=("pred_return","mean"),
    mae=("abs_error","mean"),
    median_error=("abs_error","median"),
    sign_accuracy=("sign_correct","mean")
).reset_index()

company_metrics_xgb["actual_mean_%"] = company_metrics_xgb["actual_mean"] * 100
company_metrics_xgb["predicted_mean_%"] = company_metrics_xgb["predicted_mean"] * 100
company_metrics_xgb["mae_%"] = company_metrics_xgb["mae"] * 100
company_metrics_xgb["median_error_%"] = company_metrics_xgb["median_error"] * 100

company_metrics_xgb = company_metrics_xgb[[
    "symbol","rows","actual_mean_%","predicted_mean_%","mae_%","median_error_%","sign_accuracy"
]]

print("\nPer-company evaluation on TEST set (XGBoost):")
print(company_metrics_xgb.to_string(index=False))
company_metrics_xgb.to_csv("company_eval_test_metrics_xgb.csv", index=False)



def build_model(input_dim, num_nodes=128, dropout_prob=0.2, lr=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae']
    )
    return model

model = build_model(X_train.shape[1], num_nodes=128, dropout_prob=0.2, lr=0.001)
model.summary()

history = model.fit(
    X_train.values, y_train.values,
    validation_split=0.1,
    epochs=50,
    batch_size=64,
    verbose=1
)

y_pred_test = model.predict(X_test.values).flatten()

test_eval = test_df.reset_index(drop=True).copy()
test_eval["pred_return"] = y_pred_test
test_eval["abs_error"] = (test_eval["target_return"] - test_eval["pred_return"]).abs()
test_eval["sign_correct"] = np.sign(test_eval["target_return"]) == np.sign(test_eval["pred_return"])

company_metrics = test_eval.groupby("symbol").agg(
    rows=("symbol","size"),
    actual_mean=("target_return","mean"),
    predicted_mean=("pred_return","mean"),
    mae=("abs_error","mean"),
    median_error=("abs_error","median"),
    sign_accuracy=("sign_correct","mean")
).reset_index()

company_metrics["actual_mean_%"] = company_metrics["actual_mean"] * 100
company_metrics["predicted_mean_%"] = company_metrics["predicted_mean"] * 100
company_metrics["mae_%"] = company_metrics["mae"] * 100
company_metrics["median_error_%"] = company_metrics["median_error"] * 100

company_metrics = company_metrics[[
    "symbol","rows","actual_mean_%","predicted_mean_%","mae_%","median_error_%","sign_accuracy"
]]

print("\nPer-company evaluation on TEST set:")
print(company_metrics.to_string(index=False))

company_metrics.to_csv(METRICS_CSV, index=False)
print(f"\nSaved per-company metrics -> {METRICS_CSV}")


latest = df.sort_values("report_date").groupby("symbol").tail(1).reset_index(drop=True)

latest_num = latest[FEATURES_TO_SCALE].astype(float)
latest_num_scaled = pd.DataFrame(scaler.transform(latest_num), columns=FEATURES_TO_SCALE, index=latest.index)

latest_dummies = pd.get_dummies(latest["symbol_id"], prefix="sym")
latest_dummies = latest_dummies.reindex(columns=train_dummies.columns, fill_value=0)

latest_X = pd.concat([latest_num_scaled.reset_index(drop=True), latest_dummies.reset_index(drop=True)], axis=1)

latest_X = latest_X.reindex(columns=FEATURE_COLUMNS, fill_value=0)

latest_X = latest_X.astype(np.float32)

latest["predicted_return"] = model.predict(latest_X.values).flatten()
latest["predicted_return_%"] = latest["predicted_return"] * 100

tomorrow_ranking = latest[["symbol","predicted_return_%"]].sort_values("predicted_return_%", ascending=False).reset_index(drop=True)
print("\nTomorrow prediction ranking (one row per symbol):")
print(tomorrow_ranking.to_string(index=False))

tomorrow_ranking.to_csv(TOMORROW_CSV, index=False)
print(f"Saved tomorrow predictions -> {TOMORROW_CSV}")


comparison_table = company_metrics.merge(
    company_metrics_xgb, on="symbol", suffixes=("_NN","_XGB")
)

stocks = comparison_table["symbol"]
actual = comparison_table["actual_mean_%_NN"]
nn_pred = comparison_table["predicted_mean_%_NN"]
xgb_pred = comparison_table["predicted_mean_%_XGB"]

x = np.arange(len(stocks)) 
width = 0.25  

plt.figure(figsize=(10,6))
plt.bar(x - width, actual, width=width, label="Actual", color="#1f77b4")
plt.bar(x, nn_pred, width=width, label="NN Pred", color="#ff7f0e")
plt.bar(x + width, xgb_pred, width=width, label="XGB Pred", color="#2ca02c")

plt.xticks(x, stocks)
plt.ylabel("Percentage Return (%)")
plt.title("Actual vs NN vs XGB Predicted Returns by Stock")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Show
plt.tight_layout()
plt.show()



comparison = company_metrics.merge(
    company_metrics_xgb, on="symbol", suffixes=("_NN","_XGB")
)
print("\nComparison NN vs XGBoost:")
print(comparison.to_string(index=False))


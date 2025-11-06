import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
from feature_engineering import compute_features

def label_direction(df, threshold=0.0005):
    df["future_return"] = df["close"].shift(-1) / df["close"] - 1
    conditions = [
        (df["future_return"] > threshold),
        (df["future_return"] < -threshold)
    ]
    choices = ["BUY", "SELL"]
    df["target"] = np.select(conditions, choices, default="HOLD")
    df.dropna(inplace=True)
    return df

def main():
    # Load data historis
    df = pd.read_csv("data/XAUUSDM1.csv")

    # Hitung fitur teknikal
    df = compute_features(df)

    # Label data
    df = label_direction(df)

    # Tentukan fitur dan target
    features = ["open", "high", "low", "close", "volume", "ema_20", "rsi_14", "macd", "macd_signal", "atr"]
    X = df[features]
    y = df["target"]

    # Split data train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Latih model LightGBM
    lgbm_model = LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=200)
    lgbm_model.fit(X_train, y_train)
    y_pred_lgbm = lgbm_model.predict(X_test)

    print("LightGBM Model Accuracy:", accuracy_score(y_test, y_pred_lgbm))
    print(classification_report(y_test, y_pred_lgbm))

    # Latih model CatBoost
    cat_model = CatBoostClassifier(depth=6, learning_rate=0.05, iterations=200, verbose=False)
    cat_model.fit(X_train, y_train)
    y_pred_cat = cat_model.predict(X_test)

    print("CatBoost Model Accuracy:", accuracy_score(y_test, y_pred_cat))
    print(classification_report(y_test, y_pred_cat))

    # Simpan model terlatih
    joblib.dump(lgbm_model, "model/lightgbm_model.pkl")
    cat_model.save_model("model/catboost_model.pkl")
    print("Model training selesai dan sudah disimpan.")

if __name__ == "__main__":
    main()
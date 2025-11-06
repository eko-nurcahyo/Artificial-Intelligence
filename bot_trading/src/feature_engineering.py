import pandas as pd
import ta

def compute_features(df):
    """
    Menghitung fitur indikator teknikal utama untuk data harga.
    
    Parameter:
    df : pandas.DataFrame
        DataFrame dengan kolom minimal ['open', 'high', 'low', 'close', 'volume']
    
    Return:
    df : pandas.DataFrame
        DataFrame dengan fitur teknikal tambahan tanpa NaN
    """
    try:
        # EMA 20 periode
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

        # RSI 14 periode
        rsi_indicator = ta.momentum.RSIIndicator(df["close"], window=14)
        df["rsi_14"] = rsi_indicator.rsi()

        # MACD dan sinyal MACD
        macd_indicator = ta.trend.MACD(df["close"])
        df["macd"] = macd_indicator.macd()
        df["macd_signal"] = macd_indicator.macd_signal()

        # ATR 14 periode
        atr_indicator = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14)
        df["atr"] = atr_indicator.average_true_range()

        # Buang baris yang memiliki NaN akibat perhitungan indikator
        df.dropna(inplace=True)

        return df

    except Exception as e:
        print(f"Error in compute_features: {e}")
        # Jika error, kembalikan DataFrame asli tanpa perubahan supaya tidak crash
        return df

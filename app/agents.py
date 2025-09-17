import yfinance as yf
import pandas as pd
import numpy as np
from app.models_loader import xgb_model, rf_model

FEATURES = ['Open', 'High', 'Low', 'Volume', 'MA10', 'MA50', 'MA200',
            'EMA10', 'EMA50', 'Return', 'Volatility', 'Momentum', 'RSI',
            'MACD', 'MACD_signal', 'ATR', 'Close_lag1', 'Close_lag2', 'Close_lag3']

class DataAgent:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.df = None

    def run(self):
        data = yf.download(self.ticker, period="1y", interval="1d", auto_adjust=False)
        if data.empty:
            raise ValueError(f"No data for {self.ticker}")
        df = data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].strip() for col in df.columns]
        df.columns = [str(c).strip() for c in df.columns]
        if 'Close' not in df.columns:
            if 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
            else:
                raise ValueError("No Close column found.")
        self.df = df
        return self.df

class FeatureAgent:
    def __init__(self, df):
        self.df = df.copy()

    def run(self):
        df = self.df
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['MA200'] = df['Close'].rolling(200).mean()
        df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Return'].rolling(10).std()
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        df['RSI'] = 100 - (100 / (1 + df['Return'].rolling(14).mean() / df['Return'].rolling(14).std()))
        df['MACD'] = df['EMA10'] - df['EMA50']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['ATR'] = df['High'] - df['Low']
        df['Close_lag1'] = df['Close'].shift(1)
        df['Close_lag2'] = df['Close'].shift(2)
        df['Close_lag3'] = df['Close'].shift(3)
        df = df.dropna()
        self.df = df
        return df

class ModelAgent:
    def __init__(self, df):
        self.df = df

    def run(self):
        X_last = self.df[FEATURES].iloc[[-1]]
        pred_xgb = xgb_model.predict(X_last)[0]
        pred_rf = rf_model.predict(X_last)[0]
        pred_avg = (pred_xgb + pred_rf)/2
        return pred_xgb, pred_rf, pred_avg

class ComparisonAgent:
    def __init__(self, last_close, predicted_avg):
        self.last_close = last_close
        self.predicted_avg = predicted_avg

    def run(self):
        if self.predicted_avg > self.last_close * 1.01:
            signal = "Buy"
        elif self.predicted_avg < self.last_close * 0.99:
            signal = "Sell"
        else:
            signal = "Neutral"
        return signal

class ConfidenceAgent:
    def __init__(self, pred_xgb, pred_rf, last_close):
        self.pred_xgb = pred_xgb
        self.pred_rf = pred_rf
        self.last_close = last_close

    def run(self):
        confidence = 100 - abs(self.pred_xgb - self.pred_rf)/self.last_close * 100
        return round(confidence, 2)

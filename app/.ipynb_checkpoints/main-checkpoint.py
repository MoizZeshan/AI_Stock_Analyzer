import os
import pickle
import shap
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- Relative import for agents ---
from .agents import DataAgent, FeatureAgent, ComparisonAgent, ConfidenceAgent

# --- Load Model ---
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "generalized_model.pkl"))
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

explainer = shap.TreeExplainer(model)

# --- FastAPI ---
app = FastAPI(title="AI Stock Analyzer")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Serve UI ---
UI_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui"))
if os.path.exists(UI_FOLDER):
    app.mount("/ui", StaticFiles(directory=UI_FOLDER), name="ui")

    # ✅ Serve index.html automatically at /ui
    @app.get("/ui")
    async def serve_index():
        index_file = os.path.join(UI_FOLDER, "index.html")
        return FileResponse(index_file)

# --- Root route ---
@app.get("/")
def root():
    return {"message": "AI Stock Analyzer API Running!"}

# --- Prediction Endpoint ---
@app.get("/predict")
def predict(ticker: str):
    try:
        # --- Fetch & process data ---
        df = DataAgent(ticker).run()
        df_features = FeatureAgent(df).run()

        xgb_features = [
            "Open", "High", "Low", "Volume", "MA10", "MA50", "MA200",
            "EMA10", "EMA50", "Return", "Volatility", "Momentum", "RSI",
            "MACD", "MACD_signal", "ATR", "Close_lag1", "Close_lag2", "Close_lag3"
        ]

        df_features = df_features[xgb_features + ["Close"]]
        latest_features = df_features[xgb_features].iloc[[-1]]

        # --- Model prediction ---
        pred_xgb = float(model.predict(latest_features)[0])
        last_close = float(df_features['Close'].iloc[-1])

        # --- Trading signal ---
        signal_text = ComparisonAgent(last_close, pred_xgb).run()

        # --- Confidence calculation ---
        recent_returns = df_features['Return'].dropna()
        raw_confidence = float(ConfidenceAgent(pred_xgb, last_close, recent_returns).run())
        if raw_confidence > 80:
            confidence = "High"
        elif raw_confidence > 50:
            confidence = "Medium"
        else:
            confidence = "Low"

        # --- SHAP reasoning (top 3 features) ---
        shap_values = explainer(latest_features)
        feature_importance = pd.DataFrame({
            "feature": xgb_features,
            "shap_value": shap_values.values[0]
        })
        feature_importance["abs_shap"] = feature_importance["shap_value"].abs()
        top_features = feature_importance.sort_values(by="abs_shap", ascending=False).head(3)

        reasoning_lines = [
            f"{row['feature']} had a {'positive' if row['shap_value'] > 0 else 'negative'} impact"
            for _, row in top_features.iterrows()
        ]

        # --- Final FLAT response ---
        return {
            "ticker": ticker.upper(),
            "last_close": round(last_close, 2),
            "predicted_close": round(pred_xgb, 2),
            "signal": signal_text,
            "confidence": confidence,
            "reasoning": reasoning_lines
        }

    except Exception as e:
        return {"error": str(e)}

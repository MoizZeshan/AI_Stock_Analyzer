from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.agents import DataAgent, FeatureAgent, ModelAgent, ComparisonAgent, ConfidenceAgent

app = FastAPI(title="AI Stock Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "AI Stock Analyzer API Running!"}

@app.get("/predict")
def predict(ticker: str):
    try:
        # Run agents sequentially
        df = DataAgent(ticker).run()
        df_features = FeatureAgent(df).run()
        pred_xgb, pred_rf, pred_avg = ModelAgent(df_features).run()
        last_close = df_features['Close'].iloc[-1]
        signal = ComparisonAgent(last_close, pred_avg).run()
        confidence = ConfidenceAgent(pred_xgb, pred_rf, last_close).run()

        return {
            "ticker": ticker.upper(),
            "last_close": round(last_close, 2),
            "predicted_close": round(pred_avg, 2),
            "signal": signal,
            "confidence": confidence
        }
    except Exception as e:
        return {"error": str(e)}

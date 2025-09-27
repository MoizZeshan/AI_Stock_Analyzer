import joblib
import os

# Use relative path to models folder inside your repo
models_folder = os.path.join(os.path.dirname(__file__), "..", "models")

# Load the XGBoost model
xgb_model = joblib.load(os.path.join(models_folder, "generalized_model.pkl"))

import joblib
import os

models_folder = r"C:\Users\haier\AI_Stock_Analyzer\models"

# Load only XGBoost model
xgb_model = joblib.load(os.path.join(models_folder, "generalized_model.pkl"))

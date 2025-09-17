import joblib
import os

models_folder = r"C:\Users\haier\AI_Stock_Analyzer\models"

xgb_model = joblib.load(os.path.join(models_folder, "generalized_xgb_model.pkl"))
rf_model = joblib.load(os.path.join(models_folder, "generalized_rf_model.pkl"))

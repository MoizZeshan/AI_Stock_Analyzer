import gdown
import os

# Google Drive file ID
file_id = "1NjGDUbYoI7n5czWrAplYkwz3o-_HLA3z"
output_path = "models/generalized_rf_model.pkl"

# Ensure folder exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Direct download URL
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, output_path, quiet=False, fuzzy=True)

print("Model downloaded successfully!")

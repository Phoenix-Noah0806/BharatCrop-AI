
import os
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

# Load environment variables
load_dotenv(BASE_DIR / ".env")

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("MONGODB_DATABASE", "BharatCropAI")

# Load trained models
print("Loading trained models...")

regression_model = joblib.load(
    MODEL_DIR / "crop_regression_model.joblib"
)

rf_classifier = joblib.load(
    MODEL_DIR / "crop_classification_model.joblib"
)

dt_classifier = joblib.load(
    MODEL_DIR / "crop_decision_tree.joblib"
)

print("All models loaded successfully!\n")

# Load training data for input validation
train = pd.read_csv(DATA_DIR / "train_data.csv")

valid_states = sorted(train["State_Name"].dropna().unique())
valid_crops = sorted(train["Crop"].dropna().unique())
valid_seasons = sorted(train["Season"].dropna().unique())

# Collect input
print("ENTER CROP DETAILS")
print("-------------------")

state = input("Enter State: ").strip()
district = input("Enter District: ").strip()
year = int(input("Enter Crop Year: "))
season = input("Enter Season: ").strip()
crop = input("Enter Crop: ").strip()
area = float(input("Enter Area (hectares): "))

# Validate input
if state not in valid_states:
    raise ValueError("Invalid state.")

if crop not in valid_crops:
    raise ValueError("Invalid crop.")

if season not in valid_seasons:
    raise ValueError("Invalid season.")

if area <= 0:
    raise ValueError("Area must be greater than zero.")

if year < 1997 or year > 2014:
    raise ValueError("Year must be between 1997 and 2014.")

# Validate state-district combination
valid_location = train[
    (train["State_Name"] == state) &
    (train["District_Name"] == district)
]

if valid_location.empty:
    raise ValueError("District does not belong to the selected state.")

# Prepare input
input_data = pd.DataFrame([{
    "State_Name": state,
    "District_Name": district,
    "Crop_Year": year,
    "Season": season,
    "Crop": crop,
    "Area": area
}])

# Generate predictions
predicted_log = regression_model.predict(input_data)[0]
predicted_production = float(np.expm1(predicted_log))

rf_prediction = str(rf_classifier.predict(input_data)[0])
dt_prediction = str(dt_classifier.predict(input_data)[0])

# Display results
print("\nCROP PRODUCTION PREDICTIONS")
print("---------------------------")

print("State:", state)
print("District:", district)
print("Crop:", crop)
print("Year:", year)
print("Season:", season)
print("Area:", area, "hectares")

print("\nPREDICTION RESULTS")
print("------------------")

print(
    "Random Forest Regressor:",
    round(predicted_production, 2),
    "recorded production units"
)

print("Random Forest Classifier:", rf_prediction)
print("Decision Tree Classifier:", dt_prediction)

# Save prediction in MongoDB
client = None

try:
    if not MONGODB_URI:
        raise ValueError("MONGODB_URI is missing in .env")

    client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=30000
    )

    client.admin.command("ping")

    db = client[DATABASE_NAME]
    collection = db["predictions"]

    prediction_record = {
        "input": {
            "State_Name": state,
            "District_Name": district,
            "Crop_Year": year,
            "Season": season,
            "Crop": crop,
            "Area": area
        },
        "predictions": {
            "Random_Forest_Regression": predicted_production,
            "Random_Forest_Classification": rf_prediction,
            "Decision_Tree_Classification": dt_prediction
        },
        "predicted_at": datetime.now(timezone.utc)
    }

    result = collection.insert_one(prediction_record)

    print("\nPrediction saved to MongoDB!")
    print("Collection: predictions")
    print("Document ID:", result.inserted_id)

except Exception as error:
    print("\nCould not save prediction to MongoDB:", error)

finally:
    if client is not None:
        client.close()

print("\nPrediction completed successfully!")

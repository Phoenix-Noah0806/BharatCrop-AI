
import os
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from pymongo import MongoClient

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("MONGODB_DATABASE", "BharatCropAI")

TRAIN_PATH = BASE_DIR / "data" / "processed" / "train_data.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "test_data.csv"

MODEL_DIR = BASE_DIR / "models"


def evaluate_models():

    client = None

    try:
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=30000
        )

        client.admin.command("ping")

        db = client[DATABASE_NAME]
        collection = db["model_metrics"]

        # Load test data
        test = pd.read_csv(TEST_PATH)

        features = [
            "State_Name",
            "District_Name",
            "Crop_Year",
            "Season",
            "Crop",
            "Area"
        ]

        X_test = test[features]

        results = []

        # ---------------------------------
        # 1. Random Forest Regression
        # ---------------------------------

        regression_model = joblib.load(
            MODEL_DIR / "crop_regression_model.joblib"
        )

        regression_predictions = regression_model.predict(X_test)

        predicted_production = np.expm1(regression_predictions)
        actual_production = test["Production"].values

        regression_results = {
            "model_name": "Random Forest Regressor",
            "model_type": "Regression",
            "r2_score": float(
                r2_score(actual_production, predicted_production)
            ),
            "mae": float(
                mean_absolute_error(
                    actual_production,
                    predicted_production
                )
            )
        }

        results.append(regression_results)

        # ---------------------------------
        # 2. Random Forest Classification
        # ---------------------------------

        y_test = test["Production_Class"]

        rf_model = joblib.load(
            MODEL_DIR / "crop_classification_model.joblib"
        )

        rf_predictions = rf_model.predict(X_test)

        rf_results = {
            "model_name": "Random Forest Classifier",
            "model_type": "Classification",
            "accuracy": float(
                accuracy_score(y_test, rf_predictions)
            ),
            "precision": float(
                precision_score(
                    y_test, rf_predictions,
                    pos_label="High"
                )
            ),
            "recall": float(
                recall_score(
                    y_test, rf_predictions,
                    pos_label="High"
                )
            ),
            "f1_score": float(
                f1_score(
                    y_test, rf_predictions,
                    pos_label="High"
                )
            )
        }

        results.append(rf_results)

        # ---------------------------------
        # 3. Decision Tree Classification
        # ---------------------------------

        dt_model = joblib.load(
            MODEL_DIR / "crop_decision_tree.joblib"
        )

        dt_predictions = dt_model.predict(X_test)

        dt_results = {
            "model_name": "Decision Tree Classifier",
            "model_type": "Classification",
            "accuracy": float(
                accuracy_score(y_test, dt_predictions)
            ),
            "precision": float(
                precision_score(
                    y_test, dt_predictions,
                    pos_label="High"
                )
            ),
            "recall": float(
                recall_score(
                    y_test, dt_predictions,
                    pos_label="High"
                )
            ),
            "f1_score": float(
                f1_score(
                    y_test, dt_predictions,
                    pos_label="High"
                )
            )
        }

        results.append(dt_results)

        # ---------------------------------
        # Save results in MongoDB
        # ---------------------------------

        for result in results:

            result["evaluated_at"] = datetime.now(timezone.utc)

            collection.update_one(
                {"model_name": result["model_name"]},
                {"$set": result},
                upsert=True
            )

            print("\n", result["model_name"])
            for key, value in result.items():
                print(f"{key}: {value}")

        print("\nAll evaluation results saved to MongoDB!")
        print("Collection:", collection.name)

    except Exception as error:
        print("Evaluation failed:", error)

    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    evaluate_models()

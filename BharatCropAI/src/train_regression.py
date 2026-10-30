
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = BASE_DIR / "data" / "processed" / "train_data.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "test_data.csv"

MODEL_PATH = BASE_DIR / "models" / "crop_regression_model.joblib"

# Load datasets
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Features used by the model
features = [
    "State_Name",
    "District_Name",
    "Crop_Year",
    "Season",
    "Crop",
    "Area"
]

target = "Log_Production"

X_train = train[features]
X_test = test[features]

y_train = train[target]
y_test = test[target]

# Identify categorical and numerical columns
categorical_features = [
    "State_Name",
    "District_Name",
    "Season",
    "Crop"
]

numerical_features = [
    "Crop_Year",
    "Area"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)

# Random Forest model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("Training Random Forest Regression model...")

model.fit(X_train, y_train)

print("Training completed!")

# Predict log production
predictions_log = model.predict(X_test)

# Convert predictions back to original production scale
predictions = np.expm1(predictions_log)
actual = np.expm1(y_test)

# Evaluate
r2 = r2_score(actual, predictions)
mae = mean_absolute_error(actual, predictions)

print("\nREGRESSION RESULTS")
print("------------------")
print("R2 Score:", round(r2, 4))
print("MAE:", round(mae, 2))

# Save model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\nModel saved at:", MODEL_PATH)

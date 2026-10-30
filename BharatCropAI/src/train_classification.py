
import pandas as pd
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = BASE_DIR / "data" / "processed" / "train_data.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "test_data.csv"

MODEL_PATH = BASE_DIR / "models" / "crop_classification_model.joblib"

# Load prepared datasets
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Features
features = [
    "State_Name",
    "District_Name",
    "Crop_Year",
    "Season",
    "Crop",
    "Area"
]

target = "Production_Class"

X_train = train[features]
X_test = test[features]

y_train = train[target]
y_test = test[target]

# Identify categorical and numerical features
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

# Random Forest Classifier
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("Training Random Forest Classification model...")

model.fit(X_train, y_train)

print("Training completed!")

# Make predictions
predictions = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    pos_label="High",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    pos_label="High",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    pos_label="High",
    zero_division=0
)

print("\nCLASSIFICATION RESULTS")
print("----------------------")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, predictions))

print("CONFUSION MATRIX")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["Low", "High"]
))

# Save model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\nModel saved at:", MODEL_PATH)

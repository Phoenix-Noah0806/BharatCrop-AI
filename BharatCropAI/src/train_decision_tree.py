
import pandas as pd
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = BASE_DIR / "data" / "processed" / "train_data.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "test_data.csv"

MODEL_PATH = BASE_DIR / "models" / "crop_decision_tree.joblib"

# Load data
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Input features
features = [
    "State_Name",
    "District_Name",
    "Crop_Year",
    "Season",
    "Crop",
    "Area"
]

# Target column
target = "Production_Class"

X_train = train[features]
X_test = test[features]

y_train = train[target]
y_test = test[target]

# Categorical features
categorical_features = [
    "State_Name",
    "District_Name",
    "Season",
    "Crop"
]

# Encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# Decision Tree Classifier
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

# Train model
print("Training Decision Tree Classifier...")

model.fit(X_train, y_train)

print("Training completed!")

# Predictions
predictions = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, predictions)

recall = recall_score(
    y_test,
    predictions,
    pos_label="High"
)

precision = precision_score(
    y_test,
    predictions,
    pos_label="High"
)

f1 = f1_score(
    y_test,
    predictions,
    pos_label="High"
)

print("\nDECISION TREE RESULTS")
print("---------------------")
print("Accuracy:", round(accuracy, 4))
print("Recall:", round(recall, 4))
print("Precision:", round(precision, 4))
print("F1 Score:", round(f1, 4))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, predictions))

print("CONFUSION MATRIX")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["Low", "High"]
))

# Save trained model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\nModel saved at:", MODEL_PATH)

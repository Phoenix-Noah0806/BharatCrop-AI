
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "processed" / "crop_production_cleaned.csv"
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train_data.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "test_data.csv"

# Load cleaned dataset
df = pd.read_csv(INPUT_PATH)

print("Original dataset:", df.shape)

# Remove records without a known production value
df = df.dropna(subset=["Production"]).copy()

# Exclude 2015 because it has very limited records
df = df[df["Crop_Year"] < 2015].copy()

# Remove negative production values, if any
df = df[df["Production"] >= 0].copy()

# Chronological split
train = df[df["Crop_Year"] <= 2012].copy()
test = df[df["Crop_Year"].isin([2013, 2014])].copy()

# Calculate median production for each crop using training data only
crop_medians = train.groupby("Crop")["Production"].median()

# Create classification labels
train["Production_Class"] = np.where(
    train["Production"] >= train["Crop"].map(crop_medians),
    "High",
    "Low"
)

test["Production_Class"] = np.where(
    test["Production"] >= test["Crop"].map(crop_medians),
    "High",
    "Low"
)

# Remove test records for crops not present in training data
test = test.dropna(subset=["Production_Class"]).copy()

# Log transformation for regression target
train["Log_Production"] = np.log1p(train["Production"])
test["Log_Production"] = np.log1p(test["Production"])

# Save prepared datasets
TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

train.to_csv(TRAIN_PATH, index=False)
test.to_csv(TEST_PATH, index=False)

print("\nDATA PREPARATION COMPLETE")
print("-------------------------")

print("Training records:", len(train))
print("Testing records:", len(test))

print("Training years:", sorted(train["Crop_Year"].unique()))
print("Testing years:", sorted(test["Crop_Year"].unique()))

print("Training crops:", train["Crop"].nunique())
print("Testing crops:", test["Crop"].nunique())

print("\nClassification distribution in training data:")
print(train["Production_Class"].value_counts())

print("\nClassification distribution in testing data:")
print(test["Production_Class"].value_counts())

print("\nMissing values in training data:")
print(train.isnull().sum())

print("\nSaved files:")
print(TRAIN_PATH)
print(TEST_PATH)


import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

input_path = BASE_DIR / "data" / "raw" / "crop_production.csv"
output_path = BASE_DIR / "data" / "processed" / "crop_production_cleaned.csv"

# 1. Load the original dataset
df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)

# 2. Standardize text columns
text_columns = [
    "State_Name",
    "District_Name",
    "Season",
    "Crop"
]

for col in text_columns:
    df[col] = df[col].astype("string").str.strip()

# 3. Convert numeric columns
numeric_columns = [
    "Crop_Year",
    "Area",
    "Production"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 4. Report missing values before cleaning
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# 5. Remove rows missing essential input fields
required_columns = [
    "State_Name",
    "District_Name",
    "Crop_Year",
    "Season",
    "Crop",
    "Area"
]

before = len(df)
df = df.dropna(subset=required_columns).copy()

print("\nRows removed due to missing input fields:", before - len(df))

# 6. Remove records with invalid area
before = len(df)
df = df[df["Area"] > 0].copy()

print("Rows removed due to invalid area:", before - len(df))

# 7. Remove exact duplicate records only
before = len(df)
df = df.drop_duplicates().copy()

print("Exact duplicate rows removed:", before - len(df))

# 8. Keep missing Production values in the cleaned dataset.
# They may still be useful for non-supervised analysis.
# We will exclude them from supervised model training later.

# 9. Save cleaned data
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

# 10. Print final summary
print("\nCLEANING COMPLETE")
print("Final dataset shape:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nCrop types:", df["Crop"].nunique())
print("States/territories:", df["State_Name"].nunique())
print("Districts:", df["District_Name"].nunique())
print("Years:", df["Crop_Year"].nunique())

print("\nSaved cleaned dataset to:")
print(output_path)

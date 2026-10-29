import pandas as pd

file_path = "data/raw/crop_production.csv"

df = pd.read_csv(file_path)

print("DATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nFIRST FIVE ROWS")
print(df.head())

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nEXACT DUPLICATE ROWS")
print(df.duplicated().sum())

print("\nNUMERICAL SUMMARY")
print(df.describe())

print("\nUNIQUE VALUES PER COLUMN")
print(df.nunique())

print("\nRECORDS PER YEAR")
if "Crop_Year" in df.columns:
    print(df["Crop_Year"].value_counts().sort_index())
    
print("\n" + "=" * 50)
print("11. MISSING PRODUCTION ANALYSIS")
print("=" * 50)

missing_df = df[df["Production"].isna()]

print("Rows with missing Production:", len(missing_df))

print("\nMissing Production by year:")
print(missing_df.groupby("Crop_Year").size().sort_index())

print("\nMissing Production by crop:")
print(
    missing_df.groupby("Crop")
    .size()
    .sort_values(ascending=False)
    .head(15)
)

print("\n" + "=" * 50)
print("12. ZERO AND NEGATIVE VALUES")
print("=" * 50)

print("Zero production:", (df["Production"] == 0).sum())
print("Negative production:", (df["Production"] < 0).sum())
print("Zero or negative area:", (df["Area"] <= 0).sum())

print("\n" + "=" * 50)
print("13. RECORDS IN 2015")
print("=" * 50)

print("Records in 2015:", (df["Crop_Year"] == 2015).sum())

print("\nState-wise records in 2015:")
print(
    df.loc[df["Crop_Year"] == 2015]
    .groupby("State_Name")
    .size()
    .sort_values(ascending=False)
)

print("\n" + "=" * 50)
print("14. LARGEST PRODUCTION RECORDS")
print("=" * 50)

print(
    df.nlargest(10, "Production")[
        [
            "State_Name",
            "District_Name",
            "Crop_Year",
            "Season",
            "Crop",
            "Area",
            "Production"
        ]
    ].to_string(index=False)
)

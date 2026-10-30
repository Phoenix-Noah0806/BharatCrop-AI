
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("MONGODB_DATABASE", "BharatCropAI")

CSV_PATH = BASE_DIR / "data" / "processed" / "crop_production_cleaned.csv"


def import_crop_data():
    client = None

    try:
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=120000,
            retryWrites=True
        )

        client.admin.command("ping")

        db = client[DATABASE_NAME]
        collection = db["crop_records"]

        print("Connected to MongoDB!")

        df = pd.read_csv(CSV_PATH)

        # Convert missing values to None
        df = df.astype(object).where(pd.notna(df), None)

        total_records = len(df)
        batch_size = 1000

        # Resume after records already present in MongoDB
        existing_count = collection.count_documents({})
        print("Existing records:", existing_count)

        if existing_count > total_records:
            raise ValueError(
                "MongoDB contains more records than the CSV. "
                "Check the collection before importing."
            )

        start = existing_count

        while start < total_records:
            batch_df = df.iloc[start:start + batch_size]
            records = batch_df.to_dict(orient="records")

            collection.insert_many(records, ordered=True)

            start += len(records)

            print(f"Inserted {start} / {total_records} records")

        print("\nIMPORT SUCCESSFUL!")
        print("Total records:", collection.count_documents({}))
        print("Unique crops:", df["Crop"].nunique())
        print("Collection:", collection.name)

    except (PyMongoError, ValueError, FileNotFoundError) as error:
        print("\nImport failed:", error)

    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    import_crop_data()


import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

client = MongoClient(os.getenv("MONGODB_URI"))

db = client[os.getenv("MONGODB_DATABASE", "BharatCropAI")]

collection = db["crop_records"]

print("DATABASE VERIFICATION")
print("---------------------")

# Total records
print("Total records:", collection.count_documents({}))

# Number of unique crops
print(
    "Unique crops:",
    len(collection.distinct("Crop"))
)

# Number of unique states
print(
    "Unique states:",
    len(collection.distinct("State_Name"))
)

# Missing production values
print(
    "Missing production:",
    collection.count_documents({"Production": None})
)

# Display one sample record
print("\nSample record:")
print(collection.find_one({}, {"_id": 0}))

client.close()

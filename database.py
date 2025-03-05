from pymongo import MongoClient
import os
from dotenv import load_dotenv 

# MongoDB Secure Connection
MONGO_URI = os.getenv("MONGO_URI")  # Load from environment variable
client = MongoClient(MONGO_URI)
db = client["book_recommender"]  # Database name

# Collection references
contact_collection = db["contacts"]

def insert_contact(name, email, message):
    """Insert contact details into MongoDB with error handling."""
    try:
        contact_collection.insert_one({
            "name": name,
            "email": email,
            "message": message
        })
        print("✅ Contact saved successfully!")
    except Exception as e:
        print(f"❌ Error saving contact: {e}")

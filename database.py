import pymysql
import os
from dotenv import load_dotenv
from logger import get_logger  # Import the logger

# Load environment variables
load_dotenv()

# Initialize logger
logger = get_logger("Database")

# Database credentials from .env
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = int(os.getenv("DB_PORT", 3306))

def get_connection():
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            cursorclass=pymysql.cursors.DictCursor
        )
        logger.info("Database connection successful.")
        return connection
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

def insert_contact(name, email, message):
    connection = get_connection()
    if connection:
        try:
            with connection.cursor() as cursor:
                sql = "INSERT INTO contacts (name, email, message) VALUES (%s, %s, %s)"
                cursor.execute(sql, (name, email, message))
                connection.commit()
                logger.info(f"Contact inserted: {name}, {email}")
        except Exception as e:
            logger.error(f"Error inserting contact: {e}")
        finally:
            connection.close()

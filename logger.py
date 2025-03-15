import logging
from datetime import datetime

# Custom Formatter for Date and Time
class CustomFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created).strftime("%d-%m-%Y %H:%M:%S")  # e.g., 14-03-2025 15:33:20

# Function to Get Logger
def get_logger(name):
    logger = logging.getLogger(name)
    
    # Ensure the logger only adds a handler once
    if not logger.hasHandlers():
        handler = logging.FileHandler("app.log", mode="a")  # Append logs, do not overwrite
        handler.setFormatter(CustomFormatter("%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger

# Append session start log
session_start_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
with open("app.log", "a") as f:  # Always append, never overwrite
    f.write("\n" + "=" * 80 + f"\n SESSION STARTED: {session_start_time} \n" + "=" * 80 + "\n\n")

# Function to log session end
def log_session_end():
    session_end_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    with open("app.log", "a") as f:  # Append, never overwrite
        f.write(f"\nSESSION ENDED: {session_end_time}\n" + "=" * 80 + "\n\n")

# Example usage
if __name__ == "__main__":
    logger = get_logger("Main")
    logger.info("This is a test log message.")  # Example log message

    # Log session end when script finishes
    log_session_end()

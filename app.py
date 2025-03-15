from flask import Flask, render_template, request
import pickle as pkl
import pandas as pd
import numpy as np
from database import insert_contact
from sklearn.metrics.pairwise import cosine_similarity
import os
from logger import get_logger  # Import the logger

# Initialize logger
logger = get_logger("AppServer")

# Load the pre-trained model
try:
    logger.info("Loading the pre-trained model...")
    model_data = pkl.load(open("model.pkl", "rb"))
    vectorizer = model_data["vectorizer"]
    feature_matrix = model_data["feature_matrix"]
    df_final = model_data["df_final"]

    # Reset index to avoid indexing issues
    df_final.reset_index(drop=True, inplace=True)

    # Compute cosine similarity dynamically
    cosine_sim = cosine_similarity(feature_matrix, feature_matrix)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise e  # Stop execution if model loading fails

app = Flask(__name__)

@app.route('/')
def index():
    logger.info("Rendering home page.")
    return render_template('index.html',
                           book_name=list(df_final['Title'].values[:500]),
                           author=list(df_final['Author'].values[:500]),
                           image=list(df_final['Image URL'].values[:500]),
                           category=list(df_final['Category'].values[:500]) if 'Category' in df_final.columns else ["Unknown"] * len(df_final))

@app.route('/recommend')
def recommend_ui():
    logger.info("Rendering recommendation page.")
    return render_template('recommend.html')

@app.route('/recommend_books', methods=['POST'])
def recommend():
    try:
        user_input = request.form.get('user_input', '').strip().lower()
        logger.info(f"Received recommendation request for book: {user_input}")

        # Check if book exists
        matching_indices = df_final[df_final['Title'].str.lower().str.strip() == user_input].index.tolist()

        if not matching_indices:
            logger.warning(f"Book '{user_input}' not found in the database.")
            return render_template('recommend.html', error="Book not found in the database. Please try another.")

        index = matching_indices[0]
        similar_books = list(enumerate(cosine_sim[index]))
        sorted_books = sorted(similar_books, key=lambda x: x[1], reverse=True)[:6]
        top_books = [i[0] for i in sorted_books]
        recommendations = df_final.loc[top_books, ['Title', 'Author', 'Category', 'Image URL']].values.tolist()

        logger.info(f"Recommended books for '{user_input}': {recommendations}")
        return render_template('recommend.html', data=recommendations)

    except Exception as e:
        logger.error(f"Error in recommendation: {e}")
        return render_template('recommend.html', error=f"An error occurred: {str(e)}")

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        if not name or not email or not message:
            logger.warning("Contact form submission failed: Missing fields.")
            return render_template('contact.html', error="All fields are required!")

        try:
            insert_contact(name, email, message)
            logger.info(f"Contact form submitted: Name={name}, Email={email}, Message={message}")
            return render_template('contact.html', success="Your message has been sent successfully!")
        except Exception as e:
            logger.error(f"Database error: {e}")
            return render_template('contact.html', error=f"Database error: {e}")

    logger.info("Rendering contact page.")
    return render_template('contact.html')

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5001))
    logger.info(f"Starting Flask app on port {port}...")
    app.run(debug=True, host="0.0.0.0", port=port)

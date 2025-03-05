from flask import Flask, render_template, request
import pickle as pkl
import pandas as pd
import numpy as np
from database import insert_contact
from sklearn.metrics.pairwise import cosine_similarity
import sklearn

# Load the saved model components
model_data = pkl.load(open("model.pkl", "rb"))

# Extract components
vectorizer = model_data["vectorizer"]
feature_matrix = model_data["feature_matrix"]
df_final = model_data["df_final"]

# Compute cosine similarity dynamically
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

# Flask App Initialization
app = Flask(__name__)

@app.route('/')
def index():
    """Render homepage with popular books"""
    return render_template(
        'index.html',
        book_name=list(df_final['Title'].values[:10]),  # Display top 10 books
        author=list(df_final['Author'].values[:10]),
        image=list(df_final['Image URL'].values[:10]),
        category=list(df_final['Category'].values[:10])
    )

@app.route('/recommend')
def recommend_ui():
    """Render recommendation page UI"""
    return render_template('recommend.html')

@app.route('/recommend_books', methods=['POST'])
def recommend():
    """Handle book recommendation requests"""
    try:
        user_input = request.form.get('user_input')
        if user_input not in df_final['Title'].values:
            return render_template('recommend.html', error="Book not found. Please try another.")

        # Find book index
        index = df_final[df_final['Title'] == user_input].index[0]

        # Get similarity scores
        similar_books = list(enumerate(cosine_sim[index]))

        # Sort by similarity
        sorted_books = sorted(similar_books, key=lambda x: x[1], reverse=True)[:6]

        # Extract recommended books
        data = []
        for i in sorted_books:
            try:
                item = []
                temp_df = df_final.iloc[[i[0]]]
                item.append(temp_df['Title'].values[0])
                item.append(temp_df['Author'].values[0])
                item.append(temp_df['Image URL'].values[0])
                item.append(temp_df['Category'].values[0])
                data.append(item)
            except Exception as e:
                print(f"Error processing similar items: {e}")

        return render_template('recommend.html', data=data)

    except Exception as e:
        print(f"Error in recommend_books route: {e}")
        return "An error occurred while processing your recommendation request. Please try again later."

@app.route("/contact", methods=["GET", "POST"])
def contact():
    """Handle contact form submissions"""
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")

        # Insert the data into the database
        insert_contact(name, email, message)

        # Show success message
        return render_template("contact.html", success="Your message has been sent successfully!")

    return render_template("contact.html")

if __name__ == '__main__':
    app.run(debug=True, port=5001)

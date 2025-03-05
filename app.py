from flask import Flask, render_template, request
import pickle as pkl
import pandas as pd
import numpy as np
from database import insert_contact
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained model
model_data = pkl.load(open("model.pkl", "rb"))

# Extract components
vectorizer = model_data["vectorizer"]
feature_matrix = model_data["feature_matrix"]
df_final = model_data["df_final"]

# Reset index to avoid indexing issues
df_final.reset_index(drop=True, inplace=True)

# Compute cosine similarity dynamically
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
                           book_name=list(df_final['Title'].values[:100]),
                           author=list(df_final['Author'].values[:100]),
                           image=list(df_final['Image URL'].values[:100]),
                           category=list(df_final['Category'].values[:100]) if 'Category' in df_final.columns else ["Unknown"] * len(df_final))

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books', methods=['POST'])
def recommend():
    try:
        # Get user input and clean it
        user_input = request.form.get('user_input', '').strip().lower()

        # Check if book exists in the dataset
        matching_indices = df_final[df_final['Title'].str.lower().str.strip() == user_input].index.tolist()
        
        if not matching_indices:
            return render_template('recommend.html', error="Book not found in the database. Please try another.")

        index = matching_indices[0]  # Get the first match
        
        # Get similarity scores
        similar_books = list(enumerate(cosine_sim[index]))

        # Sort books by similarity score (excluding the input book itself)
        sorted_books = sorted(similar_books, key=lambda x: x[1], reverse=True)[:6]  

        # Extract recommended book indices
        top_books = [i[0] for i in sorted_books]

        # Select the recommended books from the dataset
        recommendations = df_final.loc[top_books, ['Title', 'Author', 'Category', 'Image URL']].values.tolist()

        return render_template('recommend.html', data=recommendations)

    except Exception as e:
        return render_template('recommend.html', error=f"An error occurred: {str(e)}")


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        insert_contact(name, email, message)
        return render_template('contact.html', success="Your message has been sent successfully!")
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True,host='127.0.0.1',port=5001)

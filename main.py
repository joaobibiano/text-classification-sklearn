import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Path to save the model
model_path = 'expense_categorizer_model.joblib'

# Check if the model already exists
if os.path.exists(model_path):
    # Load the model from disk
    model = joblib.load(model_path)
    print("Model loaded from disk.")
else:
    # Read data from CSV file
    data = pd.read_csv('training.csv')

    # Display the first few rows of the dataset to understand its structure
    print(data.head())

    # Ensure consistent data preprocessing
    data['Description'] = data['Description'].str.lower()  # Convert to lowercase for consistency

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['Description'], data['Category'], test_size=0.2, random_state=42)

    # Build the machine learning pipeline
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    # Train the model
    model.fit(X_train, y_train)

    # Save the model to disk
    joblib.dump(model, model_path)
    print("Model trained and saved to disk.")

@app.route('/predict', methods=['GET'])
def predict():
    new_expenses = request.args.getlist('expenses')
    if not new_expenses:
        return jsonify({"error": "No expenses provided"}), 400
    
    predictions = model.predict(new_expenses)
    probabilities = model.predict_proba(new_expenses)

    results = []
    for expense, prediction, probability in zip(new_expenses, predictions, probabilities):
        results.append({
            "expense": expense,
            "predicted_category": prediction,
            "confidence": max(probability)
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=3434, host='0.0.0.0')

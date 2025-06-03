from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model pipeline
pipeline = joblib.load("pipeline.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Athlete Anxiety Clustering API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert data to DataFrame
        df = pd.DataFrame([data])

        # Ensure column order matches the training data
        column_order = [
    'Age', 'Years_of_Excersie_Experince', 'Weekly_Anxiety', 
    'Daily_App_Usage', 'Comfort_in_Social_Situations', 
    'Competition_Level', 'anxiety_level',
    'Gender', 'Current_Status', 'Feeling_Anxious', 
    'Preferred_Anxiety_Treatment', 'Handling_Anxiety_Situations', 
    'General_Mood', 'Preferred_Content', 'Online_Interaction_Over_Offline'
    ]

        df = df[column_order]

        # Make prediction
        cluster = pipeline.predict(df)

        # Return result
        return jsonify({"cluster": int(cluster[0])})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)

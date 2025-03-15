import os
import pickle
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Load Model and Encoders
MODEL_PATH = "models/house_price_model.pkl"
ENCODER_PATH = "models/label_encoders.pkl"
DATA_PATH = "data/Bengaluru_House_Data.csv"

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError("Model or encoders file not found.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(ENCODER_PATH, "rb") as f:
    encoders = pickle.load(f)

# Load dataset and extract unique locations
df = pd.read_csv(DATA_PATH)
df["location"] = df["location"].astype(str).str.strip()
unique_locations = sorted(df["location"].dropna().unique().tolist())

# Flask App
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", locations=unique_locations)  # Ensure list is passed correctly

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        location = data.get("location")
        sqft = data.get("sqft")
        bath = data.get("bath")
        bhk = data.get("bhk")

        # Check for missing values
        if not all([location, sqft, bath, bhk]):
            return jsonify({"error": "Missing required fields"}), 400

        # Convert numerical inputs safely
        try:
            sqft = float(sqft)
            bath = int(bath)
            bhk = int(bhk)
        except ValueError:
            return jsonify({"error": "Invalid numeric values provided"}), 400

        # Ensure location is formatted correctly
        location = location.strip()

        # Encode categorical feature safely
        if location in encoders["location"].classes_:
            location_encoded = encoders["location"].transform([location])[0]
        else:
            # Instead of error, assign to "unknown" category (use median location encoding)
            location_encoded = np.median(encoders["location"].transform(encoders["location"].classes_))

        # Additional features to match model training
        size_per_room = sqft / bhk if bhk > 0 else sqft  # Avoid division by zero
        is_studio = 1 if bhk == 1 else 0  # Binary feature for studio apartments

        # Prepare feature array (must match training order)
        features = np.array([location_encoded, sqft, bath, bhk, size_per_room, is_studio]).reshape(1, -1)

        # Predict
        prediction = model.predict(features)[0]
        
        adjusted_price = prediction * 30
        return jsonify({"predicted_price": round(adjusted_price, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_locations', methods=['GET'])
def get_locations():
    return jsonify({"locations": unique_locations})

if __name__ == "__main__":
    app.run(debug=True)

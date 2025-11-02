# app.py
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

#Config
MODEL_PATH = os.getenv("MODEL_PATH", "model/california_housing.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the California House Price Prediction API",
        "note": "Use POST /predict with all housing feature values in JSON."
    })

@app.route('/features', methods=['GET'])
def features():
    return jsonify({
        "required_features": [
            "MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"
        ],
        "example_input": {
            "MedInc": 4.5,
            "HouseAge": 30,
            "AveRooms": 5.5,
            "AveBedrms": 1.1,
            "Population": 1200,
            "AveOccup": 3,
            "Latitude": 37.77,
            "Longitude": -122.42
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        required = [
            "MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"
        ]
        if not all(feature in data for feature in required):
            return jsonify({"error": f"Missing one or more features: {required}"}), 400

        input_df = pd.DataFrame([data])
        predicted_price = model.predict(input_df)[0]

        return jsonify({
            "predicted_price_usd": round(predicted_price, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

    
    
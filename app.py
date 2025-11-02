from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "model/california_housing.pkl")

# Initialize Flask app
app = Flask(__name__)

# Load model at startup
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    model = None

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
    if model is None:
        return jsonify({"error": "Model not loaded on server."}), 500

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

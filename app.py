# py app.py or python app.py to run app on Python backend

from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
import uuid
import shutil
from backend.predictions.ensemble import train_ensemble, test_ensemble, predict_ensemble
from functools import lru_cache
from utils.s3_utils import upload_image_to_s3
from utils.dynamo_utils import save_forecast_to_dynamodb

app = Flask(__name__)

def warm_up_models():
    train_ensemble()
    print("Successfully trained all models.")
    metrics = test_ensemble()
    print("Successfully tested all models.")
    print("Ensemble method metrics:" + str(metrics))

# Define base directory and paths to different temperature datasets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = {
    "country": os.path.join(BASE_DIR, "datasets", "GlobalLandTemperaturesByCountry.csv"),
    "state": os.path.join(BASE_DIR, "datasets", "GlobalLandTemperaturesByState.csv"),
    "city": os.path.join(BASE_DIR, "datasets", "GlobalLandTemperaturesByCity.csv")
}

S3_BUCKET = "climate-forecast-results"  # ← Replace with your actual bucket
DYNAMO_TABLE = "ClimateForecasts"

# Cache location options to avoid repeated disk reads and improve performance
@lru_cache(maxsize=None)
def load_location_options(location_type):
    """
    Load and return a sorted list of unique locations based on the location type (country, state, city).
    Cached for performance using LRU cache.
    """
    path = DATASETS.get(location_type)
    if not path or not os.path.exists(path):
        return []

    column = "Country" if location_type == "country" else "State" if location_type == "state" else "City"
    df = pd.read_csv(path, usecols=[column])
    return sorted(df[column].dropna().unique())

@app.route("/")
def home():
    return ("Welcome to the Climate Change Forecast Prediction API.<br>"
            "- /train: Train all models.<br>"
            "- /test: Test all models.<br>"
            "- /predict: Make predictions.<br>"
    )

@app.route("/train", methods=["GET"])
def train_endpoint():
    train_ensemble()
    return "Successfully trained all models."

@app.route("/test", methods=["GET"])
def test_endpoint():
    results = test_ensemble()
    return jsonify(["Successfully tested all models."] + results)

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    payload = request.get_json()

    try:
        ds = payload["dataset"]
        yr = payload["year"]
        mnth = payload["month"]
        loc = payload.get("location")

        if ds in ("country", "city", "state"):
            result = predict_ensemble(ds, yr, mnth, location=loc)
        else:
            raise ValueError(f"Unknown dataset: {ds}")
        
        # Return JSON + status code
        return jsonify(result), 200

    except ValueError as e:
        # bad input, missing location, etc
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # something else went wrong
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    
    # Perform both GET calls to train and test all models on startup but only once.
    warm_up_models()

    # now start accepting requests
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
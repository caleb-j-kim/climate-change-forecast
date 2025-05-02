from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
import traceback
from functools import lru_cache

from backend.predictions.ensemble import train_ensemble, test_ensemble, predict_ensemble
from utils.s3_utils import upload_image_to_s3
from utils.dynamo_utils import save_forecast_to_dynamodb

app = Flask(__name__)


# === Config ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = {
    "country": os.path.join(BASE_DIR, "datasets", "GlobalLandTemperaturesByCountry.csv"),
    "state":   os.path.join(BASE_DIR, "datasets", "GlobalLandTemperaturesByState.csv"),
    "city":    os.path.join(BASE_DIR, "datasets", "GlobalLandTemperaturesByCity.csv"),
}
S3_BUCKET = "climate-forecast-results"
DYNAMO_TABLE = "ClimateForecasts"

# === Model warm-up ===
def warm_up_models():
    print("Training all models…")
    train_ensemble()
    print("All models trained.")

    print("Testing ensemble…")
    metrics = test_ensemble()
    print("Ensemble test metrics:", metrics)

warm_up_models()


# === Location utilities ===
@lru_cache(maxsize=None)
def load_location_options(location_type):
    path = DATASETS.get(location_type)
    if not path or not os.path.exists(path):
        return []
    col = {"country": "Country", "state": "State", "city": "City"}[location_type]
    df = pd.read_csv(path, usecols=[col])
    return sorted(df[col].dropna().unique())

# === Routes ===
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/locations")
def locations():
    loc_type = request.args.get("type")
    return jsonify(load_location_options(loc_type))

@app.route("/train", methods=["GET"])
def train_endpoint():
    train_ensemble()
    return "Successfully trained all models."

@app.route("/test", methods=["GET"])
def test_endpoint():
    results = test_ensemble()
    return jsonify(results)

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    payload = request.get_json()
    try:
        ds   = payload["dataset"]
        yr   = payload["year"]
        mnth = payload["month"]
        loc  = payload.get("location")

        if ds not in DATASETS:
            raise ValueError(f"Unknown dataset: {ds}")

        result = predict_ensemble(ds, yr, mnth, location=loc)

        # optionally store to DynamoDB
        #save_forecast_to_dynamodb({
        #    "dataset": ds,
        #    "year": yr,
        #    "month": mnth,
        #    "location": loc,
        #    **result
        #})

        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# === Startup ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

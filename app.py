# py app.py or python app.py to run app on Python backend
from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
import traceback
from functools import lru_cache
from openai import OpenAI
from dotenv import load_dotenv

from backend.predictions.ensemble import train_ensemble, test_ensemble, predict_ensemble
from backend.apis.tomorrow_io import fetch_daily_timeline, fetch_random_city_forecast, fetch_random_country_forecast, fetch_random_state_forecast
from functools import lru_cache
from utils.s3_utils import upload_image_to_s3
from utils.dynamo_utils import save_forecast_to_dynamodb

# === Init ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__)

# === Config ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = {
    "country": os.path.join(BASE_DIR, "datasets", "GlobalLandTemperaturesByCountry.csv"),
    "state":   os.path.join(BASE_DIR, "datasets", "GlobalLandTemperaturesByState.csv"),
    "city":    os.path.join(BASE_DIR, "datasets", "GlobalLandTemperaturesByCity.csv"),
}

# === Model warm-up ===
def warm_up_models():
    print("Training all models…")
    train_ensemble()
    print("All models trained.")
    print("Testing ensemble…")
    metrics = test_ensemble()
    print("Ensemble test metrics:", metrics)

warm_up_models()

# === LLM Summarization ===
def summarize_prediction(data):
    prompt = f"""
    Summarize this temperature forecast for a general audience:

    Location: {data['location']}
    Year: {data['year']}
    Month: {data['month']}
    Linear Regression: {data['lr_prediction']:.2f} °C
    Random Forest: {data['rf_prediction']:.2f} °C
    Ensemble: {data['ensemble_prediction']:.2f} °C

    Respond with a short 2-3 sentence friendly explanation.
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "You are a knowledgeable and friendly climate scientist who explains temperature forecasts in clear, simple language for a general audience. Your goal is to make predictions easy to understand, avoiding jargon while maintaining accuracy. Keep the summary concise, approachable, and informative."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI summarization error:", e)
        return "Summary unavailable due to an AI error."

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
    return jsonify(["Successfully tested all models."] + results)

@app.route("/weather/timeline", methods=["GET"])
def weather_timeline_endpoint():
    try:
        timeline = fetch_daily_timeline()
        intervals = timeline["data"]["timelines"][0]["intervals"]
        return jsonify(intervals), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/weather/forecast/random_country", methods=["GET"])
def random_country_forecast():
    try:
        country, full = fetch_random_country_forecast()
        intervals = full["data"]["timelines"][0]["intervals"]
        return jsonify({"type":"country", "location": country, "forecast": intervals}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/weather/forecast/random_city", methods=["GET"])
def random_city_forecast():
    try:
        city, full = fetch_random_city_forecast()
        intervals = full["data"]["timelines"][0]["intervals"]
        return jsonify({"type":"city", "location": city, "forecast": intervals}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/weather/forecast/random_state", methods=["GET"])
def random_state_forecast():
    try:
        state, full = fetch_random_state_forecast()
        intervals = full["data"]["timelines"][0]["intervals"]
        return jsonify({"type":"state", "location": state, "forecast": intervals}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
        summary = summarize_prediction(result)
        result["summary"] = summary

        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@lru_cache(maxsize=None)
def load_location_options(location_type):
    path = DATASETS.get(location_type)
    if not path or not os.path.exists(path):
        return []
    col = {"country": "Country", "state": "State", "city": "City"}[location_type]
    df = pd.read_csv(path, usecols=[col])
    return sorted(df[col].dropna().unique())

if __name__ == "__main__":
    
    # Perform both GET calls to train and test all models on startup but only once.
    warm_up_models()

    # now start accepting requests
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

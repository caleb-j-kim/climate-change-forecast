# py app.py or python app.py to run app on Python backend
import json
from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import pandas as pd
import traceback
from functools import lru_cache
from openai import OpenAI
from dotenv import load_dotenv
from flask_cors import CORS

from backend.predictions.ensemble import train_ensemble, test_ensemble, predict_ensemble
from backend.apis.tomorrow_io import fetch_daily_timeline, fetch_random_city_forecast, fetch_random_country_forecast, fetch_random_state_forecast
from functools import lru_cache
from utils.s3_utils import upload_image_to_s3
from utils.dynamo_utils import save_forecast_to_dynamodb

# === Init ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__)

CORS(app)

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
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    if path != "" and os.path.exists(os.path.join("frontend", "build", path)):
        return send_from_directory(os.path.join("frontend", "build"), path)
    else:
        return send_from_directory(os.path.join("frontend", "build"), "index.html")

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
        ds = payload["dataset"]
        yr = payload["year"]
        mnth = payload["month"]
        loc = payload.get("location")
        
        print(f"Debug - received request: dataset={ds}, year={yr}, month={mnth}, location={loc}")
        
        if ds not in DATASETS:
            raise ValueError(f"Unknown dataset: {ds}")
        
        # Critical fix: Make sure country locations are always strings
        if ds == "country":
            # If it's an object with a country field, extract that
            if isinstance(loc, dict) and "country" in loc:
                loc = loc["country"]
                print(f"Debug - converted country object to string: {loc}")
            # If it's a string that might be a JSON, try to parse it
            elif isinstance(loc, str) and loc.startswith("{") and loc.endswith("}"):
                try:
                    parsed = json.loads(loc)
                    if isinstance(parsed, dict) and "country" in parsed:
                        loc = parsed["country"]
                        print(f"Debug - parsed JSON string to get country: {loc}")
                except Exception as e:
                    print(f"Debug - error parsing JSON string: {e}")
                    # Keep as is if parsing fails
            
            # Final check - ensure it's a string
            if not isinstance(loc, str):
                raise ValueError(f"Country location must be a string, got {type(loc)}: {loc}")
            
        # Validation for city and state
        elif ds == "city":
            if not isinstance(loc, dict) or "city" not in loc or "country" not in loc:
                raise ValueError("City location must be a dictionary with 'city' and 'country' keys.")
                
        elif ds == "state":
            if not isinstance(loc, dict) or "state" not in loc or "country" not in loc:
                raise ValueError("State location must be a dictionary with 'state' and 'country' keys.")
        
        # Now we're sure loc has the right format for each dataset type
        result = predict_ensemble(ds, yr, mnth, location=loc)
        
        # Extra safety - double check the location in the result
        if ds == "country" and not isinstance(result["location"], str):
            print(f"Debug - warning: predict_ensemble returned non-string location for country: {result['location']}")
            
            # Force it to be a string
            if isinstance(result["location"], dict):
                if "country" in result["location"]:
                    result["location"] = result["location"]["country"]
                else:
                    # Last resort, convert to string representation
                    result["location"] = str(result["location"])
        
        # Generate the summary
        summary = summarize_prediction(result)
        result["summary"] = summary
        
        print(f"Debug - returning result with location: {result['location']}")
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

    df = pd.read_csv(path)
    if location_type == "city":
        return df[["City", "Country"]].dropna().drop_duplicates().rename(columns={"City": "city", "Country": "country"}).to_dict(orient="records")
    elif location_type == "state":
        return df[["State", "Country"]].dropna().drop_duplicates().rename(columns={"State": "state", "Country": "country"}).to_dict(orient="records")
    elif location_type == "country":
        return sorted(df["Country"].dropna().unique())

if __name__ == "__main__":
    
    # Perform both GET calls to train and test all models on startup but only once.
    warm_up_models()

    # now start accepting requests
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

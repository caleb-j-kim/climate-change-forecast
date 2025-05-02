from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
import traceback
from functools import lru_cache
from openai import OpenAI
from dotenv import load_dotenv

from backend.predictions.ensemble import train_ensemble, test_ensemble, predict_ensemble

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
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

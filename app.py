from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
from models.linear_regression import train_temperature_model
import shutil
from functools import lru_cache

# Initialize Flask app
app = Flask(__name__)

# Base directory and dataset paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = {
    "country": os.path.join(BASE_DIR, "datasets", "GlobalLandTemperaturesByCountry.csv"),
    "state": os.path.join(BASE_DIR, "datasets", "GlobalLandTemperaturesByState.csv"),
    "city": os.path.join(BASE_DIR, "datasets", "GlobalLandTemperaturesByCity.csv")
}

# Caching function to improve load performance
@lru_cache(maxsize=None)
def load_location_options(location_type):
    path = DATASETS.get(location_type)
    if not path or not os.path.exists(path):
        return []
    df = pd.read_csv(path, usecols=["Country"] if location_type == "country" else ["State"] if location_type == "state" else ["City"])
    column = "Country" if location_type == "country" else "State" if location_type == "state" else "City"
    return sorted(df[column].dropna().unique())

@app.route("/")
def index():
    countries = load_location_options("country")
    return render_template("index.html", countries=countries)

@app.route("/locations")
def get_locations():
    location_type = request.args.get("type")
    options = load_location_options(location_type)
    return jsonify(options)

@app.route("/year-range")
def year_range():
    location_type = request.args.get("type")
    location = request.args.get("location")
    path = DATASETS.get(location_type)

    try:
        df = pd.read_csv(path, usecols=["dt", location_type.capitalize()])
        df = df[df[location_type.capitalize()] == location]
        df["dt"] = pd.to_datetime(df["dt"])
        years = df["dt"].dt.year
        return jsonify({
            "min_year": int(years.min()),
            "max_year": int(years.max())
        })
    except Exception:
        return jsonify({"min_year": 1743, "max_year": 2013})

@app.route("/predict")
def predict():
    location_type = request.args.get("type", default="country")
    location = request.args.get("location", default="United States")
    year_start = request.args.get("year_start", type=int)
    year_end = request.args.get("year_end", type=int)

    filename = f"{location_type}_{location.replace(' ', '_')}_temperature_trend.png"
    output_file = os.path.join("outputs", filename)
    static_file = os.path.join("static", filename)

    try:
        train_temperature_model(
            location=location,
            year_start=year_start,
            year_end=year_end,
            save_plot=True,
            show_plot=False,
            output_file=output_file,
            dataset_path=DATASETS[location_type],
            location_col=location_type.capitalize()
        )
        shutil.copy(output_file, static_file)
        return render_template("result.html", location=location, image_path=filename)
    except ValueError as e:
        return f"<h3>Input Error: {e}</h3>", 400
    except FileNotFoundError as e:
        return f"<h3>File Error: {e}</h3>", 404
    except Exception as e:
        return f"<h3>Server Error: {e}</h3>", 500

if __name__ == "__main__":
    app.run(debug=True)
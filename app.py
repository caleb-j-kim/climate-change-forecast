from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
from models.linear_regression import train_temperature_model
import shutil
from functools import lru_cache

# Initialize Flask application
app = Flask(__name__)

# Define base directory and paths to different temperature datasets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = {
    "country": os.path.join(BASE_DIR, "datasets", "GlobalLandTemperaturesByCountry.csv"),
    "state": os.path.join(BASE_DIR, "datasets", "GlobalLandTemperaturesByState.csv"),
    "city": os.path.join(BASE_DIR, "datasets", "GlobalLandTemperaturesByCity.csv")
}

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
def index():
    """
    Render the homepage with country options preloaded (default view).
    """
    countries = load_location_options("country")
    return render_template("index.html", countries=countries)

@app.route("/locations")
def get_locations():
    """
    API endpoint to return available locations based on selected location type.
    """
    location_type = request.args.get("type")
    options = load_location_options(location_type)
    return jsonify(options)

@app.route("/year-range")
def year_range():
    """
    API endpoint to return the minimum and maximum year of available data for a given location.
    """
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
        # Fallback year range if an error occurs (e.g., missing location)
        return jsonify({"min_year": 1743, "max_year": 2013})

@app.route("/predict")
def predict():
    """
    Handles the prediction request and returns a rendered plot image of the forecast.
    Includes error handling for common input and file issues.
    """
    location_type = request.args.get("type", default="country")
    location = request.args.get("location", default="United States")
    year_start = request.args.get("year_start", type=int)
    year_end = request.args.get("year_end", type=int)

    # Output file paths
    filename = f"{location_type}_{location.replace(' ', '_')}_temperature_trend.png"
    output_file = os.path.join("outputs", filename)
    static_file = os.path.join("static", filename)

    try:
        # Train and generate plot
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

        # Copy generated plot to static folder for rendering
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

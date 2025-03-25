from flask import Flask, request, render_template
import os
from models.linear_regression import train_temperature_model
import shutil
import pandas as pd

# Get the absolute path to the current project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute path to the country-level dataset
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "GlobalLandTemperaturesByCountry.csv")


app = Flask(__name__)

@app.route("/")
def index():
    try:
        # Load dataset and extract sorted list of countries
        df = pd.read_csv(DATASET_PATH)
        countries = sorted(df["Country"].dropna().unique())
    except Exception as e:
        countries = ["United States", "India", "Brazil"] #fallback defaults
        print(f"Warning: Could not load countries: {e}")

    return render_template("index.html", countries=countries)

from flask import jsonify

@app.route("/year-range")
def year_range():
    country = request.args.get("country")
    try:
        df = pd.read_csv(DATASET_PATH)
        df = df[df["Country"] == country]
        df["dt"] = pd.to_datetime(df["dt"])
        years = df["dt"].dt.year
        return jsonify({
            "min_year": int(years.min()),
            "max_year": int(years.max())
        })
    except Exception as e:
        return jsonify({"min_year": 1800, "max_year": 2020})  # fallback


# Prediction route accepts user input and displays forecast chart
@app.route("/predict")
def predict():
    # Parse query parameters from form
    country = request.args.get("country", default = "United States")
    year_start = request.args.get("year_start", type = int, default = None)
    year_end = request.args.get("year_end", type = int, default = None)

    # Set output/static file
    filename = f"{country.replace(' ', '_')}_temperature_trend.png"
    output_file = os.path.join("outputs", filename)
    static_file = os.path.join("static", filename)

    try:
        # Train model and save plots
        train_temperature_model(
            country = country,
            year_start = year_start,
            year_end = year_end,
            save_plot = True,
            show_plot = False,
            output_file = output_file,
            dataset_path = DATASET_PATH
        )
        
        # Copy plot to static folder so browser can load it
        shutil.copy(output_file, static_file)

        # Render image from static/filename
        image_url = os.path.join("static", filename)
        return render_template("result.html", country = country, image_path = image_url)

    except ValueError as e:
        return f"<h3>Input Error: {e}</h3>", 400
    except FileNotFoundError as e:
        return f"<h3>File Error: {e}</h3>", 404
    except Exception as e:
        return f"<h3>Server Error: {e}</h3>", 500

if __name__ == "__main__":
    app.run(debug=True)
=======
from flask import Flask
app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, World!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
>>>>>>> af5d17bd622dabbb4f40007fe8988b99aea7551b

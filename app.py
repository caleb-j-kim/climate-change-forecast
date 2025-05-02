# py app.py or python app.py to run app on Python backend

from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
import uuid
import shutil
from backend.predictions.linear_regression import train_all, test_all, predict_lr
# from backend.predictions.random_forest import train_all, test_all, predict_rf
from functools import lru_cache
from utils.s3_utils import upload_image_to_s3
from utils.dynamo_utils import save_forecast_to_dynamodb

app = Flask(__name__)

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
    train_all()
    return "Successfully trained all models."

@app.route("/test", methods=["GET"])
def test_endpoint():
    results = test_all()
    return jsonify(["Successfully tested all models."] + results)

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided."}), 400
    
    dataset = data.get("dataset", "country").lower()
    try:
        year = int(data.get("year"))
        month = int(data.get("month"))
    except (TypeError, ValueError):
        return jsonify({"error": "Please provide valid integer values for 'year' and 'month'."}), 400
    location = data.get("location", None)
    
    try:
        prediction = predict_lr(dataset, year, month, location) # Should be ensemble method at the end
    except Exception as e:
        return jsonify({"error": str(e)}), 400

'''@app.route("/predict")
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

    #filename = f"{location_type}_{location.replace(' ', '_')}_temperature_trend.png"
    safe_loc = location.replace(" ", "_")
    filename = f"{location_type}_{safe_loc}_temperature_trend.png"
    output_file = os.path.join("outputs", filename)
    s3_key = f"forecasts/{location_type}/{filename}"
    #static_file = os.path.join("static", filename)

    try:

        prediction = predict_rf(dataset, year, month, location)

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

        # Upload image to S3
        #s3_key = f"forecasts/{location_type}/{filename}"
        s3_url = upload_image_to_s3(output_file, S3_BUCKET, s3_key)
        #print(f"[DEBUG] S3 URL: {s3_url}")


        # Save metadata to DynamoDB
        save_forecast_to_dynamodb(
            table_name=DYNAMO_TABLE,
            location_type=location_type,
            location=location,
            s3_url=s3_url,
            year_start=year_start,
            year_end=year_end
        )

        # Copy generated plot to static folder for rendering
        #shutil.copy(output_file, static_file)

        #return render_template("result.html", location=location, image_path=filename)

        #loads the image from the cloud
        print(f"[DEBUG] S3 URL: {s3_url}")  # Should print https://...
        return render_template("result.html", location=location, image_path=s3_url)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    return jsonify({
        "dataset": dataset,
        "year": year,
        "month": month,
        "location": location,
        "predicted_temperature": prediction
    }) 

    except ValueError as e:
        return f"<h3>Input Error: {e}</h3>", 400
    except FileNotFoundError as e:
        return f"<h3>File Error: {e}</h3>", 404
        year = int(data.get("year"))
        month = int(data.get("month"))
    except (TypeError, ValueError):
        return jsonify({"error": "Please provide valid integer values for 'year' and 'month'."}), 400
    location = data.get("location", None)

    (Leave this part out for now)
'''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    
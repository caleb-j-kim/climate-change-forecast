from flask import Flask, request, jsonify
from backend.short_term_predictions.random_forest import train_all, test_all, predict_rf

app = Flask(__name__)


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
        prediction = predict_rf(dataset, year, month, location)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    return jsonify({
        "dataset": dataset,
        "year": year,
        "month": month,
        "location": location,
        "predicted_temperature": prediction
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    
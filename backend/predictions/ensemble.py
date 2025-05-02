# Combines both linear regression and random forest models to work as one.
import os
import numpy as np
import xgboost as xgb
from backend.predictions.linear_regression import train_all as train_lr, test_all as test_lr, predict_lr
from backend.predictions.random_forest import train_all as train_rf, test_all as test_rf, predict_rf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

"""
    This ensemble method combines both the Linear Regression and Random Forest models to work as one.
    - Linear Regression is great for finding trends and relationships in data.
    - Random Forest is great for handling complex data with many features and interactions along with outliers.
    - The ensemble method combines the strengths of both models to improve prediction accuracy.
"""

def train_ensemble():
    print("Training Linear Regression models...")
    train_lr()
    print("Training Random Forest models...")
    train_rf()
    print("Ensemble training is complete.")

def test_ensemble(weights=(0.5, 0.5)):
    
    # Run the test functions for both models and collect their results.
    print("Testing Linear Regression models...")
    lr_list = test_lr()
    print("Testing Random Forest models...")
    rf_list = test_rf()
    print("Ensemble testing is complete.")
    print("Calculating ensemble method metrics...")
    w_lr, w_rf = weights

    # Convert to dictionaries
    lr_map = { list(d.keys())[0]: list(d.values())[0] for d in lr_list }
    rf_map = { list(d.keys())[0]: list(d.values())[0] for d in rf_list }   
    
    out = {}
    for ds in ("country", "city", "state"):
        lr_m = lr_map[ds]
        rf_m = rf_map[ds]

        # Calculate metrics for ensemble method
        ens = {}
        for metric in ("MSE", "MAE", "R2"):
            ens[metric] = w_lr * lr_m[metric] + w_rf * rf_m[metric]

        out[ds] = {
            "lr":       lr_m,
            "rf":       rf_m,
            "ensemble": ens
        }

    return out

def predict_ensemble(dataset, year, month, location=None):
    # Make predictions using both models and return averaged result.
    
    # Linear Regression returns a dictionary with key 'predicted_temperature'
    lr_out = predict_lr(dataset, year, month, location=location)

    # Random Forest returns a float value along with a DMatrix (similar to a JSON payload)
    rf_temp = predict_rf(dataset, year, month, location=location)

    lr_temp = lr_out['predicted_temperature']
    ensemble_temp = (lr_temp + rf_temp) / 2

    # Merge outputs
    result = {
        'dataset': dataset,
        'location': lr_out['location'],
        'month': month,
        'year': year,
        'lr_prediction': lr_temp,
        'rf_prediction': rf_temp,
        'ensemble_prediction': ensemble_temp
    }

    return result

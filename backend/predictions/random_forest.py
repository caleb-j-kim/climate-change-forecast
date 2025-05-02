import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Dictionary to hold all Random Forest models (one per dataset-location)

rf_models = {}

# Preprocess datasets and clean up data for more accurate predictions

def preprocess_country(file_path, min_year=None, max_year=None, sample_frac=None):
    data = pd.read_csv(file_path)
    
    # Convert date column to datetime and drop missing values
    data['dt'] = pd.to_datetime(data['dt'], errors = 'coerce')
    data.dropna(subset=['dt', 'AverageTemperature', 'Country'], inplace=True)
    
    # Clean up Country strings to avoid mismatch (e.g. "United Sates" vs "   United  States")
    data['Country'] = data['Country'].astype(str).str.strip().str.lower()
    
    # Extract year, month, and filter by years
    data['year'] = data['dt'].dt.year
    data['month'] = data['dt'].dt.month
    
    # Apply year filters if provided
    if min_year is not None:
        data = data[data['year'] >= min_year]
    if max_year is not None:
        data = data[data['year'] <= max_year]
        
    # Sampling for speed (e.g. sample_frac=0.2 keeps 20% of the data)
    if sample_frac is not None and 0 < sample_frac < 1.0:
        data = data.sample(frac=sample_frac, random_state=42)
        
    # Clean location strings: strip whitespace
    data['Country'] = data['Country'].astype(str).str.strip().str.lower()
    
    # Scale numerical values
    scaler = StandardScaler()
    data[['year', 'month']] = scaler.fit_transform(data[['year', 'month']])
    
    # Encode the Country column
    le = LabelEncoder()
    data['Country_encoded'] = le.fit_transform(data['Country'])
    print("After cleaning, country dataset shape:", data.shape)
    
    return data, scaler, le

def preprocess_city(file_path, min_year=None, max_year=None, sample_frac=None):
    data = pd.read_csv(file_path)
    
    # Force all column names to strings
    data.columns = [str(col) for col in data.columns]
    
    # Remove columns with missing header values
    data = data.loc[:, data.columns.notnull()]
    
    # Define required columns (Country is a required field as well due to multiple cities having the same name in different countries)
    required = ['dt', 'AverageTemperature', 'City', 'Country'] 
    
    # Drop rows missing any required field
    data = data.dropna(subset=required)
    
    # Convert dt column to datetime
    data['dt'] = pd.to_datetime(data['dt'], errors='coerce')
    
    # Determine additional columns (columns not in required)
    additional_columns = [col for col in data.columns if col not in required]
    if additional_columns:
        data['additional_count'] = data[additional_columns].notnull().sum(axis=1)
        data = data[data['additional_count'] >= 1]
        data = data.drop(columns=['additional_count'])
    
    # Clean up City strings (remove extra whitespace and force lowercase)
    data['City'] = data['City'].astype(str).str.strip().str.lower()

    # Clean up Country strings (remove extra whitespace and force lowercase)
    data['Country'] = data['Country'].astype(str).str.strip().str.lower()
    
    # Extract year and month
    data['year'] = data['dt'].dt.year
    data['month'] = data['dt'].dt.month
    
    # Apply year filters if provided
    if min_year is not None:
        data = data[data['year'] >= min_year]
    if max_year is not None:
        data = data[data['year'] <= max_year]
    
    # Sampling for speed if needed
    if sample_frac is not None and 0 < sample_frac < 1.0:
        data = data.sample(frac=sample_frac, random_state=42)
    
    # Scale numerical values (year and month)
    scaler = StandardScaler()
    data[['year', 'month']] = scaler.fit_transform(data[['year', 'month']])
    
    # Encode the City column
    le_city = LabelEncoder()
    data['City_encoded'] = le_city.fit_transform(data['City'])

    # Encode the Country column
    le_country = LabelEncoder()
    data['Country_encoded'] = le_country.fit_transform(data['Country'])
    
    print("After cleaning, city dataset shape:", data.shape)
    return data, scaler, (le_city, le_country)

def preprocess_state(file_path, min_year=None, max_year=None, sample_frac=None):
    data = pd.read_csv(file_path)

    # Force all column names to strings
    data.columns = [str(col) for col in data.columns]
    
    # Remove columns with missing header values
    data = data.loc[:, data.columns.notnull()]
    
    # Define required columns (Country is a required field as well due to multiple states having the same name in different countries)
    required = ['dt', 'AverageTemperature', 'State', 'Country']
    
    # Drop rows missing any required field
    data = data.dropna(subset=required)
    
    # Convert dt column to datetime
    data['dt'] = pd.to_datetime(data['dt'], errors='coerce')

    # Determine additional columns (columns not in required)
    additional_columns = [col for col in data.columns if col not in required]
    if additional_columns:
        data['additional_count'] = data[additional_columns].notnull().sum(axis=1)
        data = data[data['additional_count'] >= 1]
        data = data.drop(columns=['additional_count'])
    
    # Clean up State strings (remove extra whitespace and force lowercase)
    data['State'] = data['State'].astype(str).str.strip().str.lower()

    # Clean up Country strings (remove extra whitespace and force lowercase)
    data['Country'] = data['Country'].astype(str).str.strip().str.lower()
    
    # Extract year and month
    data['year'] = data['dt'].dt.year
    data['month'] = data['dt'].dt.month
    
    # Apply year filters if provided
    if min_year is not None:
        data = data[data['year'] >= min_year]
    if max_year is not None:
        data = data[data['year'] <= max_year]
    
    # Sampling for speed if needed
    if sample_frac is not None and 0 < sample_frac < 1.0:
        data = data.sample(frac=sample_frac, random_state=42)
    
    # Scale numerical values (year and month)
    scaler = StandardScaler()
    data[['year', 'month']] = scaler.fit_transform(data[['year', 'month']])
    
    # Encode the State column
    le_state = LabelEncoder()
    data['State_encoded'] = le_state.fit_transform(data['State'])

    # Encode the Country column
    le_country = LabelEncoder()
    data['Country_encoded'] = le_country.fit_transform(data['Country'])

    print("After cleaning, state dataset shape:", data.shape)
    return data, scaler, (le_state, le_country)  

# Train models with the use of Early Stopping and Hyperparameter Tuning to boost performance

def train_country(file_path):
    
    # Enter preprocessing first
    try: 
        # use hyperparamter tuning to adjust sample_frac if performance is too low
        data, scaler, le = preprocess_country(file_path, min_year=1800, max_year=2100, sample_frac=1)
    
    except Exception as e:
        print("Error during preprocessing {file_path}: {e}")
        return
    
    # Use year, month, and encoded country as features
    X = data[['year', 'month', 'Country_encoded']]
    y = data['AverageTemperature']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 10,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    
    evals = [(dval, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=200, evals=evals, early_stopping_rounds=10)
    
    rf_models['country'] = {
        "model": model,
        "features": ['year', 'month', 'Country_encoded'],
        "scaler": scaler,
        "le": le
    }
    
    print("Successfully trained Random Forest model for country dataset.")
    print("Dataset shape:", data.shape)
    
def train_city(file_path):
    try: 
        # use hyperparamter tuning to adjust sample_frac if performance is too low
        data, scaler, (le_city, le_country) = preprocess_city(file_path, min_year=1800, max_year=2100, sample_frac=1)
    
    except Exception as e:
        print(f"Error during preprocessing {file_path}: {e}")
        return
    
    # Use year, month, and encoded country as features
    X = data[['year', 'month', 'City_encoded', 'Country_encoded']]
    y = data['AverageTemperature']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 10,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    
    evals = [(dval, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=200, evals=evals, early_stopping_rounds=10)
    
    rf_models['city'] = {
        "model": model,
        "features": ['year', 'month', 'City_encoded', 'Country_encoded'],
        "scaler": scaler,
        "le": {
            "City": le_city,
            "Country": le_country
        }
    }
    
    print("Successfully trained Random Forest model for city dataset.")
    print("Dataset shape:", data.shape)

def train_state(file_path):
    try: 
        # use hyperparamter tuning to adjust sample_frac if performance is too low
        data, scaler, (le_state, le_country) = preprocess_state(file_path, min_year=1800, max_year=2100, sample_frac=1)
    
    except Exception as e:
        print(f"Error during preprocessing {file_path}: {e}")
        return
    
    # Use year, month, and encoded country as features
    X = data[['year', 'month', 'State_encoded', 'Country_encoded']]
    y = data['AverageTemperature']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 10,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    
    evals = [(dval, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=200, evals=evals, early_stopping_rounds=10)
    
    rf_models['state'] = {
        "model": model,
        "features": ['year', 'month', 'State_encoded', 'Country_encoded'],
        "scaler": scaler,
        "le": {
            "State": le_state,
            "Country": le_country
        }
    }

    print("Successfully trained Random Forest model for state dataset.")
    print("Dataset shape:", data.shape)
    
def train_all():
    country_path = "datasets/GlobalLandTemperaturesByCountry.csv"
    city_path = "datasets/GlobalLandTemperaturesByCity.csv"
    state_path = "datasets/GlobalLandTemperaturesByState.csv"

    print("Country file exists?", os.path.exists(country_path))
    print("City file exists?", os.path.exists(city_path))
    print("State file exists?", os.path.exists(state_path))
    
    # Verify if the datasets exist before training
    
    if os.path.exists(country_path):
        train_country(country_path)
    else:
        print("Country dataset not found at ", country_path)
        
    if os.path.exists(city_path):
        train_city(city_path)
    else:
        print("City dataset not found at ", city_path)
    
    if os.path.exists(state_path):
        train_state(state_path)
    else:
        print("State dataset not found at ", state_path)

# Test the models

def test_country(file_path):
    # use hyperparamter tuning to adjust sample_frac if performance is too low
    data, scaler, le = preprocess_country(file_path, min_year=1800, max_year=2100, sample_frac=1)
    X = data[['year', 'month', 'Country_encoded']]
    y = data['AverageTemperature']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 10,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    evals = [(dtest, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=200, evals=evals, early_stopping_rounds=10)
    y_pred = model.predict(xgb.DMatrix(X_test))
    
    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }
    
    print("Successfully tested Random Forest model for country dataset.")
    print("Dataset shape:", data.shape)
    print("Country model metrics: ", metrics)
    return metrics

def test_city(file_path):
    # use hyperparamter tuning to adjust sample_frac if performance is too low
    data, scaler, (le_state, le_country) = preprocess_city(file_path, min_year=1800, max_year=2100, sample_frac=1)
    X = data[['year', 'month', 'City_encoded', 'Country_encoded']]
    y = data['AverageTemperature']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 10,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    evals = [(dtest, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=200, evals=evals, early_stopping_rounds=10)
    y_pred = model.predict(xgb.DMatrix(X_test))
    
    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }
    
    print("Successfully tested Random Forest model for city dataset.")
    print("Dataset shape:", data.shape)
    print("City model metrics: ", metrics)
    return metrics

def test_state(file_path):
    # use hyperparamter tuning to adjust sample_frac if performance is too low
    data, scaler, (le_state, le_country) = preprocess_state(file_path, min_year=1800, max_year=2100, sample_frac=1)
    X = data[['year', 'month', 'State_encoded', 'Country_encoded']]
    y = data['AverageTemperature']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 10,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    evals = [(dtest, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=200, evals=evals, early_stopping_rounds=10)
    y_pred = model.predict(xgb.DMatrix(X_test))
    
    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }
    
    print("Successfully tested Random Forest model for state dataset.")
    print("Dataset shape:", data.shape)
    print("State model metrics: ", metrics)
    return metrics

def test_all():
    results = []
    country_path = "datasets/GlobalLandTemperaturesByCountry.csv"
    city_path = "datasets/GlobalLandTemperaturesByCity.csv"
    state_path = "datasets/GlobalLandTemperaturesByState.csv"

    if os.path.exists(country_path):
        res = test_country(country_path)
        
        # Ensure res is not None before appending
        if res is not None:
            results.append({"country": res})
            
    if os.path.exists(city_path):
        res = test_city(city_path)
        if res is not None:
            results.append({"city": res})

    if os.path.exists(state_path):
        res = test_state(state_path)
        if res is not None:
            results.append({"state": res})
            
    return results

# Predictions (for real time predictions)

def predict_rf(dataset, year, month, location=None):
    dataset = dataset.lower()
    if dataset not in rf_models:
        raise ValueError(f"Model for {dataset} not found. Available models: {list(rf_models.keys())}")
    
    model_info = rf_models[dataset]
    features = model_info["features"]
    
    # Prepare and scale numerical features
    input_df = pd.DataFrame([[year, month]], columns=['year', 'month'])
    scaler = model_info.get('scaler')
    if scaler is not None:
        scaled_features = scaler.transform(input_df)
        input_features = list(scaled_features[0])
    else:
        input_features = [year, month]
        
    # Encode the provided location (cleaned to lowercase)
    if dataset == "country":
        if not location:
            raise ValueError("A 'location' parameter is required for the country dataset.")
        encoder = model_info.get("le")
        loc_clean = str(location).strip().lower()
        try:
            loc_encoded = int(encoder.transform([loc_clean])[0])
        except Exception:
            available = encoder.classes_.tolist()
            raise ValueError(f"Location '{location}' (cleaned as '{loc_clean}') not found in the training data. Available locations: {available}")
        input_features.append(loc_encoded)

    elif dataset == "city":
        if not location or not isinstance(location, dict):
            # Expect a dictionary with 'city' and 'country' keys for greater accuracy
            raise ValueError("A 'location' parameter of city and country is required for the city dataset.")
        
        city_input = str(location.get("city", "")).strip().lower()
        country_input = str(location.get("country", "")).strip().lower()

        if not city_input or not country_input:
            raise ValueError("Both 'city' and 'country' keys are required in the location dictionary.")
        
        encoder_city = model_info["le"]["City"]
        encoder_country = model_info["le"]["Country"]

        try:
            city_encoded = int(encoder_city.transform([city_input])[0])
        except Exception:
            available = encoder_city.classes_.tolist()
            raise ValueError(f"City '{city_input}' not found in the training data. Available cities: {available}")

        try:
            country_encoded = int(encoder_country.transform([country_input])[0])
        except Exception:
            available = encoder_country.classes_.tolist()
            raise ValueError(f"Country '{country_input}' not found in the training data. Available countries: {available}")
        
        input_features.append(city_encoded)
        input_features.append(country_encoded)

    elif dataset == "state":
        if not location or not isinstance(location, dict):
            # Expect a dictionary with 'state' and 'country' keys for greater accuracy
            raise ValueError("A 'location' parameter of city and country is required for the state dataset.")
        
        state_input = str(location.get("state", "")).strip().lower()
        country_input = str(location.get("country", "")).strip().lower()

        if not state_input or not country_input:
            raise ValueError("Both 'state' and 'country' keys are required in the location dictionary.")
        
        encoder_state = model_info["le"]["State"]
        encoder_country = model_info["le"]["Country"]

        try:
            state_encoded = int(encoder_state.transform([state_input])[0])
        except Exception:
            available = encoder_state.classes_.tolist()
            raise ValueError(f"State '{state_input}' not found in the training data. Available states: {available}")

        try:
            country_encoded = int(encoder_country.transform([country_input])[0])
        except Exception:
            available = encoder_country.classes_.tolist()
            raise ValueError(f"Country '{country_input}' not found in the training data. Available countries: {available}")
        
        input_features.append(state_encoded)
        input_features.append(country_encoded)

    if len(input_features) != len(model_info["features"]):
        raise ValueError(f"Expected {len(model_info['features'])} features, but got {len(input_features)}.")
    
    # Create a DMatrix with feature names:
    dinput = xgb.DMatrix(np.array([input_features]), feature_names=model_info["features"])
    prediction = model_info["model"].predict(dinput)
    return float(prediction[0])
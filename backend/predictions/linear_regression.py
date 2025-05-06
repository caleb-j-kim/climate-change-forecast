# For a more streamlined and efficient approach, this code and the random_forest.py will follow the same structure.
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Enables plotting in headless environments (e.g., server-side)
import matplotlib.pyplot as plt
from datetime import datetime # for saving plots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Dictionary to hold all Linear Regression models (one per dataset-location)
lr_models = {}

"""
    Inject quadratic year and cyclical month features into the dataset to improve model performance.
    - year2 = year ** 2 (quadratic feature)
    - month_sin = sin(2 * pi * month / 12) (cyclical feature)
    - month_cos = cos(2 * pi * month / 12) (cyclical feature)
    - This allows the model to learn seasonal patterns more effectively.
"""

def add_time_features(df): 
    df = df.copy()  # Avoid modifying the original DataFrame
    df['year2'] = df['year'] ** 2 
    radians = 2 * np.pi * df['month'] / 12
    df['month_sin'] = np.sin(radians)
    df['month_cos'] = np.cos(radians)
    return df

"""
    Preprocess each dataset (country, state, city) to clean up data and prepare it for training and higher accuracy.
    This includes:
    - Converting date columns to datetime format
    - Dropping rows with missing values in required columns
    - Cleaning up string columns (e.g., removing extra whitespace, converting to lowercase)
    - Extracting year and month from date columns
    - Filtering data based on provided year ranges
    - Sampling data for speed (if needed)
    - Scaling numerical values (e.g., year, month)
"""

def preprocess_country(file_path, min_year=None, max_year=None, sample_frac=None):
    df = pd.read_csv(file_path)

    # Convert data column to datetime and drop missing values
    df['dt'] = pd.to_datetime(df['dt'], errors='coerce')
    df.dropna(subset=['dt', 'AverageTemperature', 'Country'], inplace=True)

    # Clean up Country strings to avoid mismatch (e.g. "United Sates" vs "   United  States")
    df['Country'] = df['Country'].astype(str).str.strip().str.lower()

    # Extract year, month, and filter by years
    df['year'] = df['dt'].dt.year
    df['month'] = df['dt'].dt.month

    # Apply year filters if provided
    if min_year is not None:
        df = df[df['year'] >= min_year]
    if max_year is not None:
        df = df[df['year'] <= max_year]

    # Sampling for speed (e.g. sample_frac=0.2 keeps 20% of the data)
    if sample_frac is not None and 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)

    df = add_time_features(df)

    # Clean location strings: strip whitespace
    df['Country'] = df['Country'].astype(str).str.strip().str.lower()

    # Scale numerical values
    num_feats = ['year', 'year2', 'month_sin', 'month_cos']
    scaler = StandardScaler()
    df[num_feats] = scaler.fit_transform(df[num_feats])

    # Encode categorical variables (e.g. Country)
    le = LabelEncoder()
    df['Country_encoded'] = le.fit_transform(df['Country'])
    print("After cleaning, country dataset shape:", df.shape)

    return df, le, scaler

def preprocess_city(file_path, min_year=None, max_year=None, sample_frac=None):
    df = pd.read_csv(file_path)

    # Force all column names to strings
    df.columns = [str(col) for col in df.columns]

    # Remove columns with missing header values
    df = df.loc[:, df.columns.notnull()]

    # Define required columns (Country is a required field as well due to multiple cities have the same name in different countries)
    required = ['dt', 'AverageTemperature', 'City', 'Country']

    # Drop rows missing any of the required columns
    df = df.dropna(subset=required)

    # Convert dt column to datetime
    df['dt'] = pd.to_datetime(df['dt'], errors='coerce')

    # Determine additional columns (columns not in required)
    additional_columns = [col for col in df.columns if col not in required]
    if additional_columns:
        df['additional_count'] = df[additional_columns].notnull().sum(axis=1)
        df = df[df['additional_count'] >= 1]
        df = df.drop(columns=['additional_count']) 

    # Clean up City strings (remove extra whitespace and force lowercase)
    df['City'] = df['City'].astype(str).str.strip().str.lower()

    # Clean up Country strings (remove extra whitespace and force lowercase)
    df['Country'] = df['Country'].astype(str).str.strip().str.lower()

    # Extract year and month
    df['year'] = df['dt'].dt.year
    df['month'] = df['dt'].dt.month

    # Apply year filters if provided
    if min_year is not None:
        df = df[df['year'] >= min_year]
    if max_year is not None:
        df = df[df['year'] <= max_year]
    
    # Sampling for speed if needed
    if sample_frac is not None and 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)

    df = add_time_features(df)

    # Scale numerical values (e.g. year and month)
    num_feats = ['year', 'year2', 'month_sin', 'month_cos']
    scaler = StandardScaler()
    df[num_feats] = scaler.fit_transform(df[num_feats])

    # Encode the City column
    le_city = LabelEncoder()
    df['City_encoded'] = le_city.fit_transform(df['City'])

    # Encode the Country column
    le_country = LabelEncoder()
    df['Country_encoded'] = le_country.fit_transform(df['Country'])

    print("After cleaning, city dataset shape:", df.shape)
    return df, scaler, (le_city, le_country) # Tuple for both encoders as they are used in the prediction function


def preprocess_state(file_path, min_year=None, max_year=None, sample_frac=None):
    df = pd.read_csv(file_path)

    # Force all column names to strings
    df.columns = [str(col) for col in df.columns]

    # Remove columns with missing header values
    df = df.loc[:, df.columns.notnull()]

    # Define required columns (Country is a required field as well due to multiple states have the same name in different countries)
    required = ['dt', 'AverageTemperature', 'State', 'Country']

    # Drop rows missing any required field
    df = df.dropna(subset=required)

    # Convert dt column to datetime
    df['dt'] = pd.to_datetime(df['dt'], errors='coerce')

    # Determine additional columns (columns not in required)
    additional_columns = [col for col in df.columns if col not in required]
    if additional_columns:
        df['additional_count'] = df[additional_columns].notnull().sum(axis=1)
        df = df[df['additional_count'] >= 1]
        df = df.drop(columns=['additional_count'])

    # Clean up State strings (remove extra whitespace and force lowercase)
    df['State'] = df['State'].astype(str).str.strip().str.lower()

    # Clean up Country strings (remove extra whitespace and force lowercase)
    df['Country'] = df['Country'].astype(str).str.strip().str.lower()

    # Extract year and month
    df['year'] = df['dt'].dt.year
    df['month'] = df['dt'].dt.month

    # Apply year filters if provided
    if min_year is not None:
        df = df[df['year'] >= min_year]
    if max_year is not None:
        df = df[df['year'] <= max_year]

    # Sampling for speed if needed
    if sample_frac is not None and 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)

    df = add_time_features(df)

    # Scale numerical values (e.g. year and month)
    num_feats = ['year', 'year2', 'month_sin', 'month_cos']
    scaler = StandardScaler()
    df[num_feats] = scaler.fit_transform(df[num_feats])

    # Encode the State column
    le_state = LabelEncoder()
    df['State_encoded'] = le_state.fit_transform(df['State'])

    # Encode the Country column
    le_country = LabelEncoder()
    df['Country_encoded'] = le_country.fit_transform(df['Country'])

    print("After cleaning, state dataset shape:", df.shape)
    return df, scaler, (le_state, le_country) 

"""
    Trains a linear regression model to forecast average temperature trends for a specified location.
    - The trained model is saved for future predictions.
"""

def train_country(file_path):

    # Enter preprocessing first
    try:
        df, le, scaler = preprocess_country(file_path, min_year=1800, max_year=2100, sample_frac=1)
    
    except Exception as e:
        print("Error during preprocessing {file_path}: {e}")
        return
    
    # Use year, month, and encoded country as features
    feats = ['year','year2','month_sin','month_cos','Country_encoded']
    X, y = df[feats], df['AverageTemperature']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    lr_models['country'] = {
        'model': model,
        'features': ['year', 'month', 'Country_encoded'],
        'scaler': scaler,
        'le': le,
        'features': feats
    }

    print("Successfully trained Linear Regression model for country dataset.")
    print("Dataset shape:", df.shape) # Make sure this matches the one in preprocess_country    

def train_city(file_path):
    try: 
        df, scaler, (le_city, le_country) = preprocess_city(file_path, min_year=1800, max_year=2100, sample_frac=1)
    
    except Exception as e:
        print(f"Error during preprocessing {file_path}: {e}")
        return

    # Use year, month, and encoded city as features
    feats = ['year','year2','month_sin','month_cos','City_encoded', 'Country_encoded']
    X, y = df[feats], df['AverageTemperature']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    lr_models['city'] = {
        'model': model,
        'features': ['year', 'month', 'City_encoded', 'Country_encoded'],
        'scaler': scaler,
        'le': {'City': le_city, 'Country': le_country},
        'features': feats
    }

    print("Successfully trained Linear Regression model for city dataset.")
    print("Dataset shape:", df.shape)

def train_state(file_path):

    # Enter preprocessing first
    try: 
        df, scaler, (le_state, le_country) = preprocess_state(file_path, min_year=1800, max_year=2100, sample_frac=1)
    
    except Exception as e:
        print("Error during preprocessing {file_path}: {e}")
        return

    # Use year, month, and encoded state as features
    feats = ['year','year2','month_sin','month_cos', 'State_encoded', 'Country_encoded']
    X, y = df[feats], df['AverageTemperature']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    lr_models['state'] = {
        'model': model,
        'features': ['year', 'month', 'State_encoded', 'Country_encoded'],
        'scaler': scaler,
        'le': {'State': le_state, 'Country': le_country},
        'features': feats
    }

    print("Successfully trained Linear Regression model for state dataset.")
    print("Dataset shape:", df.shape)

def train_all(): # Train all models using one function that's called by one singular RESTful API endpoint
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

"""
    Tests the trained models by making predictions on the test set and calculating performance metrics.
    Saves or displays a plot of the prediction based on input flags.
    Returns a dictionary with the model name and its performance metrics.
    The metrics include Mean Squared Error (MSE), R-squared (R2), and Mean Absolute Error (MAE).
"""

def test_country(file_path):
    df, scaler, le = preprocess_country(file_path, min_year=1800, max_year=2100, sample_frac=1)
    feats = lr_models['country']['features']
    X, y = df[feats], df['AverageTemperature']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Test linear regression model
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }
    
    print("Successfully tested Random Forest model for country dataset.")
    print("Dataset shape:", df.shape)
    print("Country model metrics: ", metrics)
    return metrics

def test_city(file_path):
    df, scaler, (le_city, le_country) = preprocess_city(file_path, min_year=1800, max_year=2100, sample_frac=1)
    feats = lr_models['city']['features']
    X, y = df[feats], df['AverageTemperature']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Test linear regression model
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }
    
    print("Successfully tested Random Forest model for city dataset.")
    print("Dataset shape:", df.shape)
    print("City model metrics: ", metrics)
    return metrics

def test_state(file_path):
    df, scaler, (le_state, le_country) = preprocess_state(file_path, min_year=1800, max_year=2100, sample_frac=1)
    feats = lr_models['state']['features']
    X, y = df[feats], df['AverageTemperature']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Test linear regression model
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }
    
    print("Successfully tested Random Forest model for state dataset.")
    print("Dataset shape:", df.shape)
    print("State model metrics: ", metrics)
    return metrics

def test_all(): # Test all models using one function that's called by one singular RESTful API endpoint
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

"""
    Predicts the average temperature for a given location using the trained Linear Regression model.
    The location can be a country, state, or city.
    The function takes the location name, year, and month as input parameters.
    It returns the predicted average temperature for that location as both a value and a plot.
    - Each graph is labeled with their dataset they referenced along with a time-stamp of when the prediction was made.
    - The title of the graph is "(Dataset Name) + (Location) Monthly Temperature Trend of (Year)"
    (The plot is saved to a file and can be uploaded to S3 or displayed as needed.)
"""

def predict_lr(dataset, year, month, location=None, 
               save_plot=True, show_plot=False,
               output_file="outputs/linear_regression_prediction_plot.png"):
    ds = dataset.lower()
    print("DEBUG predict_lr dataset:", ds)
    if ds not in lr_models:
        raise ValueError(f"Model for {ds} not found. Available models: {list(lr_models.keys())}")
    
    # Variables for plotting and prediction
    info = lr_models[ds]
    features = info['features']
    model = info['model']
    scaler = info['scaler']
    le = info['le']

    # Build dataframe for prediction
    dfp = pd.DataFrame({'year':[year],'month':[month]})
    dfp = add_time_features(dfp)
    num_feats = ['year','year2','month_sin','month_cos']
    dfp[num_feats] = scaler.transform(dfp[num_feats])


    # Encode location
    if ds == 'country':
        # Handle both string and dictionary format for countries
        if isinstance(location, dict) and 'country' in location:
            location = location['country']
        if not isinstance(location, str):
            raise ValueError("Country location must be a string or a dictionary with 'country' key.")
        
        code = le.transform([location.strip().lower()])[0]
        dfp['Country_encoded'] = code

    elif ds == 'city':
        if not isinstance(location, dict) or "city" not in location or "country" not in location:
            raise ValueError("City location must be a dictionary with 'city' and 'country' keys.")
        dfp['City_encoded']    = le['City'].transform([location['city'].strip().lower()])[0]
        dfp['Country_encoded'] = le['Country'].transform([location['country'].strip().lower()])[0]

    elif ds == 'state':
        if not isinstance(location, dict) or "state" not in location or "country" not in location:
            raise ValueError("State location must be a dictionary with 'state' and 'country' keys.")
        dfp['State_encoded']   = le['State'].transform([location['state'].strip().lower()])[0]
        dfp['Country_encoded'] = le['Country'].transform([location['country'].strip().lower()])[0]

    else:
        raise ValueError(f"Invalid dataset '{ds}'. Available datasets: {list(lr_models.keys())}")
    
    # Make a prediction using the model
    Xp = dfp[features]
    prediction = float(model.predict(Xp)[0])

    # Plot historical trend for this prediction
    csv_map = {
        'country': "datasets/GlobalLandTemperaturesByCountry.csv",
        'city': "datasets/GlobalLandTemperaturesByCity.csv",
        'state': "datasets/GlobalLandTemperaturesByState.csv"
    }

    # Filter 'raw' and building with a DateTime index
    raw=pd.read_csv(csv_map[ds])
    raw['dt']=pd.to_datetime(raw['dt'],errors='coerce')
    raw.dropna(subset=['AverageTemperature'],inplace=True)

    if ds=='country':
        # Handle both string and dict for country location when filtering
        country_name = location if isinstance(location, str) else location.get('country', '')
        filt = raw['Country'].str.strip().str.lower() == country_name.strip().lower()
    elif ds=='city':
        filt = (
          (raw['City'].str.strip().str.lower()    == location['city'].strip().lower()) &
          (raw['Country'].str.strip().str.lower() == location['country'].strip().lower())
        )
    else:
        filt = (
          (raw['State'].str.strip().str.lower()   == location['state'].strip().lower()) &
          (raw['Country'].str.strip().str.lower() == location['country'].strip().lower())
        )

    # Build a true monthly time-series for the given location and restrict to the year
    year_df = raw.loc[filt & (raw['dt'].dt.year==year), ['dt','AverageTemperature']].copy()
    year_df['month'] = year_df['dt'].dt.month
    monthly = ( year_df
                .groupby('month')['AverageTemperature']
                .mean()
                .reindex(range(1,13))
                .reset_index() )

    # Plot scatter + trend line
    plt.figure(figsize=(10,6))

    if monthly['AverageTemperature'].notna().sum() >= 2:
        # enough history: scatter + trend
        plt.scatter(monthly['month'], monthly['AverageTemperature'], label='Monthly Avg', alpha=0.8)
        lr = LinearRegression()
        m = monthly[['month']].values
        y = monthly['AverageTemperature'].values
        mask = ~np.isnan(y)
        lr.fit(m[mask], y[mask])
        plt.plot(monthly['month'], lr.predict(m), linestyle='--', label='Trend')
    else:
        # fallback: plot long‐term seasonal avg + model trend
        df_loc = raw[filt].copy()
        df_loc['month'] = df_loc['dt'].dt.month
        seasonal = ( df_loc
                     .groupby('month')['AverageTemperature']
                     .mean()
                     .reindex(range(1,13))
                     .reset_index() )
        plt.scatter(seasonal['month'], seasonal['AverageTemperature'],
                    label='Hist. avg. by month', alpha=0.5, color='gray')

        # build a 12‐month forecast matrix properly
        months = np.arange(1,13)
        X_pred_rows = []
        for mm in months:
            tmp = pd.DataFrame({'year':[year], 'month':[mm]})
            tmp = add_time_features(tmp)
            tmp[num_feats] = scaler.transform(tmp[num_feats])
            # inject the same encoded values from dfp
            for feat in features:
                if feat not in num_feats:
                    tmp[feat] = dfp[feat].iloc[0]
            X_pred_rows.append(tmp[features].iloc[0].values)
        X_pred = np.vstack(X_pred_rows)
        y_pred = model.predict(X_pred)
        plt.plot(months, y_pred, linestyle='--', label='Model trend')
        plt.title(f"No {year} history for {location!r}, showing seasonal + model")

    plt.xticks(range(1,13), ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    plt.xlabel("Month")
    plt.ylabel("Avg Temp (°C)")
    plt.grid(alpha=0.3)
    plt.legend()
    # Save the plot to a file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/linear_regression_prediction_plot_{ds}_{ts}.png"
    if save_plot:
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Plot saved to {output_file}.")
    if show_plot:
        plt.show()
    plt.close()

    # Return the prediction value
    return {
        "dataset": ds,
        "location": location if isinstance(location, str) else (location.get('city') or location.get('state')),
        "month": month,
        "predicted_temperature": float(prediction),
        "year": year,
    }
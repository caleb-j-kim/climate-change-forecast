# Perform GET calls with the Tomorrow.io Weather API to update the weather data with modern entries.

from dotenv import load_dotenv 
load_dotenv()  

import os
import random
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

API_KEY = os.getenv("TOMORROW_API_KEY")
if not API_KEY:
    raise RuntimeError("TOMORROW_API_KEY not set in environment variables.")
print(f"API Key: {API_KEY}")

# bounds for "random" coordinates (continental US currently)
MIN_LAT, MAX_LAT = 25.0, 50.0
MIN_LON, MAX_LON = -125.0, -65.0

"""
    Load the datasets and use the contents to randomly pick any city, state, or country in the world.
"""

MODULE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_DIR, os.pardir, os.pardir))
DATA_DIR     = os.path.join(PROJECT_ROOT, "datasets")

country_list = (
    pd.read_csv(os.path.join(DATA_DIR, "GlobalLandTemperaturesByCountry.csv"))
      ["Country"]
      .dropna()
      .str.strip()
      .unique()
      .tolist()
)

city_list = (
    pd.read_csv(os.path.join(DATA_DIR, "GlobalLandTemperaturesByCity.csv"), usecols=["City"])
      ["City"]
      .dropna()
      .str.strip()
      .unique()
      .tolist()
)

state_list = (
    pd.read_csv(os.path.join(DATA_DIR, "GlobalLandTemperaturesByState.csv"), usecols=["State"])
      ["State"]
      .dropna()
      .str.strip()
      .unique()
      .tolist()
)

# a small sample list of cities; swap in whatever you're tracking
CITIES = [
    "new york", "los angeles", "chicago", "houston", "miami",
    "london", "paris", "tokyo", "sydney", "cape town",
]

"""
    Returns a random lattitude and longitude string from the bounds.
"""
def get_random_coordinates() -> str:
    # anywhere on Earth
    lat = random.uniform(-90,  90)
    lon = random.uniform(-180, 180)
    return f"{lat:.4f},{lon:.4f}" # 

"""
    Fetches a one-day, daily-resolution historical timeline for a random location.
    Uses the /v4/timelines GET endpoint.
"""

def fetch_daily_timeline() -> dict:
    loc = get_random_coordinates()
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    params = {
        "location": loc,
        "fields":   "temperature",
        "timesteps":"1d",
        "startTime": today.isoformat(),
        "endTime":   (today + timedelta(days=1)).isoformat(),
        "units":    "metric",
        "apikey":   API_KEY,
    }
    resp = requests.get("https://api.tomorrow.io/v4/timelines", params=params)
    resp.raise_for_status()
    return resp.json()

"""
    Fetch a short-term forecast for a random city, country, and state from our list.
    Uses the /v4/timelines GET endpoint.
"""

def fetch_random_country_forecast() -> tuple[str, dict]:
    country = random.choice(country_list)
    now     = datetime.now(timezone.utc)
    params = {
        "location": country,
        "fields":   "temperature,humidity,windSpeed",
        "timesteps":"1d",
        "startTime": now.isoformat(),
        "endTime":   (now + timedelta(days=5)).isoformat(),
        "units":    "metric",
        "apikey":   API_KEY,
    }
    resp = requests.get("https://api.tomorrow.io/v4/timelines", params=params)
    resp.raise_for_status()
    return country, resp.json()

def fetch_random_city_forecast() -> tuple[str, dict]:
    city = random.choice(city_list)
    now = datetime.now(timezone.utc)
    params = {
        "location": city,
        "fields":   "temperature,humidity,windSpeed",
        "timesteps":"1d",
        "startTime": now.isoformat(),
        "endTime":   (now + timedelta(days=5)).isoformat(),
        "units":    "metric",
        "apikey":   API_KEY,
    }
    resp = requests.get("https://api.tomorrow.io/v4/timelines", params=params)
    resp.raise_for_status()
    return city, resp.json()

def fetch_random_state_forecast() -> tuple[str, dict]:
    state = random.choice(state_list)
    now   = datetime.now(timezone.utc)
    params = {
        "location": state,
        "fields":   "temperature,humidity,windSpeed",
        "timesteps":"1d",
        "startTime": now.isoformat(),
        "endTime":   (now + timedelta(days=5)).isoformat(),
        "units":    "metric",
        "apikey":   API_KEY,
    }
    resp = requests.get("https://api.tomorrow.io/v4/timelines", params=params)
    resp.raise_for_status()
    return state, resp.json()
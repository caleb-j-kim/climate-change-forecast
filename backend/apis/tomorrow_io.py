# Perform GET calls with the Tomorrow.io Weather API to update the weather data with modern entries.

from dotenv import load_dotenv 
load_dotenv()  

import os
import random
import requests
from datetime import datetime, timedelta, timezone

API_KEY = os.getenv("TOMORROW_API_KEY")
if not API_KEY:
    raise RuntimeError("TOMORROW_API_KEY not set in environment variables.")
print(f"API Key: {API_KEY}")

# bounds for "random" coordinates (continental US currently)
MIN_LAT, MAX_LAT = 25.0, 50.0
MIN_LON, MAX_LON = -125.0, -65.0

# a small sample list of cities; swap in whatever you're tracking
CITIES = [
    "new york", "los angeles", "chicago", "houston", "miami",
    "london", "paris", "tokyo", "sydney", "cape town",
]

"""
    Returns a random lattitude and longitude string from the bounds.
"""
def get_random_coordinates() -> str:
    lat = random.uniform(MIN_LAT, MAX_LAT)
    lon = random.uniform(MIN_LON, MAX_LON)
    return f"{lat:.4f},{lon:.4f}" # 

"""
    Fetches a one-day, daily-resolution historical timeline for a random location.
    Uses the /v4/timelines GET endpoint.
"""

def fetch_daily_timeline() -> dict:
    loc = get_random_coordinates()
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = today.isoformat()
    end   = (today + timedelta(days=1)).isoformat()

    params = {
        "location": loc,
        "fields": "temperature",
        "timesteps": "1d",
        "startTime": start,
        "endTime": end,
        "units": "metric",
        "apikey": API_KEY,
    }

    resp = requests.get("https://api.tomorrow.io/v4/timelines", params=params)
    resp.raise_for_status()
    return resp.json()

"""
    Fetch a short-term forecast for a random city from our list.
    Uses the /v4/timelines GET endpoint.
"""
def fetch_random_city_forecast() -> tuple[str, dict]:
    city = random.choice(CITIES)
    today = datetime.now(timezone.utc)
    start = today.isoformat()
    end   = (today + timedelta(days=5)).isoformat()

    params = {
        "location": city,
        "fields": "temperature,humidity,windSpeed",
        "timesteps": "1d",
        "startTime": start,
        "endTime": end,
        "units": "metric",
        "apikey": API_KEY,
    }

    resp = requests.get("https://api.tomorrow.io/v4/timelines", params=params)
    resp.raise_for_status()
    return city, resp.json()
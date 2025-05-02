import boto3
from datetime import datetime
from boto3.dynamodb.conditions import Key
import pandas as pd

def save_forecast_to_dynamodb(table_name, location_type, location, s3_url, year_start=None, year_end=None):
    """
    Save a forecast record into the unified ClimateForecasts table.
    Uses single-table design with partition key = 'location_type#location'
    and sort key = ISO timestamp.
    """
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)

    partition_key = f"{location_type}#{location.replace(' ', '_')}"
    sort_key = datetime.utcnow().isoformat()

    item = {
        "PK": partition_key,
        "SK": sort_key,
        "location_type": location_type,
        "location": location,
        "s3_url": s3_url,
        "timestamp": sort_key
    }

    if year_start is not None and year_end is not None:
        item["year_range"] = f"{year_start}-{year_end}"

    table.put_item(Item=item)

def query_temperature_data(table_name, location_type, location):
    """
    Query temperature data for a given location from DynamoDB and return as a pandas DataFrame.
    Assumes table is named like 'GlobalTempsByCountry' and PK is location, SK is 'dt'.
    """
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)

    pk = location.replace(" ", "_")

    response = table.query(
        KeyConditionExpression=Key(location_type.capitalize()).eq(pk),
        ProjectionExpression="dt, AverageTemperature, AverageTemperatureUncertainty"
    )

    items = response.get("Items", [])
    if not items:
        return pd.DataFrame()  # Empty fallback

    df = pd.DataFrame(items)
    df["dt"] = pd.to_datetime(df["dt"])
    df["AverageTemperature"] = pd.to_numeric(df["AverageTemperature"])
    df["AverageTemperatureUncertainty"] = pd.to_numeric(df["AverageTemperatureUncertainty"])
    return df
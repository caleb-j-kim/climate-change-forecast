import boto3
from datetime import datetime

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

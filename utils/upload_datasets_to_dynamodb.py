import boto3
import pandas as pd
from tqdm import tqdm

dynamodb = boto3.resource("dynamodb", region_name="us-east-2")

def upload_to_dynamodb(table_name, df, key_fn):
    table = dynamodb.Table(table_name)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Uploading to {table_name}"):
        item = {k: str(v) if pd.notna(v) else None for k, v in row.items()}
        item.update(key_fn(row))
        table.put_item(Item=item)

def main():
    # City
    df_city = pd.read_csv("datasets/GlobalLandTemperaturesByCity.csv").dropna(subset=["City", "Country", "dt"])
    upload_to_dynamodb(
        "GlobalLandTemperaturesByCity",
        df_city,
        lambda r: {"CityCountry": f"{r['City']}_{r['Country']}", "dt": r["dt"]}
    )

    # State
    df_state = pd.read_csv("datasets/GlobalLandTemperaturesByState.csv").dropna(subset=["State", "Country", "dt"])
    upload_to_dynamodb(
        "GlobalLandTemperaturesByState",
        df_state,
        lambda r: {"StateCountry": f"{r['State']}_{r['Country']}", "dt": r["dt"]}
    )

    # Country
    df_country = pd.read_csv("datasets/GlobalLandTemperaturesByCountry.csv").dropna(subset=["Country", "dt"])
    upload_to_dynamodb(
        "GlobalLandTemperaturesByCountry",
        df_country,
        lambda r: {"Country": r["Country"], "dt": r["dt"]}
    )

if __name__ == "__main__":
    main()
    
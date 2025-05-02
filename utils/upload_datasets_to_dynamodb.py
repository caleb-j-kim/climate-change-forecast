import boto3
import pandas as pd
from tqdm import tqdm

dynamodb = boto3.resource("dynamodb", region_name="us-east-2")

def batch_upload(table_name, df, key_fn):
    table = dynamodb.Table(table_name)
    with table.batch_writer(overwrite_by_pkeys=list(key_fn(df.iloc[0]).keys())) as batch:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Uploading to {table_name}"):
            item = {k: str(v) if pd.notna(v) else None for k, v in row.items()}
            item.update(key_fn(row))
            batch.put_item(Item=item)

def main():
    # Upload GlobalLandTemperaturesByCity.csv
    df_city = pd.read_csv("datasets/GlobalLandTemperaturesByCity.csv").dropna(subset=["City", "Country", "dt"])
    batch_upload(
        "GlobalLandTemperaturesByCity",
        df_city,
        lambda r: {"CityCountry": f"{r['City']}_{r['Country']}", "dt": r["dt"]}
    )

    # Upload GlobalLandTemperaturesByState.csv
    df_state = pd.read_csv("datasets/GlobalLandTemperaturesByState.csv").dropna(subset=["State", "Country", "dt"])
    batch_upload(
        "GlobalLandTemperaturesByState",
        df_state,
        lambda r: {"StateCountry": f"{r['State']}_{r['Country']}", "dt": r["dt"]}
    )

    # Upload GlobalLandTemperaturesByCountry.csv
    df_country = pd.read_csv("datasets/GlobalLandTemperaturesByCountry.csv").dropna(subset=["Country", "dt"])
    batch_upload(
        "GlobalLandTemperaturesByCountry",
        df_country,
        lambda r: {"Country": r["Country"], "dt": r["dt"]}
    )

if __name__ == "__main__":
    main()

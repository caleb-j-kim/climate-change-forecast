import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # For non-GUI environments
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def train_temperature_model(
    location: str,
    year_start: int,
    year_end: int,
    save_plot: bool = True,
    show_plot: bool = False,
    output_file: str = "outputs/prediction.png",
    dataset_path: str = "datasets/GlobalLandTemperaturesByCountry.csv",
    location_col: str = "Country"
) -> LinearRegression:

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {os.path.abspath(dataset_path)}")

    df = pd.read_csv(dataset_path, usecols=["dt", "AverageTemperature", location_col])
    df = df.dropna(subset=["AverageTemperature", location_col])
    if location not in df[location_col].unique():
        raise ValueError(f"{location_col} '{location}' not found in dataset.")

    df = df[df[location_col] == location].copy()
    df["dt"] = pd.to_datetime(df["dt"])
    df["Year"] = df["dt"].dt.year

    if year_start is not None:
        df = df[df["Year"] >= year_start]
    if year_end is not None:
        df = df[df["Year"] <= year_end]

    yearly_avg = df.groupby("Year")["AverageTemperature"].mean().reset_index()

    X = yearly_avg["Year"].values.reshape(-1, 1)
    y = yearly_avg["AverageTemperature"].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, label="Observed", alpha=0.6)
    plt.plot(X, y_pred, color="red", label="Trend")
    plt.title(f"{location_col} '{location}' Temperature Trend")
    plt.xlabel("Year")
    plt.ylabel("Average Temperature (°C)")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if save_plot:
        plt.savefig(output_file, bbox_inches="tight")
    if show_plot:
        plt.show()

    return model

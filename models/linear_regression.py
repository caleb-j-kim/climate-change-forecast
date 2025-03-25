import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Ensures compatibility with macOS GUI environments
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

"""
    country (str): Selected country
    year_start (int, optional): Starting year
    year_end (int, optional): Ending year
    save_plot (bool): Save chart to a png file
    show_plot (bool): Display chart in a new window
    output_file (str): Output plot image
    """
def train_temperature_model(
    country: str,
    year_start: int = None,
    year_end: int = None,
    save_plot: bool = True,
    show_plot: bool = False,
    output_file: str = "outputs/temperature_trend.png",
    dataset_path: str = "datasets/GlobalLandTemperaturesByCountry.csv"
) -> LinearRegression:

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at path: {os.path.abspath(dataset_path)}")
    df = pd.read_csv(dataset_path)

    # Select columns and drop rows with missing temperature values
    df = df[["dt", "AverageTemperature", "Country"]].dropna()

    # Country validation
    unique_countries = df["Country"].unique()
    if country not in unique_countries:
        raise ValueError(
            f"Country '{country}' not found in dataset. "
            f"Available examples include: {sorted(unique_countries)[:10]}"
        )

    # Filter the dataset for the selected country
    df = df[df["Country"] == country].copy()
    df["dt"] = pd.to_datetime(df["dt"])
    df["Year"] = df["dt"].dt.year

    # Apply year range filters
    if year_start is not None:
        df = df[df["Year"] >= year_start]
    if year_end is not None:
        df = df[df["Year"] <= year_end]

    # Aggregate average temperature by year
    yearly_avg = df.groupby("Year")["AverageTemperature"].mean().reset_index()

    # Prepare features (X) and target variable (y)
    X = yearly_avg["Year"].values.reshape(-1, 1)
    y = yearly_avg["AverageTemperature"].values.reshape(-1, 1)

    # Train linear regression model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Plot actual temperature vs. predicted trend line
    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, label="Observed Average Temperature", alpha=0.6)
    plt.plot(X, y_pred, color="red", label="Linear Regression Trend")
    plt.title(f"Average Yearly Temperature Trend in {country}")
    plt.xlabel("Year")
    plt.ylabel("Average Temperature (°C)")
    plt.legend()
    plt.grid(True)

    # Creates output dir if doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok = True)

    # Save/display the plot
    if save_plot:
        plt.savefig(output_file, bbox_inches = "tight")
        print(f"Plot saved to: {os.path.abspath(output_file)}")
    if show_plot:
        plt.show()

    return model

# Example usage for direct script execution
if __name__ == "__main__":
    # This block can be edited for testing specific scenarios
    train_temperature_model(
        country = "India",
        year_start = 1950,
        year_end = 2020,
        save_plot = True,
        show_plot = False,
        output_file = "india_temperature_trend.png"
    )

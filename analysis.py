# Import Libraries

import pandas as pd
import polars as pl
from sklearn.linear_model import LinearRegression  # Import Linear Regression
import matplotlib.pyplot as plt  # Import Matplotlib

# FUNCTIONS


# Load the Spotify dataset
def load_data(filepath):
    return pd.read_csv(filepath)


# Filter tracks with high popularity
def filter_high_popularity(df):
    return df[df["popularity_category"] == "High"]


# Filter tracks with high energy
def filter_high_energy(df):
    return df[df["energy"] >= 0.8]


# Filter Pop tracks with high popularity
def filter_popular_pop(df):
    return df[(df["genre"] == "Pop") & (df["popularity_category"] == "High")]


# Group tracks by genre and calculate mean popularity and count
def summarize_by_genre(df):
    return (
        df.groupby("genre")["popularity"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )


# Train a linear regression model to predict popularity
def train_linear_model(df, feature):
    # Select the feature as the input
    X = df[[feature]]

    # Select popularity as the output
    y = df["popularity"]

    # Create and train the model
    model = LinearRegression()
    model.fit(X, y)

    return model


# MAIN ANALYSIS

if __name__ == "__main__":

    # Import the Dataset

    df = load_data("data/spotify_artist_streaming_2020_2025.csv")

    # Inspect the Dataset

    print("\nDisplay the first few rows using .head() to get a quick overview:\n")
    print(df.head())

    print(
        "\n\nUse .info() and .describe() to understand data types and summary statistics:\n"
    )

    df.info()

    print("\n")
    print(df.describe())

    print("\n\nCheck for missing values and duplicates (optional):\n")

    # Count missing values in each column
    missing_values = df.isnull().sum()

    if missing_values.sum() == 0:
        print("Missing values: 0\n")
    else:
        print(missing_values[missing_values > 0])

    # Count duplicate rows
    print("Duplicate rows:", df.duplicated().sum())

    # Basic Filtering and Grouping

    print("\n\nApply filters to extract meaningful subsets of the data:\n")

    # Filter 1: Tracks with high popularity
    high_popularity = filter_high_popularity(df)

    print("High popularity tracks:")
    print(high_popularity.head())

    # Filter 2: Tracks with high energy
    high_energy = filter_high_energy(df)

    print("\n\nHigh energy tracks:")
    print(high_energy.head())

    # Filter 3: High popularity pop tracks
    popular_pop = filter_popular_pop(df)

    print("\n\nHigh popularity pop tracks:\n")
    print(popular_pop.head())

    # Group tracks by genre and calculate summary statistics
    print(
        "\n\nGroup tracks by genre and calculate summary statistics for popularity:\n"
    )

    genre_summary = summarize_by_genre(df)

    print(genre_summary)

    # Explore a Machine Learning Algorithm

    print("\nLinear Regression chosen as an ML algorithm.\n")

    # Energy Model

    print("\nEnergy model:\n")
    print("Input: energy\nOutput: popularity")

    # Train model using energy to predict popularity
    energy_model = train_linear_model(df, "energy")

    print("Intercept:", energy_model.intercept_)
    print("Energy coefficient:", energy_model.coef_[0])

    # Danceability Model

    print("\nDanceability model:\n")
    print("Input: danceability\nOutput: popularity")

    # Train model using danceability to predict popularity

    danceability_model = train_linear_model(df, "danceability")

    print("Intercept:", danceability_model.intercept_)
    print("Danceability coefficient:", danceability_model.coef_[0])

    # Visualization

    # Create a scatter plot between energy and popularity
    plt.scatter(df["energy"], df["popularity"], alpha=0.1, s=5)

    # Label the graph
    plt.xlabel("Energy")
    plt.ylabel("Popularity")
    plt.title("Energy vs. Track Popularity")

    # Display the graph
    plt.show()

    # Optional Polars Analysis

    print("\n\nPolars Analysis\n")

    # Load the same dataset using Polars
    df_polars = pl.read_csv("data/spotify_artist_streaming_2020_2025.csv")

    print(df_polars.head())

    # Filter high popularity tracks
    high_popularity_polars = df_polars.filter(pl.col("popularity_category") == "High")

    print("\nHigh popularity tracks:")
    print(high_popularity_polars.head())

    # Group by genre and calculate average popularity
    genre_summary_polars = (
        df_polars.group_by("genre")
        .agg(pl.col("popularity").mean().alias("mean_popularity"))
        .sort("mean_popularity", descending=True)
    )

    print("\nAverage popularity by genre:")
    print(genre_summary_polars)

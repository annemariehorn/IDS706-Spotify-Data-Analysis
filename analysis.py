# Import the Dataset

import pandas as pd
import polars as pl
from sklearn.linear_model import LinearRegression  # Import Linear Regression
import matplotlib.pyplot as plt  # Import Matplotlib

# Inspect the Dataset

df = pd.read_csv("data/spotify_artist_streaming_2020_2025.csv")

print("\nDisplay the first few rows using .head() to get a quick overview:\n")
print(df.head())

print(
    "\n\nUse .info() and .describe() to understand data types and summary statistics:\n"
)
df.info()
print("\n")
print(df.describe())

print("\n\nCheck for missing values and duplicates (optional):\n")

missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    print("Missing values: 0\n")
else:
    print(missing_values[missing_values > 0])

print("Duplicate rows:", df.duplicated().sum())


# Basic Filtering and Grouping

print("\n\nApply filters to extract meaningful subsets of the data:\n")

# Filter 1: Tracks with high popularity
high_popularity = df[df["popularity_category"] == "High"]

print("High popularity tracks:")
print(high_popularity.head())

# Filter 2: Tracks with high energy
high_energy = df[df["energy"] >= 0.8]

print("\n\nHigh energy tracks:")
print(high_energy.head())

# Filter 3: High popularity pop tracks
popular_pop = df[(df["genre"] == "Pop") & (df["popularity_category"] == "High")]

print("\n\nHigh popularity pop tracks:\n")
print(popular_pop.head())

# Use groupby() or equivalent on a selected variable and compute summary statistics (e.g., mean, count):

print("\n\nGroup tracks by genre and calculate summary statistics for popularity:\n")

genre_summary = (
    df.groupby("genre")["popularity"]
    .agg(["mean", "count"])
    .sort_values("mean", ascending=False)
)
print(genre_summary)

# Explore a Machine Learning Algorithm

print("\nLinear Regression chosen as an ML algorithm.\n")

# Begin experimenting with model inputs and outputs.

print("\nEnergy model:\n")

print("Input: energy\nOutput: popularity")

X = df[["energy"]]
y = df["popularity"]

model = LinearRegression()  # Create the Model
model.fit(X, y)  # Train the Model

print("Intercept:", model.intercept_)
print("Energy coefficient:", model.coef_[0])

print("\nDanceability model:\n")

print("Input: danceability\nOutput: popularity")

X = df[["danceability"]]
y = df["popularity"]

model = LinearRegression()  # Create the Model
model.fit(X, y)  # Train the Model

print("Intercept:", model.intercept_)
print("Danceability coefficient:", model.coef_[0])

# Visualization

# Scatter plot between energy and popularity
plt.scatter(df["energy"], df["popularity"], alpha=0.1, s=5)

plt.xlabel("Energy")
plt.ylabel("Popularity")
plt.title("Energy vs. Track Popularity")

plt.show()

# Optional Polars

print("\n\nPolars Analysis\n")

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

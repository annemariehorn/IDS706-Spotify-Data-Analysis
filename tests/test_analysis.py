from analysis import (
    load_data,
    filter_high_popularity,
    summarize_by_genre,
    train_linear_model,
)

DATA_PATH = "data/spotify_artist_streaming_2020_2025.csv"


# Test data loading
def test_load_data():
    df = load_data(DATA_PATH)

    assert not df.empty
    assert "popularity" in df.columns
    assert "energy" in df.columns


# Test high popularity filter
def test_filter_high_popularity():
    df = load_data(DATA_PATH)
    high_popularity = filter_high_popularity(df)

    assert not high_popularity.empty
    assert (high_popularity["popularity_category"] == "High").all()


# Test linear regression model
def test_train_linear_model():
    df = load_data(DATA_PATH)
    model = train_linear_model(df, "energy")

    predictions = model.predict(df[["energy"]].head())

    assert len(predictions) == 5
    assert model.coef_.shape == (1,)


# Test full analysis workflow
def test_full_workflow():
    df = load_data(DATA_PATH)
    high_popularity = filter_high_popularity(df)
    genre_summary = summarize_by_genre(high_popularity)
    model = train_linear_model(df, "energy")
    predictions = model.predict(df[["energy"]].head())

    assert not df.empty
    assert not high_popularity.empty
    assert not genre_summary.empty
    assert len(predictions) == 5

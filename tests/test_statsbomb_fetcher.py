import pandas as pd

from src.data_ingestion.statsbomb_fetcher import fetch_barcelona_matches, fetch_match_events


def test_fetch_barcelona_matches_returns_dataframe():
    df = fetch_barcelona_matches()

    assert isinstance(df, pd.DataFrame), "The fetcher must return a Pandas DataFrame."
    assert not df.empty, "The returned DataFrame should not be empty."

    expected_columns = ["match_id", "match_date", "home_team", "away_team"]
    for col in expected_columns:
        assert col in df.columns, f"Missing expected column: {col}"


def test_fetch_match_events_returns_dataframe():
    events_df = fetch_match_events(match_id=16136)

    assert isinstance(events_df, pd.DataFrame), "Events fetcher must return a DataFrame."
    assert not events_df.empty, "Events DataFrame should not be empty."

    tactical_columns = ["type", "player", "team", "minute"]
    for col in tactical_columns:
        assert col in events_df.columns, f"Missing tactical column: {col}"

import pytest
import pandas as pd
from pathlib import Path
from src.data_ingestion.kaggle_fetcher import load_sqlite_match_data


def test_load_sqlite_match_data_returns_dataframe():
    """
    Test that our SQLite fetcher properly executes the JOIN query and formats the Date.
    """
    db_path = Path("data/raw/european_database.sqlite")

    if db_path.exists():
        df = load_sqlite_match_data(db_path=str(db_path))

        assert isinstance(df, pd.DataFrame), "Must return a Pandas DataFrame."
        assert not df.empty, "DataFrame should not be empty."

        assert 'country' in df.columns, "Missing 'country' from the JOIN."
        assert 'name' in df.columns, "Missing division 'name' from the JOIN."
        assert pd.api.types.is_datetime64_any_dtype(df['Date']), "Date column was not converted to datetime."
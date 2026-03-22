import pytest
import pandas as pd
from pathlib import Path
from src.data_ingestion.player_stats_fetcher import load_player_stats_csv


def test_load_player_stats_csv_returns_dataframe():
    """
    Test that our player stats fetcher properly reads the 2024-2025 Kaggle CSV.
    """
    file_path = Path("data/raw/players_data_light-2024_2025.csv")

    if file_path.exists():
        df = load_player_stats_csv(filepath=str(file_path))

        assert isinstance(df, pd.DataFrame), "Must return a Pandas DataFrame."
        assert not df.empty, "DataFrame should not be empty."

        expected_columns = ['Player', 'Squad', 'Comp', 'xG', 'PrgP', 'Tkl+Int']
        for col in expected_columns:
            assert col in df.columns, f"Missing expected advanced metric column: {col}"
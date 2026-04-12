import pandas as pd

from src.match_prediction.feature_pipeline import build_historical_features


def test_build_historical_features_prevents_time_travel_and_calculates_advanced_stats():
    mock_history = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-01-01", "2023-01-08", "2023-01-15"]),
            "HomeTeam": ["Chelsea", "AwayFC", "Chelsea"],
            "AwayTeam": ["HomeFC", "Chelsea", "Arsenal"],
            "FTHG": [2, 1, 0],
            "FTAG": [0, 1, 0],
            "FTR": ["H", "D", "D"],
        }
    )
    ml_dataset = build_historical_features(mock_history)
    assert isinstance(ml_dataset, pd.DataFrame), "Must return a Pandas DataFrame."
    match_3_features = ml_dataset.iloc[-1]
    assert match_3_features["home_recent_goals_scored"] == 3, "Calculated future goals."
    assert match_3_features["home_recent_points"] == 4, "Points calculated incorrectly."
    assert "home_recent_goal_diff" in match_3_features, "Missing Goal Difference feature."
    assert "home_recent_clean_sheets" in match_3_features, "Missing Clean Sheets feature."
    assert match_3_features["home_recent_goal_diff"] == 2, "Failed to calculate rolling Goal Difference."
    assert match_3_features["home_recent_clean_sheets"] == 1, "Failed to calculate rolling Clean Sheets."

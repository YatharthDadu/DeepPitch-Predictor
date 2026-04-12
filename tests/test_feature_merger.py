import pandas as pd

from src.match_prediction.feature_merger import combine_prediction_features


def test_combine_prediction_features_returns_single_row_dataframe():
    home_form = {
        "recent_goals_scored": 10,
        "recent_goals_conceded": 2,
        "recent_points": 13,
        "recent_form_rating": 0.866,
    }

    away_form = {"recent_goals_scored": 4, "recent_goals_conceded": 8, "recent_points": 4, "recent_form_rating": 0.266}

    historical_h2h = {"h2h_home_win_rate": 0.65, "h2h_away_win_rate": 0.15, "h2h_draw_rate": 0.20}

    features_df = combine_prediction_features(
        home_team="Man City",
        away_team="West Ham",
        historical_stats=historical_h2h,
        home_form=home_form,
        away_form=away_form,
    )

    # Assert
    assert isinstance(features_df, pd.DataFrame), "Must return a Pandas DataFrame."
    assert len(features_df) == 1, "Inference DataFrame must be exactly 1 row."
    assert "home_recent_goals_scored" in features_df.columns, "Home prefix missing."
    assert "away_recent_points" in features_df.columns, "Away prefix missing."
    assert "h2h_home_win_rate" in features_df.columns, "Historical stats missing."
    assert features_df.iloc[0]["home_recent_goals_scored"] == 10

import pandas as pd

from src.match_prediction.historical_extractor import calculate_h2h_stats


def test_calculate_h2h_stats_returns_correct_rates():
    mock_history = pd.DataFrame(
        {
            "HomeTeam": ["Man City", "Arsenal", "Man City", "Arsenal", "Man City"],
            "AwayTeam": ["Arsenal", "Man City", "Arsenal", "Man City", "Arsenal"],
            "FTR": ["H", "H", "D", "A", "H"],
        }
    )

    h2h_stats = calculate_h2h_stats(history_df=mock_history, target_home_team="Man City", target_away_team="Arsenal")

    assert isinstance(h2h_stats, dict), "Must return a dictionary of features."

    assert h2h_stats["h2h_home_win_rate"] == 0.60

    assert h2h_stats["h2h_away_win_rate"] == 0.20

    assert h2h_stats["h2h_draw_rate"] == 0.20


def test_calculate_h2h_stats_handles_no_history():
    empty_history = pd.DataFrame(columns=["HomeTeam", "AwayTeam", "FTR"])
    h2h_stats = calculate_h2h_stats(empty_history, "Man City", "Wrexham")
    assert h2h_stats["h2h_home_win_rate"] == 0.333
    assert h2h_stats["h2h_draw_rate"] == 0.334

import pytest

from src.match_prediction.user_input import parse_recent_form


def test_parse_recent_form_returns_correct_features():
    user_input_string = "2-0, 3-1, 1-1, 0-2, 1-2"
    features = parse_recent_form(user_input_string)

    assert isinstance(features, dict), "Must return a dictionary of features."

    assert features["recent_goals_scored"] == 7, "Failed to calculate Goals Scored."

    assert features["recent_goals_conceded"] == 6, "Failed to calculate Goals Conceded."

    assert features["recent_points"] == 7, "Failed to calculate Points correctly (Win=3, Draw=1, Loss=0)."


def test_parse_recent_form_handles_invalid_input():
    with pytest.raises(ValueError):
        parse_recent_form("Won 2-0, Lost, 1-1")

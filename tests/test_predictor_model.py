import pytest
import pandas as pd
from src.match_prediction.predictor_model import MatchPredictor


def test_match_predictor_training_and_inference():
    train_features = pd.DataFrame({
        'home_recent_points': [15, 0, 5, 12, 1],
        'away_recent_points': [0, 15, 5, 2, 12]
    })
    train_labels = pd.Series([2, 0, 1, 2, 0])
    predictor = MatchPredictor()
    predictor.train(X_train=train_features, y_train=train_labels)
    inference_row = pd.DataFrame({'home_recent_points': [10], 'away_recent_points': [3]})
    probabilities = predictor.predict_probabilities(inference_row)
    assert isinstance(probabilities, dict), "Output must be a dictionary."
    assert 'Home Win' in probabilities, "Missing Home Win key."
    assert 'Draw' in probabilities, "Missing Draw key."
    assert 'Away Win' in probabilities, "Missing Away Win key."
    total_prob = sum(probabilities.values())
    assert 0.99 <= total_prob <= 1.01, f"Probabilities must sum to 100%, got {total_prob}"
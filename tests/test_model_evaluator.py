import pandas as pd

from src.match_prediction.model_evaluator import evaluate_model
from src.match_prediction.predictor_model import MatchPredictor


def test_evaluate_model_returns_metrics_report():
    import numpy as np

    np.random.seed(42)

    mock_features = pd.DataFrame(
        {
            "h2h_home_win_rate": np.random.uniform(0.5, 1.0, 200),
            "home_recent_points": np.random.randint(10, 15, 200),
            "away_recent_points": np.random.randint(0, 5, 200),
        }
    )
    mock_labels = pd.Series(np.random.choice([0, 1, 2], size=200, p=[0.1, 0.2, 0.7]))

    engine = MatchPredictor()
    report = evaluate_model(engine, mock_features, mock_labels)
    assert isinstance(report, dict), "Evaluator must return a dictionary report."
    assert "accuracy" in report, "Missing Accuracy score."
    assert 0.0 <= report["accuracy"] <= 1.0, "Accuracy must be a percentage between 0 and 1."
    assert "confusion_matrix" in report, "Missing Confusion Matrix."
    assert len(report["confusion_matrix"]) == 3, "Matrix should be 3x3 (Win/Draw/Loss)."
    assert "feature_importances" in report, "Missing Feature Importances."
    assert "home_recent_points" in report["feature_importances"], "Missing specific feature in importance mapping."

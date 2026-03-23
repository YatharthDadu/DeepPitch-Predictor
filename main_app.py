import logging
import pandas as pd

# Import our custom ECC modules
from src.match_prediction.user_input import parse_recent_form
from src.match_prediction.feature_merger import combine_prediction_features
from src.match_prediction.predictor_model import MatchPredictor

# ECC Rule: Centralized Application Logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def bootstrap_synthetic_training(predictor: MatchPredictor):
    """
    TEMPORARY: Generates a small batch of synthetic data to 'warm up' the XGBoost model
    so it can make predictions today. In a production environment, this data
    would come from querying our Kaggle SQLite database.
    """
    print(">> Winding up the XGBoost Engine (Training on synthetic historical data)...")
    import numpy as np
    np.random.seed(42)

    X_train = pd.DataFrame({
        'h2h_home_win_rate': np.random.uniform(0, 1, 100),
        'h2h_away_win_rate': np.random.uniform(0, 1, 100),
        'h2h_draw_rate': np.random.uniform(0, 1, 100),
        'home_recent_goals_scored': np.random.randint(0, 15, 100),
        'home_recent_goals_conceded': np.random.randint(0, 15, 100),
        'home_recent_points': np.random.randint(0, 15, 100),
        'home_recent_form_rating': np.random.uniform(0, 1, 100),
        'away_recent_goals_scored': np.random.randint(0, 15, 100),
        'away_recent_goals_conceded': np.random.randint(0, 15, 100),
        'away_recent_points': np.random.randint(0, 15, 100),
        'away_recent_form_rating': np.random.uniform(0, 1, 100)
    })

    y_train = pd.Series(np.random.choice([0, 1, 2], size=100))

    predictor.train(X_train, y_train)
    print(">> Engine Ready.\n")


def run_cli():
    print("==================================================")
    print("   PITCH-PREDICT AI : MATCH OUTCOME ENGINE        ")
    print("==================================================")

    engine = MatchPredictor()
    bootstrap_synthetic_training(engine)

    while True:
        try:
            print("\n--- NEW PREDICTION ---")
            home_team = input("Enter Home Team name (or 'q' to quit): ").strip()
            if home_team.lower() == 'q':
                break

            home_form_str = input(
                f"Enter {home_team}'s last 5 matches (Format: GF-GC, e.g., '2-0, 1-1, 0-1, 3-2, 2-2'): ")

            away_team = input("Enter Away Team name: ").strip()
            away_form_str = input(f"Enter {away_team}'s last 5 matches (Format: GF-GC): ")

            home_features = parse_recent_form(home_form_str)
            away_features = parse_recent_form(away_form_str)

            mock_h2h = {'h2h_home_win_rate': 0.45, 'h2h_away_win_rate': 0.35, 'h2h_draw_rate': 0.20}

            inference_row = combine_prediction_features(
                home_team, away_team, mock_h2h, home_features, away_features
            )

            ml_ready_row = inference_row.drop(columns=['home_team', 'away_team'])

            # 4. Predict!
            print("\n[AI] Analyzing Form, Historical Data, and Momentum...")
            probabilities = engine.predict_probabilities(ml_ready_row)

            print(f"\n🏆 MATCH PREDICTION: {home_team} vs {away_team}")
            print(f"   Home Win ({home_team}):  {probabilities['Home Win'] * 100:.1f}%")
            print(f"   Draw:               {probabilities['Draw'] * 100:.1f}%")
            print(f"   Away Win ({away_team}):  {probabilities['Away Win'] * 100:.1f}%")
            print("==================================================\n")

        except ValueError as ve:
            print(f"\n[INPUT ERROR] {ve}. Let's try that again.")
        except Exception as e:
            print(f"\n[SYSTEM ERROR] {e}")


if __name__ == "__main__":
    run_cli()
import logging
import pandas as pd
from pathlib import Path

from src.data_ingestion.kaggle_fetcher import load_sqlite_match_data
from src.match_prediction.historical_extractor import calculate_h2h_stats
from src.match_prediction.user_input import parse_recent_form
from src.match_prediction.feature_merger import combine_prediction_features
from src.match_prediction.predictor_model import MatchPredictor
from src.match_prediction.model_evaluator import evaluate_model
from src.match_prediction.feature_pipeline import build_historical_features

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def prepare_training_data(ml_df: pd.DataFrame) -> tuple:

    ml_df['home_recent_form_rating'] = round(ml_df['home_recent_points'] / 15.0, 3)
    ml_df['away_recent_form_rating'] = round(ml_df['away_recent_points'] / 15.0, 3)

    features = [
        'home_recent_goals_scored', 'home_recent_goals_conceded', 'home_recent_points', 'home_recent_form_rating',
        'home_recent_goal_diff', 'home_recent_clean_sheets',
        'away_recent_goals_scored', 'away_recent_goals_conceded', 'away_recent_points', 'away_recent_form_rating',
        'away_recent_goal_diff', 'away_recent_clean_sheets'
    ]

    X = ml_df[features].copy()


    X['h2h_home_win_rate'] = 0.45
    X['h2h_away_win_rate'] = 0.30
    X['h2h_draw_rate'] = 0.25

    y = ml_df['Target_Label']
    return X, y


def run_cli():
    print("==================================================")
    print("   PITCH-PREDICT AI : MATCH OUTCOME ENGINE        ")
    print("==================================================")

    db_path = Path("data/raw/european_database.sqlite")
    if not db_path.exists():
        print(f"[SYSTEM ERROR] Could not find database at {db_path}.")
        return

    print(">> Connecting to SQLite Historical Database...")
    history_df = load_sqlite_match_data(str(db_path))

    print(">> Engineering time-sealed historical momentum features (Preventing Data Leakage)...")
    ml_df = build_historical_features(history_df, rolling_window=5)
    X, y = prepare_training_data(ml_df)

    print(">> Initiating Train/Test Split & Model Evaluation...")
    engine = MatchPredictor()
    report = evaluate_model(engine, X, y)

    print("\n--- 🧠 AI EVALUATION REPORT ---")
    print(f"Accuracy on Hidden Test Data: {report['accuracy'] * 100:.1f}%")
    print("\nTop 3 Most Important Stats to the AI:")
    top_3 = list(report['feature_importances'].items())[:3]
    for stat, importance in top_3:
        print(f"  - {stat}: {importance * 100:.1f}% weight")
    print("--------------------------------\n")

    print(">> Retraining XGBoost brain on 100% of historical data for maximum accuracy...")
    engine.train(X, y)
    print(">> Engine Ready.\n")

    while True:
        try:
            print("==================================================")
            home_team = input("Enter Home Team name (or 'q' to quit): ").strip()
            if home_team.lower() == 'q':
                break

            home_form_str = input(f"Enter {home_team}'s last 5 matches (Format: GF-GC): ")
            away_team = input("Enter Away Team name: ").strip()
            away_form_str = input(f"Enter {away_team}'s last 5 matches (Format: GF-GC): ")

            home_features = parse_recent_form(home_form_str)
            away_features = parse_recent_form(away_form_str)

            print("\n[AI] Querying historical archives for Head-to-Head data...")
            real_h2h_stats = calculate_h2h_stats(history_df, home_team, away_team)

            inference_row = combine_prediction_features(
                home_team, away_team, real_h2h_stats, home_features, away_features
            )

            ml_ready_row = inference_row.drop(columns=['home_team', 'away_team'])
            ml_ready_row = ml_ready_row[X.columns]

            print("[AI] Analyzing Form, Historical Data, and Momentum...")
            probabilities = engine.predict_probabilities(ml_ready_row)

            print(f"\n🏆 MATCH PREDICTION: {home_team} vs {away_team}")
            print(f"   Home Win ({home_team}):  {probabilities['Home Win'] * 100:.1f}%")
            print(f"   Draw:               {probabilities['Draw'] * 100:.1f}%")
            print(f"   Away Win ({away_team}):  {probabilities['Away Win'] * 100:.1f}%")

        except ValueError as ve:
            print(f"\n[INPUT ERROR] {ve}. Let's try that again.")
        except Exception as e:
            print(f"\n[SYSTEM ERROR] {e}")


if __name__ == "__main__":
    run_cli()
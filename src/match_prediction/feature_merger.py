import logging
import pandas as pd

logger = logging.getLogger(__name__)

def combine_prediction_features(
        home_team: str,
        away_team: str,
        historical_stats: dict,
        home_form: dict,
        away_form: dict
) -> pd.DataFrame:

    try:
        logger.info(f"Merging features for {home_team} (Home) vs {away_team} (Away)...")

        merged_features = {
            'home_team': home_team,
            'away_team': away_team
        }

        for key, value in historical_stats.items():
            merged_features[key] = value

        for key, value in home_form.items():
            merged_features[f"home_{key}"] = value

        for key, value in away_form.items():
            merged_features[f"away_{key}"] = value
        features_df = pd.DataFrame([merged_features])

        logger.info(f"Successfully created inference DataFrame with {len(features_df.columns)} features.")
        return features_df

    except Exception as e:
        logger.error(f"Failed to merge features for {home_team} vs {away_team}. Error: {str(e)}")
        raise
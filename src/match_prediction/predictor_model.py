import logging
import pandas as pd
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

class MatchPredictor:

    def __init__(self):
        logger.info("Initializing the XGBoost Match Predictor...")
        self.model = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42
        )

        self.is_trained = False
        self.class_mapping = {
            0: 'Away Win',
            1: 'Draw',
            2: 'Home Win'
        }

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        try:
            logger.info(f"Starting model training on {len(X_train)} historical matches...")

            self.model.fit(X_train, y_train)
            self.is_trained = True

            logger.info("Model training completed successfully.")

        except Exception as e:
            logger.error(f"Failed to train the XGBoost model. Error: {str(e)}")
            raise

    def predict_probabilities(self, features_df: pd.DataFrame) -> dict:
        try:
            if not self.is_trained:
                raise ValueError("The MatchPredictor must be trained before making predictions.")

            if len(features_df) != 1:
                logger.warning(f"Expected 1 inference row, but received {len(features_df)}.")

            logger.info("Calculating outcome probabilities...")
            probabilities_array = self.model.predict_proba(features_df)[0]

            result_dict = {
                self.class_mapping[0]: round(float(probabilities_array[0]), 3),
                self.class_mapping[1]: round(float(probabilities_array[1]), 3),
                self.class_mapping[2]: round(float(probabilities_array[2]), 3)
            }

            logger.info(f"Prediction successful: {result_dict}")
            return result_dict

        except Exception as e:
            logger.error(f"Failed to generate match predictions. Error: {str(e)}")
            raise
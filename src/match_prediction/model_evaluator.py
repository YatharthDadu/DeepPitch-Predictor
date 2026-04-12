import logging

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def evaluate_model(predictor, X: pd.DataFrame, y: pd.Series) -> dict:

    try:
        logger.info(f"Starting model evaluation on {len(X)} historical records...")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        logger.info(f"Split data: {len(X_train)} training rows, {len(X_test)} testing rows.")

        predictor.train(X_train, y_train)

        logger.info("Predicting outcomes for the hidden test set...")
        predictions = predictor.model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        raw_cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
        cm_list = raw_cm.tolist()

        importances = predictor.model.feature_importances_
        feature_importance_dict = {col: float(imp) for col, imp in zip(X.columns, importances)}

        feature_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))

        report = {
            "accuracy": float(accuracy),
            "confusion_matrix": cm_list,
            "feature_importances": feature_importance_dict,
        }

        logger.info(f"Evaluation Complete! AI Accuracy: {accuracy * 100:.1f}%")
        return report

    except Exception as e:
        logger.error(f"Failed to evaluate the model. Error: {str(e)}")
        raise

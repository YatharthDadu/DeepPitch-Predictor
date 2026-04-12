import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_player_stats_csv(filepath: str) -> pd.DataFrame:
    """
    Loads season-long player statistics from a local Kaggle CSV file.
    """
    try:
        logger.info(f"Loading local player statistics from {filepath}...")

        raw_df = pd.read_csv(filepath)

        if raw_df.empty:
            logger.warning("The CSV was read successfully but contains no data.")
            return pd.DataFrame()

        processed_df = raw_df.copy(deep=True)

        processed_df.columns = processed_df.columns.str.strip()

        logger.info(f"Successfully loaded stats for {len(processed_df)} players.")
        return processed_df

    except FileNotFoundError:
        logger.error(f"Could not find the file at {filepath}. Please ensure it is placed in data/raw/")
        raise
    except Exception as e:
        logger.error(f"Failed to load player stats CSV. Error: {str(e)}")
        raise

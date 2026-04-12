import logging
import sqlite3

import pandas as pd

logger = logging.getLogger(__name__)


def load_sqlite_match_data(db_path: str) -> pd.DataFrame:
    """
    Loads historical match and division data from a local Kaggle SQLite database.
    """
    try:
        logger.info(f"Connecting to local SQLite database at {db_path}...")

        with sqlite3.connect(db_path) as conn:
            query = """
                    SELECT m.*, \
                           d.name, \
                           d.country
                    FROM matchs m
                             JOIN divisions d ON d.division == m.Div \
                    """

            logger.info("Executing JOIN query on 'matchs' and 'divisions' tables...")
            raw_df = pd.read_sql_query(query, conn)

        if raw_df.empty:
            logger.warning("Query executed successfully but returned no data.")
            return pd.DataFrame()

        processed_df = raw_df.copy(deep=True)

        processed_df = processed_df.assign(Date=lambda x: pd.to_datetime(x["Date"]))

        logger.info(f"Successfully loaded {len(processed_df)} match records.")
        return processed_df

    except Exception as e:
        logger.error(f"Failed to load data from SQLite DB. Error: {str(e)}")
        raise

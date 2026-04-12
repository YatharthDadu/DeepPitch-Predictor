import logging

import pandas as pd
from dotenv import load_dotenv
from statsbombpy import sb

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fetch_barcelona_matches() -> pd.DataFrame:
    """
    Fetches historical match data for FC Barcelona from StatsBomb's open dataset.
    Uses La Liga (competition_id=11) and the 2018/2019 season (season_id=4) as our baseline.
    """
    try:
        logger.info("Initiating fetch for La Liga matches from StatsBomb API...")

        matches_df = sb.matches(competition_id=11, season_id=4)

        if matches_df.empty:
            logger.warning("StatsBomb API returned an empty DataFrame.")
            return pd.DataFrame()

        processed_df = matches_df.copy(deep=True)

        barca_mask = (processed_df["home_team"] == "Barcelona") | (processed_df["away_team"] == "Barcelona")
        barca_matches_df = processed_df[barca_mask].copy(deep=True)

        logger.info(f"Successfully fetched and isolated {len(barca_matches_df)} Barcelona matches.")
        return barca_matches_df

    except Exception as e:
        logger.error(f"Failed to fetch match data from StatsBomb API. Error details: {str(e)}")
        raise


def fetch_match_events(match_id: int) -> pd.DataFrame:
    """
    Fetches detailed event data (passes, shots, tactics) for a specific match_id.
    """
    try:
        logger.info(f"Fetching event data for match_id: {match_id}...")

        events_df = sb.events(match_id=match_id)

        if events_df.empty:
            logger.warning(f"No events found for match_id: {match_id}.")
            return pd.DataFrame()

        processed_events = events_df.copy(deep=True)

        logger.info(f"Successfully fetched {len(processed_events)} events for match {match_id}.")
        return processed_events

    except Exception as e:
        logger.error(f"Failed to fetch events for match_id {match_id}. Error: {str(e)}")
        raise

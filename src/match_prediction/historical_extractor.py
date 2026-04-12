import logging
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_h2h_stats(history_df: pd.DataFrame, target_home_team: str, target_away_team: str) -> dict:
    try:
        logger.info(f"Extracting historical H2H stats for {target_home_team} vs {target_away_team}...")
        required_cols = {'HomeTeam', 'AwayTeam', 'FTR'}
        if history_df.empty or not required_cols.issubset(history_df.columns):
            logger.warning("History DataFrame is empty or invalid. Defaulting to 33% split.")
            return _default_split()

        scenario_a = history_df[
            (history_df['HomeTeam'] == target_home_team) &
            (history_df['AwayTeam'] == target_away_team)
            ]

        scenario_b = history_df[
            (history_df['HomeTeam'] == target_away_team) &
            (history_df['AwayTeam'] == target_home_team)
            ]

        total_matches = len(scenario_a) + len(scenario_b)

        if total_matches == 0:
            logger.info("Teams have no historical matchups in this dataset. Defaulting to 33% split.")
            return _default_split()

        home_wins_a = (scenario_a['FTR'] == 'H').sum()
        home_wins_b = (scenario_b['FTR'] == 'A').sum()
        target_home_wins = home_wins_a + home_wins_b

        away_wins_a = (scenario_a['FTR'] == 'A').sum()
        away_wins_b = (scenario_b['FTR'] == 'H').sum()
        target_away_wins = away_wins_a + away_wins_b

        draws = (scenario_a['FTR'] == 'D').sum() + (scenario_b['FTR'] == 'D').sum()

        h2h_stats = {
            'h2h_home_win_rate': round(target_home_wins / total_matches, 3),
            'h2h_away_win_rate': round(target_away_wins / total_matches, 3),
            'h2h_draw_rate': round(draws / total_matches, 3)
        }

        logger.info(f"H2H Stats calculated from {total_matches} matches: {h2h_stats}")
        return h2h_stats

    except Exception as e:
        logger.error(f"Failed to calculate H2H stats. Error: {str(e)}")
        raise


def _default_split() -> dict:
    return {
        'h2h_home_win_rate': 0.333,
        'h2h_away_win_rate': 0.333,
        'h2h_draw_rate': 0.334
    }
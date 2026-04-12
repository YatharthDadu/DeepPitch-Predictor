import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_historical_features(match_history: pd.DataFrame, rolling_window: int = 5) -> pd.DataFrame:

    try:
        logger.info("Building time-sealed historical features with Advanced Stats...")

        df = match_history.copy(deep=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        home_records = df[["Date", "HomeTeam", "FTHG", "FTAG", "FTR"]].copy()
        home_records.columns = ["Date", "Team", "GF", "GA", "Result"]
        home_records["Points"] = home_records["Result"].map({"H": 3, "D": 1, "A": 0})

        away_records = df[["Date", "AwayTeam", "FTAG", "FTHG", "FTR"]].copy()
        away_records.columns = ["Date", "Team", "GF", "GA", "Result"]
        away_records["Points"] = away_records["Result"].map({"A": 3, "D": 1, "H": 0})

        team_history = pd.concat([home_records, away_records]).sort_values(["Team", "Date"])

        team_history["GoalDiff"] = team_history["GF"] - team_history["GA"]
        team_history["CleanSheet"] = (team_history["GA"] == 0).astype(int)

        def calculate_rolling(series):
            return series.shift(1).rolling(rolling_window, min_periods=1).sum()

        team_history["recent_goals_scored"] = team_history.groupby("Team")["GF"].transform(calculate_rolling)
        team_history["recent_goals_conceded"] = team_history.groupby("Team")["GA"].transform(calculate_rolling)
        team_history["recent_points"] = team_history.groupby("Team")["Points"].transform(calculate_rolling)
        team_history["recent_goal_diff"] = team_history.groupby("Team")["GoalDiff"].transform(calculate_rolling)
        team_history["recent_clean_sheets"] = team_history.groupby("Team")["CleanSheet"].transform(calculate_rolling)
        team_history = team_history.fillna(0)
        home_cols = [
            "Date",
            "Team",
            "recent_goals_scored",
            "recent_goals_conceded",
            "recent_points",
            "recent_goal_diff",
            "recent_clean_sheets",
        ]
        home_features = team_history[home_cols].copy()
        home_features.columns = [
            "Date",
            "HomeTeam",
            "home_recent_goals_scored",
            "home_recent_goals_conceded",
            "home_recent_points",
            "home_recent_goal_diff",
            "home_recent_clean_sheets",
        ]
        away_features = team_history[home_cols].copy()
        away_features.columns = [
            "Date",
            "AwayTeam",
            "away_recent_goals_scored",
            "away_recent_goals_conceded",
            "away_recent_points",
            "away_recent_goal_diff",
            "away_recent_clean_sheets",
        ]
        ml_df = df.merge(home_features, on=["Date", "HomeTeam"], how="left")
        ml_df = ml_df.merge(away_features, on=["Date", "AwayTeam"], how="left")
        ml_df["Target_Label"] = ml_df["FTR"].map({"A": 0, "D": 1, "H": 2})
        ml_df = ml_df.dropna(subset=["Target_Label"]).reset_index(drop=True)

        logger.info(f"Successfully engineered ML dataset with {len(ml_df)} safe records.")
        return ml_df

    except Exception as e:
        logger.error(f"Failed to build historical features. Error: {str(e)}")
        raise

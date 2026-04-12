import logging

logger = logging.getLogger(__name__)


def parse_recent_form(score_string: str) -> dict:
    try:
        scores = [s.strip() for s in score_string.split(",")]

        if len(scores) != 5:
            raise ValueError(f"Expected exactly 5 recent matches, but got {len(scores)}.")

        gf_total = 0
        gc_total = 0
        points_total = 0
        gd_total = 0
        cs_total = 0

        for score in scores:
            if score.count("-") != 1:
                raise ValueError(f"Invalid score format: '{score}'. Must be 'GF-GC' (e.g., '2-1').")

            gf_str, gc_str = score.split("-")

            if not (gf_str.isdigit() and gc_str.isdigit()):
                raise ValueError(f"Scores must be numbers. Invalid input: '{score}'.")

            gf = int(gf_str)
            gc = int(gc_str)

            gf_total += gf
            gc_total += gc

            gd_total += gf - gc
            if gc == 0:
                cs_total += 1
            if gf > gc:
                points_total += 3
            elif gf == gc:
                points_total += 1
        features = {
            "recent_goals_scored": gf_total,
            "recent_goals_conceded": gc_total,
            "recent_points": points_total,
            "recent_form_rating": round(points_total / 15.0, 3),
            "recent_goal_diff": gd_total,  # NEW
            "recent_clean_sheets": cs_total,  # NEW
        }

        logger.info(f"Successfully parsed user input into advanced features: {features}")
        return features

    except Exception as e:
        logger.error(f"Failed to parse recent form. Input: {score_string}. Error: {str(e)}")
        raise

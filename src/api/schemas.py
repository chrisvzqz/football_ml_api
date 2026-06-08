from pydantic import BaseModel

class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    matchday: int
    season_start_year: int
    home_points_last_5: int
    away_points_last_5: int
    home_avg_goals_scored_last_5: float
    away_avg_goals_scored_last_5: float
    home_avg_goals_conceded_last_5: float
    away_avg_goals_conceded_last_5: float
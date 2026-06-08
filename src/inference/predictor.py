from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_model.joblib"


class MatchPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = joblib.load(model_path)

    def predict(self, match_df: pd.DataFrame):
        prediction = self.model.predict(match_df)
        return prediction[0] 
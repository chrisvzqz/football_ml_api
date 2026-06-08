from fastapi import FastAPI
import pandas as pd
from src.inference.predictor import MatchPredictor
from src.api.schemas import MatchRequest

app = FastAPI()
predictor = MatchPredictor()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(match: MatchRequest):
    match_df = pd.DataFrame(match.model_dump(), index=[0])
    prediction = predictor.predict(match_df)
    return {"predicted_winner": prediction}
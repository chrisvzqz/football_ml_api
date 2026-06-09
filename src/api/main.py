from fastapi import FastAPI
import pandas as pd
from src.inference.predictor import MatchPredictor
from src.api.schemas import MatchRequest
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

predictor = MatchPredictor()
logger.info("Model loaded successfully.")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(match: MatchRequest):
    match_df = pd.DataFrame(match.model_dump(), index=[0])
    logger.info(
        f"Prediction requested: {match.home_team} vs {match.away_team}"
    )

    try:
        prediction = predictor.predict(match_df)
        logger.info(f"Prediction result: {prediction}")
        return {"predicted_winner": prediction}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise # Lanza la misma excepción que acaba de capturar
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "matches_ml_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"


def load_csv(file_path):
    return pd.read_csv(file_path)


def split_features_target(df, target_column="winner"):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                ["home_team", "away_team"],
            )
        ],
        remainder="passthrough",
    )

    model = LogisticRegression(max_iter=2000, random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline


def run_training_pipeline(file_path=DATA_PATH):
    df = load_csv(file_path)

    df = df.drop(columns=["utc_date"])

    train_set, test_set = train_test_split(
        df,
        test_size=0.2,
        stratify=df["winner"],
        random_state=42,
    )

    X_train, y_train = split_features_target(train_set)
    X_test, y_test = split_features_target(test_set)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = MODELS_DIR / f"logistic_regression_model_{timestamp}.joblib"

    joblib.dump(pipeline, model_path)

    print(f"Model saved at: {model_path}")


if __name__ == "__main__":
    run_training_pipeline()
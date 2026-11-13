"""Train the Decision Tree pipeline using the same setup as gaming.ipynb."""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "online_gaming_behavior_dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "final_engagement_tree_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

CATEGORICAL_FEATURES = ["Gender", "Location", "GameGenre", "GameDifficulty"]
NUMERIC_FEATURES = [
    "Age", "PlayTimeHours", "InGamePurchases", "SessionsPerWeek",
    "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked",
]
TARGET = "EngagementLevel"
REQUIRED_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place online_gaming_behavior_dataset.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"The dataset is missing required columns: {missing}")

    # Match the notebook's modelling approach: remove ID and split 80/20.
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    y = df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )

    # Hyperparameters from the final pruned pipeline in gaming.ipynb.
    classifier = DecisionTreeClassifier(
        random_state=42,
        criterion="entropy",
        max_depth=12,
        min_samples_split=2,
        min_samples_leaf=10,
        ccp_alpha=0.000357,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", classifier)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    report = classification_report(y_test, predictions, output_dict=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    metrics = {
        "test_accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "test_rows": int(len(y_test)),
        "class_report": report,
        "model": "DecisionTreeClassifier",
        "parameters": classifier.get_params(),
    }
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Test accuracy: {accuracy:.4%}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()

"""Train and compare candidate models for KOI disposition classification.

Three models are compared, each chosen for a specific reason:
- Logistic Regression: simple, interpretable baseline
- Random Forest: handles non-linear feature interactions, robust to outliers
- Gradient Boosting: often the strongest of the three on tabular data

The best-performing model (by test accuracy) is saved to disk.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocess import CORE_FEATURES, TARGET_COLUMN

PROCESSED_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "koi_clean.csv"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "model.pkl"
MODEL_COMPARISON_PATH = (
    Path(__file__).resolve().parent.parent / "reports" / "model_comparison.csv"
)

RANDOM_STATE = 42


def load_clean_data(processed_data_path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned KOI dataset produced by preprocess.py."""
    return pd.read_csv(processed_data_path)


def split_data(df: pd.DataFrame):
    """Split features and target into train/test sets."""
    X = df[CORE_FEATURES]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def get_candidate_models() -> dict:
    """Return the models to compare, each wrapped in a scaling pipeline."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]
        ),
    }


def evaluate_models(models: dict, X_train, X_test, y_train, y_test) -> dict:
    """Fit each model, print its classification report, and return test accuracies."""
    accuracies = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        accuracies[name] = accuracy

        print(f"\n=== {name} ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, predictions))

    return accuracies


def save_model_comparison(accuracies: dict, output_path: Path = MODEL_COMPARISON_PATH) -> None:
    """Save each model's test accuracy so the dashboard can display it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df = pd.DataFrame(
        sorted(accuracies.items()), columns=["model", "accuracy"]
    )
    comparison_df.to_csv(output_path, index=False)
    print(f"Saved model comparison to {output_path}")


if __name__ == "__main__":
    df = load_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)

    models = get_candidate_models()
    accuracies = evaluate_models(models, X_train, X_test, y_train, y_test)
    save_model_comparison(accuracies)

    best_model_name = max(accuracies, key=accuracies.get)
    best_model = models[best_model_name]

    print(f"\nBest model: {best_model_name} (accuracy: {accuracies[best_model_name]:.4f})")

    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best model to {MODEL_PATH}")

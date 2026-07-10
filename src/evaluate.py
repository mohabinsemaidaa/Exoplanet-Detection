"""Evaluate the saved model: confusion matrix and feature importance.

Produces two plots in reports/ that document how well the final model
performs and which features it relies on most.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from src.preprocess import CORE_FEATURES
from src.train import MODEL_PATH, load_clean_data, split_data

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.png"


def plot_confusion_matrix(model, X_test, y_test, output_path: Path = CONFUSION_MATRIX_PATH) -> None:
    """Plot and save the confusion matrix for the model's test predictions."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, xticks_rotation=45)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved confusion matrix to {output_path}")


def plot_feature_importance(model, output_path: Path = FEATURE_IMPORTANCE_PATH) -> None:
    """Plot and save feature importances, if the model supports them."""
    trained_model = model.named_steps["model"]

    if not hasattr(trained_model, "feature_importances_"):
        print(f"{type(trained_model).__name__} has no feature_importances_; skipping plot")
        return

    importances = trained_model.feature_importances_
    order = importances.argsort()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([CORE_FEATURES[i] for i in order], importances[order])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved feature importance plot to {output_path}")


if __name__ == "__main__":
    df = load_clean_data()
    _, X_test, _, y_test = split_data(df)

    model = joblib.load(MODEL_PATH)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(model, X_test, y_test)
    plot_feature_importance(model)

import pandas as pd

from src.evaluate import plot_confusion_matrix, plot_feature_importance
from src.preprocess import CORE_FEATURES, TARGET_COLUMN
from src.train import get_candidate_models, split_data


def _sample_dataframe(n_per_class=20):
    classes = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]
    rows = []
    for i, label in enumerate(classes):
        for j in range(n_per_class):
            row = {feature: float(i * 10 + j) for feature in CORE_FEATURES}
            row[TARGET_COLUMN] = label
            rows.append(row)
    return pd.DataFrame(rows)


def test_plot_confusion_matrix_creates_file(tmp_path):
    df = _sample_dataframe()
    X_train, X_test, y_train, y_test = split_data(df)

    model = get_candidate_models()["random_forest"]
    model.fit(X_train, y_train)

    output_path = tmp_path / "confusion_matrix.png"
    plot_confusion_matrix(model, X_test, y_test, output_path=output_path)

    assert output_path.exists()


def test_plot_feature_importance_creates_file(tmp_path):
    df = _sample_dataframe()
    X_train, X_test, y_train, y_test = split_data(df)

    model = get_candidate_models()["random_forest"]
    model.fit(X_train, y_train)

    output_path = tmp_path / "feature_importance.png"
    plot_feature_importance(model, output_path=output_path)

    assert output_path.exists()

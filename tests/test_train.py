import pandas as pd

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


def test_split_data_is_stratified():
    df = _sample_dataframe()

    X_train, X_test, y_train, y_test = split_data(df)

    assert len(X_train) + len(X_test) == len(df)
    assert set(y_test.unique()) == {"CONFIRMED", "CANDIDATE", "FALSE POSITIVE"}


def test_candidate_models_can_fit_and_predict():
    df = _sample_dataframe()
    X_train, X_test, y_train, y_test = split_data(df)

    models = get_candidate_models()

    for model in models.values():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

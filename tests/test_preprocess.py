import pandas as pd

from src.preprocess import CORE_FEATURES, TARGET_COLUMN, clean_data


def test_clean_data_keeps_only_expected_columns():
    raw_df = pd.DataFrame(
        {
            **{feature: [1.0, 2.0] for feature in CORE_FEATURES},
            TARGET_COLUMN: ["CONFIRMED", "CANDIDATE"],
            "koi_score": [0.9, 0.1],
            "koi_pdisposition": ["CANDIDATE", "CANDIDATE"],
        }
    )

    clean_df = clean_data(raw_df)

    assert set(clean_df.columns) == set(CORE_FEATURES + [TARGET_COLUMN])


def test_clean_data_drops_rows_with_missing_core_features():
    raw_df = pd.DataFrame(
        {
            **{feature: [1.0, None] for feature in CORE_FEATURES},
            TARGET_COLUMN: ["CONFIRMED", "CANDIDATE"],
        }
    )

    clean_df = clean_data(raw_df)

    assert len(clean_df) == 1
    assert clean_df.iloc[0][TARGET_COLUMN] == "CONFIRMED"

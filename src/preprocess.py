"""Clean the raw KOI cumulative table into a model-ready DataFrame.

Based on findings from notebooks/01_exploratory_analysis.ipynb:
- koi_pdisposition and koi_score leak the target and must be excluded
- only a small set of core physical features is needed
- missingness in those features is low, so incomplete rows can be dropped
"""

from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "koi_cumulative.csv"
PROCESSED_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "koi_clean.csv"
)

TARGET_COLUMN = "koi_disposition"

CORE_FEATURES = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_kepmag",
    "koi_impact",
]


def load_raw_data(raw_data_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw KOI cumulative table from disk."""
    return pd.read_csv(raw_data_path, comment="#")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Select core features and target, then drop incomplete rows."""
    columns_to_keep = CORE_FEATURES + [TARGET_COLUMN]
    clean_df = df[columns_to_keep].dropna()
    return clean_df.reset_index(drop=True)


def save_clean_data(df: pd.DataFrame, processed_data_path: Path = PROCESSED_DATA_PATH) -> None:
    """Write the cleaned DataFrame to disk."""
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_data_path, index=False)


if __name__ == "__main__":
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    save_clean_data(clean_df)
    print(f"Cleaned {len(raw_df)} rows down to {len(clean_df)} rows")
    print(f"Saved to {PROCESSED_DATA_PATH}")

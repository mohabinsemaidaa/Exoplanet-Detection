"""Download the Kepler Objects of Interest (KOI) cumulative table from the
NASA Exoplanet Archive and save it as a local CSV file.
"""

from pathlib import Path

import requests

NASA_EXOPLANET_ARCHIVE_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+*+from+cumulative"
    "&format=csv"
)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "koi_cumulative.csv"


def download_koi_data(output_path: Path = OUTPUT_PATH) -> None:
    """Download the KOI cumulative table and write it to output_path."""
    print(f"Downloading KOI data from {NASA_EXOPLANET_ARCHIVE_URL}")
    response = requests.get(NASA_EXOPLANET_ARCHIVE_URL, timeout=60)
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    print(f"Saved {len(response.content):,} bytes to {output_path}")


if __name__ == "__main__":
    download_koi_data()

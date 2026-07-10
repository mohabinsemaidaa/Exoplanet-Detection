# Exoplanet Detection

An end-to-end machine learning project that predicts whether an astronomical
observation is likely to be a confirmed exoplanet, using public data from
NASA's Kepler mission.

## Project Goal

Given measurements collected about a candidate transit signal (orbital
period, transit depth, stellar properties, etc.), predict whether the
observation represents a confirmed exoplanet, a candidate, or a false
positive.

## Dataset

This project uses the **Kepler Objects of Interest (KOI) cumulative table**
from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).

It was chosen because it is:
- an official, well-documented NASA dataset
- moderately sized and easy to work with
- structured with a clear classification target (`CONFIRMED`, `CANDIDATE`,
  `FALSE POSITIVE`) already provided by domain experts

## Project Status

🚧 Early development. See sections below as they are completed.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Download the dataset:

```bash
python3 src/download_data.py
```

This saves the KOI cumulative table to `data/raw/koi_cumulative.csv`
(not committed to the repo — download it fresh instead).

## Model Comparison

_Coming soon._

## Results

_Coming soon._

## Future Improvements

_Coming soon._

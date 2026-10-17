"""Loading and cleaning the diamonds dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

NUMERIC_COLUMNS = ["carat", "depth", "table", "price", "x", "y", "z"]
CATEGORICAL_COLUMNS = ["cut", "color", "clarity"]
REQUIRED_COLUMNS = ["carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y", "z"]
DIMENSION_COLUMNS = ["x", "y", "z"]

# Largest plausible length, width or depth of a diamond, in mm.
# The largest real stones in the data measure about 10.7 mm, so 20 mm is roughly twice that.
MAX_DIMENSION_MM = 20.0


def load_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the diamonds data from a local CSV, or from seaborn when no path is given.

    Reading a local CSV lets tests and the Docker image run without internet access.
    Both sources return the same columns and dtypes: cut, color and clarity as plain
    strings (encoding them is the job of ``features.py``), everything else numeric.

    Raises:
        ValueError: if any of the required columns is missing.
    """
    if csv_path is None:
        import seaborn as sns  # imported lazily: only needed for the online source

        df = sns.load_dataset("diamonds")
    else:
        df = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col])
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove invalid rows and report how many each rule removed.

    The rules run in this order, and each count refers to rows still present at that step:

    1. ``duplicates``: exact duplicate rows. Two diamonds could in theory share every
       attribute, but with 10 columns (several to two decimals) a repeat is far more likely
       to be a double entry, and keeping it would let the same row land in both the
       train and the test set.
    2. ``zero_dimensions``: x, y or z is 0 or negative. A diamond cannot have zero size;
       these are missing measurements stored as 0.
    3. ``outliers``: x, y or z above ``MAX_DIMENSION_MM`` (20 mm). The original group
       notebook spotted y = 58.9 mm and z = 31.8 mm but never removed them. Those rows
       are clearly typing errors (a 0.51 ct stone with x = 5.12 mm and depth 61.8 %
       should be about 3.2 mm deep, not 31.8 mm) and they distort any model that uses x, y or z.

    Returns:
        The cleaned frame with a fresh index, and a dict mapping each rule name to the
        number of rows it removed. The counts add up to ``len(df) - len(cleaned)``.
    """
    removed: dict[str, int] = {}

    before = len(df)
    df = df.drop_duplicates()
    removed["duplicates"] = before - len(df)

    before = len(df)
    df = df[(df[DIMENSION_COLUMNS] > 0).all(axis=1)]
    removed["zero_dimensions"] = before - len(df)

    before = len(df)
    df = df[(df[DIMENSION_COLUMNS] <= MAX_DIMENSION_MM).all(axis=1)]
    removed["outliers"] = before - len(df)

    return df.reset_index(drop=True), removed

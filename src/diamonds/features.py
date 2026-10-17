"""Turning cut, color and clarity into ordered integers a model can use."""

from __future__ import annotations

import pandas as pd

# Each list runs from worst to best, so a higher code always means a better grade.
CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

ORDINAL_LEVELS: dict[str, list[str]] = {
    "cut": CUT_ORDER,
    "color": COLOR_ORDER,
    "clarity": CLARITY_ORDER,
}


def encode_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with cut, color and clarity replaced by integer codes.

    The grades have a natural order (Fair < ... < Ideal, J < ... < D, I1 < ... < IF), so
    one integer column per grade keeps that order and stays easy to read in feature
    importances. One-hot encoding would split each grade into 5-8 columns and throw the
    order away, which a tree model would then have to relearn from the data.

    Raises:
        ValueError: if a column holds a value that is not a known grade. Silently turning
            it into NaN would only fail later, far from the cause.
    """
    out = df.copy()
    for col, levels in ORDINAL_LEVELS.items():
        codes = {level: i for i, level in enumerate(levels)}
        unknown = sorted(set(out[col].astype(str)) - codes.keys())
        if unknown:
            raise ValueError(f"Unknown {col} value(s) {unknown}; expected one of {levels}")
        out[col] = out[col].astype(str).map(codes).astype("int64")
    return out

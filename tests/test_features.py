import pandas as pd
import pytest

from diamonds.features import CLARITY_ORDER, COLOR_ORDER, CUT_ORDER, encode_ordinal


def make_frame(cut, color, clarity) -> pd.DataFrame:
    return pd.DataFrame({"cut": cut, "color": color, "clarity": clarity})


def test_ordinal_encoding_order():
    # One row per grade, listed worst to best; the other two columns hold any valid grade.
    cut = encode_ordinal(make_frame(CUT_ORDER, "D", "IF"))["cut"]
    color = encode_ordinal(make_frame("Ideal", COLOR_ORDER, "IF"))["color"]
    clarity = encode_ordinal(make_frame("Ideal", "D", CLARITY_ORDER))["clarity"]

    # Codes strictly increase from worst to best: Fair < ... < Ideal, J < ... < D, I1 < ... < IF.
    for codes in (cut, color, clarity):
        assert codes.is_monotonic_increasing and codes.is_unique

    # And the order lists themselves are the real-world grading order.
    assert CUT_ORDER == ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    assert COLOR_ORDER == ["J", "I", "H", "G", "F", "E", "D"]
    assert CLARITY_ORDER == ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]


def test_unknown_category_raises():
    with pytest.raises(ValueError, match="Excellent"):
        encode_ordinal(make_frame(["Excellent"], ["D"], ["IF"]))


def test_encoded_sample_has_no_missing_values(raw_sample):
    encoded = encode_ordinal(raw_sample)
    assert encoded[["cut", "color", "clarity"]].notna().all(axis=None)

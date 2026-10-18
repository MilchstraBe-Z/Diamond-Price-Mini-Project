import pandas as pd
import pytest

from diamonds.data import DIMENSION_COLUMNS, MAX_DIMENSION_MM, clean, encode_ordinal, load_data


def test_load_data_rejects_missing_column(tmp_path, raw_sample):
    path = tmp_path / "no_price.csv"
    raw_sample.drop(columns="price").to_csv(path, index=False)
    with pytest.raises(ValueError, match="price"):
        load_data(path)


def test_clean_removes_zero_dimensions(raw_sample):
    cleaned, _ = clean(raw_sample)
    assert (cleaned[DIMENSION_COLUMNS] > 0).all(axis=None)


def test_clean_removes_duplicates(raw_sample):
    cleaned, _ = clean(raw_sample)
    assert cleaned.duplicated().sum() == 0


def test_clean_removes_outliers(raw_sample):
    cleaned, _ = clean(raw_sample)
    assert (cleaned[["y", "z"]] <= MAX_DIMENSION_MM).all(axis=None)


def test_clean_reports_counts(raw_sample):
    cleaned, removed = clean(raw_sample)
    assert set(removed) == {"duplicates", "zero_dimensions", "outliers"}
    assert sum(removed.values()) == len(raw_sample) - len(cleaned)


def test_clean_does_not_modify_input(raw_sample):
    before = raw_sample.copy()
    clean(raw_sample)
    assert raw_sample.equals(before)


# ------------------------------------------------------------------------------


def make_frame(cut, color, clarity) -> pd.DataFrame:
    return pd.DataFrame({"cut": cut, "color": color, "clarity": clarity})


def test_unknown_category_raises():
    with pytest.raises(ValueError, match="Excellent"):
        encode_ordinal(make_frame(["Excellent"], ["D"], ["IF"]))


def test_encoded_sample_has_no_missing_values(raw_sample):
    encoded = encode_ordinal(raw_sample)
    assert encoded[["cut", "color", "clarity"]].notna().all(axis=None)

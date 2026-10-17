import pytest

from diamonds.data import DIMENSION_COLUMNS, MAX_DIMENSION_MM, clean, load_data


def test_sample_contains_bad_rows(raw_sample):
    # Guard: the tests below only mean something if the sample has rows to remove.
    assert raw_sample.duplicated().any()
    assert (raw_sample[DIMENSION_COLUMNS] <= 0).any(axis=None)
    assert (raw_sample[DIMENSION_COLUMNS] > MAX_DIMENSION_MM).any(axis=None)


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


def test_load_data_rejects_missing_column(tmp_path, raw_sample):
    path = tmp_path / "no_price.csv"
    raw_sample.drop(columns="price").to_csv(path, index=False)
    with pytest.raises(ValueError, match="price"):
        load_data(path)

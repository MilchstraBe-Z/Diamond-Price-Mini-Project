import pytest

from diamonds.data import clean
from diamonds.features import encode_ordinal
from diamonds.model import evaluate, split, train_baseline, train_forest


@pytest.fixture
def splits(raw_sample):
    cleaned, _ = clean(raw_sample)
    return split(encode_ordinal(cleaned))


def test_forest_beats_baseline(splits):
    X_train, X_test, y_train, y_test = splits
    baseline = evaluate(train_baseline(X_train, y_train), X_test, y_test)
    forest = evaluate(train_forest(X_train, y_train), X_test, y_test)
    assert forest["mae"] < baseline["mae"]


def test_split_is_reproducible(raw_sample):
    first = split(encode_ordinal(clean(raw_sample)[0]))[1]
    second = split(encode_ordinal(clean(raw_sample)[0]))[1]
    assert first.index.equals(second.index)


def test_evaluate_returns_test_metrics(splits):
    X_train, X_test, y_train, y_test = splits
    scores = evaluate(train_baseline(X_train, y_train), X_test, y_test)
    assert set(scores) == {"r2", "mae"}
    assert scores["mae"] > 0

import pytest

from diamonds.data import clean, encode_ordinal
from diamonds.model import evaluate, split, train_baseline, train_forest


@pytest.fixture
def splits(raw_sample):
    cleaned, _ = clean(raw_sample)
    return split(encode_ordinal(cleaned))


def test_evaluate_returns_test_metrics(splits):
    X_train, X_test, y_train, y_test = splits
    scores = evaluate(train_baseline(X_train, y_train), X_test, y_test)
    assert set(scores) == {"r2", "mae"}
    assert scores["mae"] > 0


def test_forest_beats_baseline(splits):
    X_train, X_test, y_train, y_test = splits
    baseline = evaluate(train_baseline(X_train, y_train), X_test, y_test)
    forest = evaluate(train_forest(X_train, y_train), X_test, y_test)
    assert forest["mae"] < baseline["mae"]



"""Train/test split, the two models and their test-set evaluation."""

from __future__ import annotations

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

TARGET = "price"
BASELINE_FEATURES = ["carat"]
FOREST_FEATURES = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]

RANDOM_STATE = 42
TEST_SIZE = 0.25


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split an encoded frame into X_train, X_test, y_train, y_test.

    The split is fixed (``random_state=42``, 25 % test, same as the group notebook) so
    both models are scored on exactly the same unseen rows and runs are repeatable.
    """
    X = df[FOREST_FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Linear regression on carat only: the simplest model worth beating."""
    return LinearRegression().fit(X_train[BASELINE_FEATURES], y_train)


def train_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Random forest on all features, with default hyperparameters.

    A forest handles the non-linear carat-price curve and interactions between size and
    grade without manual feature engineering, and needs no feature scaling.
    """
    forest = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    return forest.fit(X_train[FOREST_FEATURES], y_train)


def evaluate(model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Score a fitted model on the test set: R² and mean absolute error in USD.

    Only test rows are used. Training scores say how well a model memorised its data,
    not how well it predicts new diamonds.
    """
    predictions = model.predict(X_test[list(model.feature_names_in_)])
    return {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
    }


def feature_importances(forest: RandomForestRegressor) -> dict[str, float]:
    """Return the forest's feature importances, largest first."""
    pairs = zip(forest.feature_names_in_, forest.feature_importances_, strict=True)
    return {name: float(value) for name, value in sorted(pairs, key=lambda p: -p[1])}

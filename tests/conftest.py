from pathlib import Path

import pandas as pd
import pytest

from diamonds.data import load_data

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "sample.csv"


@pytest.fixture
def sample_csv() -> Path:
    return SAMPLE_CSV


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    """The 200-row sample, including its deliberate bad rows."""
    return load_data(SAMPLE_CSV)

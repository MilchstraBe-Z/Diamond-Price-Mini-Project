"""Build data/sample.csv: a small offline sample of the diamonds data for tests and Docker.

The sample holds 190 random valid rows plus 10 bad rows on purpose, so the cleaning
rules have something to remove:

- 4 exact duplicates of valid rows
- 3 real rows with a zero dimension
- the 3 real rows with y or z above 20 mm

Run from the repo root (needs internet once): python scripts/make_sample.py
"""

from pathlib import Path

import pandas as pd

from diamonds.data import CATEGORICAL_COLUMNS, DIMENSION_COLUMNS, MAX_DIMENSION_MM, load_data

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sample.csv"
SEED = 42


def main() -> None:
    df = load_data().drop_duplicates()

    has_zero = (df[DIMENSION_COLUMNS] <= 0).any(axis=1)
    too_big = (df[DIMENSION_COLUMNS] > MAX_DIMENSION_MM).any(axis=1)
    valid = df[~has_zero & ~too_big]

    good = valid.sample(n=190, random_state=SEED)
    duplicates = good.sample(n=4, random_state=SEED)
    zero_dims = df[has_zero].sample(n=3, random_state=SEED)
    outliers = df[too_big]

    sample = pd.concat([good, duplicates, zero_dims, outliers])
    sample = sample.sample(frac=1, random_state=SEED)  # shuffle so bad rows are spread out

    for col in CATEGORICAL_COLUMNS:
        missing = set(df[col]) - set(sample[col])
        if missing:
            raise SystemExit(f"Sample is missing {col} levels {missing}; change SEED.")

    OUT_PATH.parent.mkdir(exist_ok=True)
    sample.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(sample)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()

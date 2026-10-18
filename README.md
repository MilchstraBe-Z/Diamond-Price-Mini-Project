# diamond-price

[![CI](https://github.com/MilchstraBe-Z/Diamond-Price-Mini-Project/actions/workflows/ci.yml/badge.svg)](https://github.com/MilchstraBe-Z/Diamond-Price-Mini-Project/actions/workflows/ci.yml)

Predicts the price of a diamond from its size and grades, as a small tested Python package
with a command-line tool, a Docker image and a GitHub Actions pipeline.

It started as a four-person course notebook (Data Analytics, KIT/FZI) analysing about 54,000
diamonds. This repo rebuilds that analysis as a package: explicit data-cleaning rules,
cut/color/clarity used in the model, test-set-only metrics, and every push linted, tested and
built in CI.

## Results

Full data (53,940 rows) → 53,772 after cleaning. Test set: 25 % of the cleaned data
(13,443 diamonds), never seen during training.

| Model | Features | Test R² | Test MAE |
|---|---|---|---|
| Group notebook | linear regression on x, y, z | 0.788 | $1,368 |
| Baseline | linear regression on carat | 0.847 | $1,011 |
| **Random forest** | carat, cut, color, clarity, depth, table, x, y, z | **0.982** | **$268** |

The forest's predictions are off by $268 on average, against $1,011 for the baseline.
Most important features: carat (0.61), y (0.27), clarity (0.06), color (0.03).

Rows removed by the cleaning rules: 146 duplicates, 19 with a zero dimension, 3 outliers
(a dimension above 20 mm).

## Quick start

### Local (Python 3.11+)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"

diamonds train                                  # full data, downloaded via seaborn
diamonds train --csv data/sample.csv            # 200-row offline sample
diamonds train --csv data/diamonds.csv          # full data from the local copy
```

Each run prints a metrics table and writes `report.json` (row counts, rows removed by each
cleaning rule, test metrics, top 5 feature importances). Use `--output PATH` to write it
elsewhere. A missing file or a missing column prints `error: ...` and exits with code 1.

### Docker

```bash
docker build -t diamond-price .
docker run --rm diamond-price                                         # runs on the sample
docker run --rm -v "$PWD/data:/data" diamond-price train --csv /data/diamonds.csv
```

### Tests and lint

```bash
pytest
ruff check .
ruff format --check .
```

12 tests, all offline on `data/sample.csv`. The sample contains 10 bad rows on purpose
(4 duplicates, 3 zero dimensions, the 3 real y/z typing errors) so every cleaning rule is
exercised. `scripts/make_sample.py` rebuilds it.

## How it works

```
src/diamonds/
├── data.py    load_data():      seaborn or local CSV, checks the 10 required columns, fixes dtypes
│              clean():          drops duplicates, zero dimensions, x/y/z > 20 mm; reports counts
│              encode_ordinal(): cut Fair→Ideal, color J→D, clarity I1→IF as integers;
│                                an unknown grade raises instead of becoming NaN
├── model.py   split():          fixed 75/25 split (random_state=42)
│              train_baseline(): linear regression on carat
│              train_forest():   random forest (100 trees) on all 9 features
│              evaluate():       test-set R² and MAE
└── cli.py     diamonds train [--csv PATH] [--output PATH]

tests/
├── test_data.py   loading, each cleaning rule, counts, no side effects, encoding
├── test_model.py  metrics shape, forest beats baseline
└── test_cli.py    report.json contents, error exit code
```

CI (`.github/workflows/ci.yml`) runs on every push and pull request to `master`:

1. **test** job: ruff lint, ruff format check and pytest on Python 3.11 and 3.12
   (matrix, `fail-fast: false`, pip cache).
2. **docker** job, only if `test` passes: build the image and smoke-test it by running it
   and checking its output for the forest result.

## What changed from the notebook

| | Group notebook | This repo |
|---|---|---|
| Outliers | y ≈ 59 mm and z ≈ 32 mm found but kept | removed (any dimension > 20 mm), with a test |
| Duplicates, zero sizes | removed | removed, and each rule reports how many rows it dropped |
| Cut, color, clarity | not used in any model | ordinal-encoded and used by the forest |
| Evaluation | mostly training-set R² | test-set R² and MAE only |
| Models | one forest per single feature, `max_depth=3` | one forest on all features vs. a carat baseline |
| Reproducibility | run cells top to bottom | `pip install`, one command, Docker, CI |

## Project layout

```
.github/workflows/ci.yml   CI pipeline
data/diamonds.csv          full dataset (local copy, 53,940 rows)
data/sample.csv            200-row offline sample for tests and Docker
notebooks/                 the original group notebook, unchanged
scripts/make_sample.py     regenerates data/sample.csv
src/diamonds/              the package
tests/                     pytest suite
Dockerfile, .dockerignore  container image
pyproject.toml             dependencies, CLI entry point, pytest and ruff config
```

## Credits

The original analysis in [`notebooks/original_case_study.ipynb`](notebooks/original_case_study.ipynb)
is group work by Nurul Fazlina Zainal Abidin, Shituan Lin, Medha Agarwal and Hongyi Zhao.
The package, tests, Docker setup and CI in this repo were built afterwards by Hongyi Zhao.

Data: the `diamonds` dataset from [seaborn-data](https://github.com/mwaskom/seaborn-data).

## License

MIT, see [LICENSE](LICENSE).

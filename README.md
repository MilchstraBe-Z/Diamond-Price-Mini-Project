# diamond-price

[![CI](https://github.com/MilchstraBe-Z/Diamond-Price-Mini-Project/actions/workflows/ci.yml/badge.svg)](https://github.com/MilchstraBe-Z/Diamond-Price-Mini-Project/actions/workflows/ci.yml)

Predicts the price of a diamond from its size and grades, as a small tested Python package
with a command-line tool, a Docker image and a GitHub Actions pipeline.

It started as a four-person course notebook (Data Analytics, KIT/FZI) analysing about 54,000
diamonds. This repo rebuilds that analysis as a package: proper data cleaning, cut/color/clarity
used in the model, test-set-only metrics, and every push linted, tested and built in CI.

## Results

Test set: 25 % of the cleaned data (13,443 diamonds), never seen during training.

| Model | Features | Test R² | Test MAE |
|---|---|---|---|
| Group notebook | linear regression on x, y, z | 0.788 | $1,368 |
| Baseline | linear regression on carat | 0.847 | $1,011 |
| **Random forest** | carat, cut, color, clarity, depth, table, x, y, z | **0.982** | **$268** |

The forest's predictions are off by $268 on average, against $1,011 for the baseline.
Most important features: carat (0.62), y (0.27), clarity (0.06), color (0.03).

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

Each run prints a metrics table and writes `report.json` (metrics, rows removed by each
cleaning rule, top feature importances). Use `--output PATH` to write it elsewhere.

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

All tests run offline on `data/sample.csv`, which contains a few bad rows on purpose
(duplicates, zero dimensions, y/z typing errors) so every cleaning rule is exercised.

## How it works

```
src/diamonds/
├── data.py      load_data(): seaborn or local CSV, checks required columns
│                clean(): drops duplicates, zero dimensions, x/y/z > 20 mm; reports counts
├── features.py  encode_ordinal(): cut Fair→Ideal, color J→D, clarity I1→IF as integers
├── model.py     fixed 75/25 split, baseline, random forest, test-set R² and MAE
└── cli.py       diamonds train [--csv PATH] [--output PATH]
```

CI (`.github/workflows/ci.yml`) runs on every push and pull request to `main`:
ruff and pytest on Python 3.11 and 3.12, then, only if those pass, a Docker build and a
smoke-test run of the image.

## What changed from the notebook

| | Group notebook | This repo |
|---|---|---|
| Outliers | y ≈ 59 mm and z ≈ 32 mm found but kept | removed (any dimension > 20 mm), with a test |
| Duplicates, zero sizes | removed | removed, and each rule reports how many rows it dropped |
| Cut, color, clarity | not used in any model | ordinal-encoded and used by the forest |
| Evaluation | mostly training-set R² | test-set R² and MAE only |
| Models | one forest per single feature, `max_depth=3` | one forest on all features vs. a carat baseline |
| Reproducibility | run cells top to bottom | `pip install`, one command, Docker, CI |

## Credits

The original analysis in [`notebooks/original_case_study.ipynb`](notebooks/original_case_study.ipynb)
is group work by Nurul Fazlina Zainal Abidin, Shituan Lin, Medha Agarwal and Hongyi Zhao.
The package, tests, Docker setup and CI in this repo were built afterwards by Hongyi Zhao.

Data: the `diamonds` dataset from [seaborn-data](https://github.com/mwaskom/seaborn-data).

## License

MIT, see [LICENSE](LICENSE).

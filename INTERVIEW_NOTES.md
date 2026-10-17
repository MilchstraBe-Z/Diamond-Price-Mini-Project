# Interview notes (local only, never committed: listed in .git/info/exclude)

## 5-minute demo

| Min | Show | Say |
|---|---|---|
| 0–1 | README on GitHub (badge, results table) | Group notebook, 54k diamonds; I rebuilt it alone as a package. Forest: $268 MAE vs $1,011 baseline. |
| 1–2 | `src/diamonds/` tree | data → features → model → cli; each file has one job, so each can be tested alone. |
| 2–3 | `clean()` in data.py + `test_clean_removes_outliers` | The notebook found y = 58.9 mm but kept it. I drop anything > 20 mm (largest real stone is 10.7 mm), and a test protects the rule. |
| 3–4 | Actions tab: a PR run | ruff → pytest on 3.11 and 3.12 → Docker build only if tests pass. Show the merged PR with its checks. |
| 4–5 | Terminal: `docker run --rm diamond-price` | Same result on any machine. Next: tune the forest with CV, log-price target, push image to a registry on tags. |

Live-demo trick (optional): set `MAX_DIMENSION_MM = 100.0`, run `pytest`, show the red test, revert.

Commands to have ready:

```bash
docker build -t diamond-price .
docker run --rm diamond-price
pytest -v
```

## Answers

**Which parts were the group's, which are mine?** Group: the exploration, the notebook's
cleaning of duplicates/zero sizes, single-feature trees and forests. Mine: the whole package,
the outlier rule, using cut/color/clarity, test-set-only evaluation, tests, Docker, CI, README.

**Why a random forest?** Price grows non-linearly with carat and grades interact with size;
a forest captures both with no feature engineering or scaling, and it is a strong default
for tabular data.

**Why not tune it?** Scope: the goal was a finished, tested pipeline. Defaults already reach
R² 0.98. Tuning would be `RandomizedSearchCV` on the training set only, with the test set
untouched until the end.

**How do you know it doesn't overfit?** Every number I report is on a held-out 25 % test set.
Train R² 0.997 vs test 0.982: a small gap. Next step would be 5-fold cross-validation.

**Why ordinal, not one-hot?** The grades are ordered. One integer column keeps the order, gives
9 features instead of ~25, and readable importances. Trees split on thresholds, so "≥ VS1"
is one split. For a linear model the equal-spacing assumption would matter more.

**Missing column in the CSV?** `load_data` checks all 10 columns up front and raises
`ValueError: Input data is missing required columns: ['price']`; the CLI prints it and exits 1.
There is a test for it. An unknown grade raises too, instead of becoming NaN.

**Make CI faster?** Already: pip cache, cancelling outdated runs, the Docker dependency layer.
More: Docker layer cache in Actions (`docker/build-push-action` with `cache-from: type=gha`),
run lint in its own short job, use `uv` instead of pip.

**Add deployment?** A third job, `needs: docker`, only on tags (`on: push: tags: ['v*']`):
log in to a registry (GHCR with `GITHUB_TOKEN`, `permissions: packages: write`), build, tag
with the version, push.

**Value for a Bosch engineering team?** Same pattern for test-bench or measurement data:
explicit, tested data-quality rules (what got removed and why), reproducible runs in Docker,
and every change checked by CI before anyone relies on the numbers.

**Did you use AI tools?** Answer honestly, e.g.: "I used Claude Code as a pair programmer to
scaffold the package, tests, Docker and CI from my plan. I reviewed every file, can explain
each line, checked the numbers against the original notebook, and verified the cache and
test behaviour myself." (Commits carry a Co-Authored-By line, so be consistent with that.)

## Dockerfile: line by line

- `FROM python:3.11-slim`: small official base image with Python, same version as CI.
- `ENV ...`: no .pyc files, logs unbuffered (show up immediately), no pip cache in the image.
- `COPY pyproject.toml` + stub install: dependencies get their own layer. Docker reuses a layer
  while its inputs are unchanged, so code or README edits skip the slow pip install (verified:
  step shows `CACHED`).
- `COPY src/`, `data/sample.csv`, `pip install --no-deps .`: install the real package only.
- `useradd` + `USER app`: don't run as root; if the process is compromised it has fewer rights.
- `ENTRYPOINT ["diamonds"]` + `CMD [...]`: the image *is* the CLI; CMD is the default
  argument list and is replaced by whatever you pass to `docker run`.
- Image vs container: the image is the read-only template; a container is one running instance.
- `.dockerignore`: keeps .venv, .git, notebooks, tests and the full 2.7 MB CSV out of the build
  context: faster builds, smaller image, no secrets copied by accident.

## ci.yml: line by line

- `on: push / pull_request` to main: checks every change before and after merging.
- `permissions: contents: read`: least privilege for the workflow token.
- `concurrency`: a new push cancels the outdated run on the same branch.
- Runner: a fresh GitHub-hosted Ubuntu VM per job, thrown away afterwards.
- `matrix` 3.11/3.12: same steps on both versions, `fail-fast: false` so one failure does not
  hide the other.
- `setup-python` with `cache: pip`: reuses downloaded wheels, keyed on pyproject.toml.
- `ruff check`, `ruff format --check`, `pytest -v`: lint, style, tests; any non-zero exit fails.
- `docker` job `needs: test`: no image from code that fails tests; jobs otherwise run in parallel.
- Smoke test: run the image and `grep` for "forest" in its output.

## Still to do before the interview

- [ ] Push `main`, then the two branches; open PRs (`docs/readme` first, then
      `feature/llm-explain`) and merge once CI is green
- [ ] Check the badge turns green on the README
- [ ] Optional: try `diamonds explain` once with a real API key
- [ ] Screenshots: green CI run, the merged PR, `docker run` output
- [ ] Rehearse twice with a timer (< 6 min)
- [ ] Résumé bullet

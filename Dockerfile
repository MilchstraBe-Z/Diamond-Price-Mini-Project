FROM python:3.11-slim

# No .pyc files, unbuffered logs, smaller image.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1) Dependencies first. Only pyproject.toml is copied, so this slow layer is rebuilt
#    only when the dependencies change, not on every code or README edit. The empty
#    stubs just let pip resolve the project; the real files replace them below.
COPY pyproject.toml ./
RUN mkdir -p src/diamonds && touch src/diamonds/__init__.py README.md \
    && pip install . \
    && rm -rf src README.md

# 2) The code and the offline sample, then the package itself (dependencies already in).
COPY README.md ./
COPY src/ src/
COPY data/sample.csv data/sample.csv
RUN pip install --no-deps .

# 3) Run as an unprivileged user that may write report.json into /app.
RUN useradd --create-home --uid 1000 app && chown app /app
USER app

ENTRYPOINT ["diamonds"]
CMD ["train", "--csv", "data/sample.csv"]

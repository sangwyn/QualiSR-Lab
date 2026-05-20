FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# RUN apt-get update \
#     && apt-get install -y --no-install-recommends git build-essential \
#     && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY qualisr_lab ./qualisr_lab
COPY scripts ./scripts
COPY configs ./configs
COPY scores ./scores
COPY features ./features

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[regressors]" \
    && python -m pip cache purge

CMD ["qualisr-run-regressors", "--config", "configs/default.json", "--no-plots"]

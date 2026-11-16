FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev
COPY data/raw/SeoulBikeData.csv ./data/raw/
COPY models/model.joblib ./models/
ENV PATH="/app/.venv/bin:$PATH" OMP_NUM_THREADS=2
EXPOSE 8000
CMD ["uvicorn", "seoul_bike.api:app", "--host", "0.0.0.0", "--port", "8000"]

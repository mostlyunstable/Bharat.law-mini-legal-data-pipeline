FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# Optional build deps (kept small, but helps when wheels aren't available).
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
CMD ["python", "-m", "scripts.run_pipeline"]

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Prevent Python from writing pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      libffi-dev \
      libssl-dev \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home app
WORKDIR /home/app/chess-endgame-mcts

COPY requirements.txt ./
RUN if [ -s requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; else echo "# empty" > requirements.txt; fi


CMD ["python", "-m", "src.train"]
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Prevent Python from writing pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies needed for many Python packages (numpy, scipy, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      libffi-dev \
      libssl-dev \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user and application directory
RUN useradd --create-home app
WORKDIR /home/app/chess-endgame-mcts

# Copy requirements first to leverage Docker layer caching
# Add a no-op requirements file default so builds don't fail if none exists
COPY requirements.txt ./
RUN if [ -s requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; else echo "# empty" > requirements.txt; fi

# Copy application source
COPY . .

# Ensure files are owned by non-root user
RUN chown -R app:app /home/app/chess-endgame-mcts
USER app

# Default port (adjust if your app uses a different one)
EXPOSE 8000

# Default command: run main.py if present, otherwise open a shell
CMD ["python", "-m", "src.train"]
# syntax=docker/dockerfile:1
FROM python:3.10-slim

# System dependencies (for PIL etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .
COPY test.b64 .

# Default command
CMD ["python", "predictor.py"]


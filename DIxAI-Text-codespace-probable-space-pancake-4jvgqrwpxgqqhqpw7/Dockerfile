# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies from pyproject.toml
RUN pip install --no-cache-dir .

# Metadata
LABEL maintainer="Haythem Ghazouani"
LABEL version="0.1.0"
LABEL description="Decision-Information Explainable AI (DIxAI) Reproducibility Environment"

# Default command: run the master reproducibility script
CMD ["python", "reproduce_results.py"]

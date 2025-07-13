# Dockerfile

# Use official Python image with necessary build tools
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libatlas-base-dev \
    libboost-python-dev \
    libboost-thread-dev \
    libopenblas-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy your code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Start Flask app
CMD ["python", "app.py"]

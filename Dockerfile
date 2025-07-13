# Use a stable Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install required system dependencies for dlib, face-recognition, OpenCV, etc.
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

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expose Flask default port
EXPOSE 5000

# Run the Flask server
CMD ["python", "app.py"]

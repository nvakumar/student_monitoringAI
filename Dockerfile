# Use a base image with Python
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libatlas-base-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Upgrade pip and install Python packages
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expose port (Flask default)
EXPOSE 5000

# Run your Flask app
CMD ["python", "app.py"]

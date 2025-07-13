# Use official Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy only what's needed
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]

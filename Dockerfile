FROM python:3.11-slim

# System dependencies for OpenCV + MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    libgomp1 wget curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p logs static/events static/exports

# Expose port
EXPOSE 5000

# Environment defaults
ENV FLASK_ENV=production
ENV HOST=0.0.0.0
ENV PORT=5000

CMD ["python", "run.py"]

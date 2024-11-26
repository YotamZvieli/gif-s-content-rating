# Stage 1: Build the React frontend
FROM node:16 as frontend-builder
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Set up Flask backend
FROM python:3.9-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY Python-Backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend and model files
COPY Python-Backend/ ./

# Copy built frontend to serve as static files
COPY --from=frontend-builder /app/build ./static

# Expose the Flask app's port
EXPOSE 5000

# Run the Flask app
CMD ["python", "gifs-rate-server.py"]

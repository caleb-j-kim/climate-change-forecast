# === FRONTEND ===
FROM node:18 AS frontend-build
WORKDIR /app/frontend

# Copy package.json and install dependencies
COPY frontend/package*.json ./
RUN npm install

# Copy React source code
COPY frontend/public ./public
COPY frontend/src ./src
COPY frontend/.env* ./

# Build React app
RUN npm run build

# === BACKEND ===
FROM python:3.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Remove any existing frontend/build directory
RUN rm -rf frontend/build

# Copy built React app from frontend-build stage
COPY --from=frontend-build /app/frontend/build ./frontend/build

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_PORT=5000

# Expose the port
EXPOSE 5000

# Run Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
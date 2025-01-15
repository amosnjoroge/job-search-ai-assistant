# Use python slim image as base
FROM python:3.9-slim

# Define build-time arguments with default value
ARG PORT=8501

# Define environment variable from ARG
ENV PORT=${PORT}

# Set the working directory
WORKDIR /app

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create .dockerignore from .gitignore
COPY .gitignore .dockerignore

# Copy the application code
# Docker will automatically use .dockerignore to exclude files
COPY . .

# Create and switch to non-root user for security
RUN useradd -m ajsAssistant
USER ajsAssistant

# Expose the port using the ARG
EXPOSE ${PORT}

# Add healthcheck to monitor application status
HEALTHCHECK CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

# Command to run the app using the environment variable
CMD streamlit run main.py --server.port=${PORT}
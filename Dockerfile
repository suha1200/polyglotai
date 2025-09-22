# Use official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . /app

# ✅ Install dependencies from requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Default command when container runs (bash for dev)
CMD ["bash"]


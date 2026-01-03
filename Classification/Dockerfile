FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy requirements from project root
COPY ../requirements.txt .

# Install dependencies globally (inside container)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY churn_api.py model_1.0.bin ./

# Expose port
EXPOSE 8686

# Run Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8686", "churn_api:app"]
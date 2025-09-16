# Use a specific, stable version of Python as the base image
FROM python:3.9.18-slim

# Set environment variables for Python and the application port
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

# Set the working directory inside the container
WORKDIR /app

# Create a non-root user for enhanced security
RUN adduser --disabled-password --gecos "" appuser

# Copy the requirements file and install dependencies
# This is done before copying the app code for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy the rest of the application code into the container
COPY . .

# Change ownership of the app directory to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Expose the port the app will run on
EXPOSE $PORT

# Add a healthcheck to ensure the application is running correctly
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:$PORT/health || exit 1

# Command to run the application using Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

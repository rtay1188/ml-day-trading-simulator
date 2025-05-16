# Use Python 3.13 slim as the base image
FROM python:3.13-slim

# Set fake environment variables for testing (default values can be overridden at runtime)
ENV DBNAME="stockspostgresdb" \
    USER="pguser" \
    PASSWORD="pgadmin" \
    HOST="10.0.0.77" \
    PORT="5432"

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements.txt first to leverage Docker's caching
COPY requirements.txt .

RUN pip install --upgrade setuptools wheel

# Install dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy each file explicitly
COPY config.py .
COPY db.py .
COPY fetcher.py .
COPY main.py .
COPY model.py .
COPY strategy.py .

# Specify the command to run the application
CMD ["python", "main.py"]

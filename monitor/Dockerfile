# Python 3.11-slim is a good choice for a lightweight base
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore --disable-pip-version-check -r requirements.txt

# Copy the source code
COPY ./src/monitor .

# Command to run the application
CMD ["python", "-u", "monitor.py"]

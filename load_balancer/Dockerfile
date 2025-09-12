# Python 3.11-slim is a good choice for a lightweight base
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code preserving package layout
COPY ./src /app

# Command to run the application using package path
CMD ["uvicorn", "load_balancer.main:app", "--host", "0.0.0.0", "--port", "8000"]

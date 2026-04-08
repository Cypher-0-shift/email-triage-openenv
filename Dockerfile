FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies (Updated to look in root)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy full package
COPY . /app/env/

# Set PYTHONPATH so imports work
ENV PYTHONPATH=/app/env

WORKDIR /app/env

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
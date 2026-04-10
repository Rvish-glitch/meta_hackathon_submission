FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY environment/ environment/
COPY inference.py .
COPY openenv.yaml .

# Default: run inference script (requires HF_TOKEN at runtime)
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "inference.py"]

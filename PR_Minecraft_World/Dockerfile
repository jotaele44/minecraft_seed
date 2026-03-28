FROM python:3.11-slim

# Install GDAL (required by rasterio) and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin \
        libgdal-dev \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps before copying source (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create output directories
RUN mkdir -p data/raw data/processed output/logs output/heightmap output/worldpainter

# Default: run the full pipeline (will use synthetic DEM if no network)
CMD ["python", "tools/run_pipeline.py"]

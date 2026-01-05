# CPU-friendly base. If you need GPU later, see the GPU notes below.
FROM python:3.11-slim

# Prevent python from writing .pyc files, and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python deps first (better Docker layer caching)
COPY streamlit_app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project into the container
# (This includes saved_models_tensorflow/, streamlit_app/, etc.)
COPY . /app

# Streamlit configuration
EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the app
CMD ["streamlit", "run", "streamlit_app/app.py"]

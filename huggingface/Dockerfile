# TensorFlow is stable on Python 3.10/3.11 (NOT 3.13)
FROM python:3.11-slim

WORKDIR /app

# Avoid .pyc files + keep logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps:
# - curl: healthcheck
# - build-essential: only needed if any wheels must compile (safe to keep)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python deps first (better Docker layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy app code
COPY src/ /app/src/
COPY .streamlit/ /app/.streamlit/

# Hugging Face Spaces typically expects port 7860 by default.
# (If you set app_port in the Space README, match it here too.)
ENV PORT=7860
EXPOSE 7860

# Streamlit health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

# Run Streamlit
CMD ["bash", "-lc", "streamlit run src/streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT}"]

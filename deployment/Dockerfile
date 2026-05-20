# ============================================================================
# Dockerfile — Streamlit App Deployment Configuration
# ============================================================================
# This Dockerfile defines the container configuration for deploying
# the Predictive Engine Maintenance Streamlit app on HuggingFace Spaces.
#
# Configuration:
# - Base image: Python 3.9 slim
# - Dependencies: installed from requirements.txt
# - Port: 7860 (HuggingFace Spaces default)
# - User: non-root (security best practice)
# ============================================================================

FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user (HuggingFace Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Copy application files
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Expose port for Streamlit
EXPOSE 7860

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Launch Streamlit app
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

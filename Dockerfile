# Production-ready Python 3.12
FROM python:3.12-slim

# Setup non-root user for Hugging Face security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements from your backend folder and install
COPY --chown=user backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all files from the backend folder into the container
COPY --chown=user backend/ .

# Hugging Face requires port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

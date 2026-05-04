FROM python:3.11-slim

# System deps needed by tensorflow / opencv / ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces conventions: run as a non-root user with uid 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HOME=/home/user \
    METRABS_CACHE=/home/user/metrabs_models

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["python", "app.py"]

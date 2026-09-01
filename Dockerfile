# Hugging Face Spaces (SDK: docker) mengharapkan app mendengarkan di port 7860.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn==23.0.0

COPY app.py .
COPY Outputs/artifacts ./Outputs/artifacts
COPY Outputs/models/cluster_profiles.json ./Outputs/models/
COPY Outputs/templates ./Outputs/templates
COPY Outputs/static ./Outputs/static

ENV PORT=7860
EXPOSE 7860
# --preload memuat UMAP 77 MB sekali sebelum fork, bukan sekali per worker.
# --timeout 120 memberi ruang untuk load + JIT warm-up saat startup.
CMD ["gunicorn", "-w", "2", "--preload", "--timeout", "120", "-b", "0.0.0.0:7860", "app:app"]

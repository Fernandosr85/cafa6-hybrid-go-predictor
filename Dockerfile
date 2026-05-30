# ──────────────────────────────────────────────────────────────
# CAFA-6 Hybrid Predictor — LAFA Container
# MMseqs2 + k-mer Jaccard + Reranker (IA-weighted)
# Author: Fernando Rodrigues
# License: MIT
# ──────────────────────────────────────────────────────────────
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pipeline source
COPY predict.py .

# ── Default paths used by LAFA ──────────────────────────────
# LAFA will mount data via --fasta / --train_fasta / etc.
# Output is written to /output/predictions.tsv by default.

RUN mkdir -p /data /output

# Entrypoint: LAFA calls this with --fasta and expects output TSV
ENTRYPOINT ["python", "/app/predict.py"]

# Default CMD (can be overridden by LAFA)
# LAFA typically provides:
#   --fasta /data/test.fasta
#   --train_fasta /data/train.fasta
#   --train_terms /data/train_terms.tsv
#   --ia /data/ia.tsv
#   --output /output/predictions.tsv
CMD ["--help"]

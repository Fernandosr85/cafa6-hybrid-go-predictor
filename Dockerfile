# ──────────────────────────────────────────────────────────────
# CAFA-6 Hybrid Predictor + GO Closure — LAFA Container
# MMseqs2 + k-mer Jaccard + IA-weighted Reranker + GO Propagation
# Author: Fernando Rodrigues | License: MIT
# ──────────────────────────────────────────────────────────────
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py .

# Pre-download go-basic.obo so container works offline
RUN python -c "\
import urllib.request, os; \
url='http://purl.obolibrary.org/obo/go/go-basic.obo'; \
urllib.request.urlretrieve(url, '/tmp/go-basic.obo'); \
print('go-basic.obo downloaded:', os.path.getsize('/tmp/go-basic.obo'), 'bytes')"

RUN mkdir -p /data /output

ENTRYPOINT ["python", "/app/predict.py", "--obo", "/tmp/go-basic.obo"]

# LAFA call example:
# docker run --rm \
#   -v /path/to/data:/data \
#   -v /path/to/output:/output \
#   cafa6-hybrid \
#     --fasta       /data/test.fasta \
#     --train_fasta /data/train.fasta \
#     --train_terms /data/train_terms.tsv \
#     --ia          /data/ia.tsv \
#     --output      /output/predictions.tsv
CMD ["--help"]

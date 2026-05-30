# CAFA-6 Hybrid GO Term Predictor

**MMseqs2 + k-mer Jaccard + IA-weighted Reranker**  
LAFA-compatible protein function prediction method.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Method Overview

This pipeline predicts Gene Ontology (GO) terms for query proteins using a **precision-first hybrid strategy**:

1. **k-mer Jaccard similarity** (k=5) — fast sparse neighbor retrieval across all training proteins
2. **Hybrid fusion** — when MMseqs2 precomputed hits are available, combines them with k-mer hits (weights: 0.75 / 0.25)
3. **IA-weighted reranker** — transfers GO terms from neighbors with rank decay, vote consensus boost, and Information Accretion weighting
4. **Frequency penalty** — penalizes very common/generic GO terms
5. **Emission profiles** — adaptive thresholds and per-ontology caps based on maximum neighbor similarity
6. **Gap fill + fallback** — guarantees minimal predictions even for orphan proteins

Outputs predictions for all three GO ontologies:
- **P** — Biological Process (max 10 terms/protein)
- **F** — Molecular Function (max 7 terms/protein)
- **C** — Cellular Component (max 7 terms/protein)

Global cap: **14 terms per protein**.

---

## Usage

### Local (Python)

```bash
pip install -r requirements.txt

python predict.py \
  --fasta       test.fasta \
  --train_fasta train_sequences.fasta \
  --train_terms train_terms.tsv \
  --ia          ia.tsv \
  --output      predictions.tsv
```

### Docker

```bash
# Build
docker build -t cafa6-hybrid .

# Run
docker run --rm \
  -v /path/to/data:/data \
  -v /path/to/output:/output \
  cafa6-hybrid \
    --fasta       /data/test.fasta \
    --train_fasta /data/train.fasta \
    --train_terms /data/train_terms.tsv \
    --ia          /data/ia.tsv \
    --output      /output/predictions.tsv
```

---

## Input Format

| File | Format | Description |
|------|--------|-------------|
| `--fasta` | FASTA | Test protein sequences |
| `--train_fasta` | FASTA | Training protein sequences |
| `--train_terms` | TSV (no header): `protein_id`, `GO_term`, `ontology` | Training GO annotations |
| `--ia` | TSV (no header): `GO_term`, `IA_score` | Information Accretion scores |

FASTA headers support UniProt format: `sp|ACCESSION|ENTRY_NAME` → accession is extracted automatically.

## Output Format

Tab-separated, no header:
```
protein_id\tGO:XXXXXXX\t0.XXX
```

---

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `k` | 5 | k-mer length |
| `vocab_size` | 50,000 | k-mer vocabulary size |
| `top_k_similar` | 35 | neighbors retrieved per query |
| `min_jaccard` | 0.08 | minimum Jaccard to keep neighbor |
| `ia_weight` | 0.45 | Information Accretion boost weight |
| `rank_decay` | 0.88 | exponential rank decay for neighbors |
| `freq_penalty_threshold` | 0.04 | frequency above which terms are penalized |
| `max_global_terms_per_protein` | 14 | hard global cap |

---

## Requirements

- Python 3.10+
- numpy, pandas, scipy, scikit-learn
- biopython (optional, used for robust FASTA parsing)
- tqdm

See `requirements.txt` for pinned versions.

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Citation

If you use this method, please cite the CAFA-6 competition and reference this repository.

> Rodrigues, F. (2025). *CAFA-6 Hybrid GO Term Predictor*. GitHub. [link]

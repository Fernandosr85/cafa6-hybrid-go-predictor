"""
CAFA-6 Hybrid GO Term Predictor (MMseqs2 + k-mer Jaccard + Reranker)
LAFA-compatible entrypoint.

Usage:
    python predict.py --fasta input.fasta --train_fasta train.fasta \
                      --train_terms train_terms.tsv --ia ia.tsv \
                      --output predictions.tsv

Input:
    --fasta       : FASTA file with test protein sequences
    --train_fasta : FASTA file with training sequences
    --train_terms : TSV with columns [protein_id, GO_term, ontology] (no header)
    --ia          : TSV with columns [GO_term, IA_score] (no header)
    --output      : Output TSV path  (protein_id <TAB> GO_term <TAB> score)

Output format (tab-separated, no header):
    protein_id\tGO:XXXXXXX\t0.XXX
"""

import argparse
import gc
import math
import os
import pickle
import sys
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
ROOT_GENERIC = {"GO:0008150", "GO:0003674", "GO:0005575"}

CONFIG = {
    "k": 5,
    "vocab_size": 50000,
    "top_k_similar": 35,
    "min_jaccard": 0.08,
    "batch_size": 800,
    "mmseqs2_weight": 0.75,
    "rank_decay": 0.88,
    "sim_power": 1.35,
    "votes_weight": 0.10,
    "ia_weight": 0.45,
    "cov_boost": 0.03,
    "freq_penalty_threshold": 0.04,
    "freq_penalty_strength": 0.80,
    "max_global_terms_per_protein": 14,
    "min_score_floor": 0.02,
    "score_clip_max": 0.92,
    "gap_fill_only_low_confidence": True,
    "gap_fill_threshold": 8,
    "gap_fill_score": 0.06,
    "fallback_top_terms": 60,
    "fallback_min_terms": 6,
    "ontology_weights": {"P": 1.05, "F": 0.95, "C": 1.00},
    "final_score_compress": 0.88,
    "final_score_power": 1.10,
    "final_min_score": 0.01,
    "min_term_frequency": 10,
}


# ──────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────
def normalize_uniprot_id(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if "|" in s:
        parts = s.split("|")
        if len(parts) >= 2 and len(parts[1]) >= 4:
            s = parts[1].strip()
    if "-" in s:
        base, suf = s.split("-", 1)
        if suf.isdigit() and len(base) >= 6:
            s = base
    return s


def load_fasta(path: str) -> pd.DataFrame:
    records = []
    try:
        from Bio import SeqIO
        for r in SeqIO.parse(path, "fasta"):
            pid = normalize_uniprot_id(r.id)
            records.append({"Id": pid, "Sequence": str(r.seq)})
    except ImportError:
        # Fallback manual parser
        pid, seq_parts = None, []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(">"):
                    if pid:
                        records.append({"Id": pid, "Sequence": "".join(seq_parts)})
                    pid = normalize_uniprot_id(line[1:].split()[0])
                    seq_parts = []
                else:
                    seq_parts.append(line)
        if pid:
            records.append({"Id": pid, "Sequence": "".join(seq_parts)})
    return pd.DataFrame(records)


def calculate_physicochemical_features(sequence: str) -> np.ndarray:
    if not sequence:
        return np.zeros(8, dtype=np.float32)
    hydrophobic = set("AILMFWV")
    polar = set("STNQ")
    charged_pos = set("KRH")
    charged_neg = set("DE")
    aromatic = set("FYW")
    total = len(sequence)
    return np.array([
        total,
        sum(aa in hydrophobic for aa in sequence) / total,
        sum(aa in polar for aa in sequence) / total,
        sum(aa in charged_pos for aa in sequence) / total,
        sum(aa in charged_neg for aa in sequence) / total,
        sum(aa in aromatic for aa in sequence) / total,
        sequence.count("C") / total,
        sequence.count("P") / total,
    ], dtype=np.float32)


# ──────────────────────────────────────────────
# K-MER HELPERS
# ──────────────────────────────────────────────
def extract_kmers_set(sequence: str, k: int = 5) -> set:
    if not sequence or len(sequence) < k:
        return set()
    return set(sequence[i:i+k] for i in range(len(sequence)-k+1))


def sequence_to_sparse_vector(sequence: str, kmer_to_idx: dict, k: int = 5):
    kmers = extract_kmers_set(sequence, k)
    idx = [kmer_to_idx[x] for x in kmers if x in kmer_to_idx]
    data = [1] * len(idx)
    return idx, data


def find_top_similar_batch_kmer(query_vectors, train_matrix, top_k: int = 40):
    dot = query_vectors.dot(train_matrix.T).toarray()
    qsz = np.array(query_vectors.sum(axis=1)).ravel()[:, None]
    tsz = np.array(train_matrix.sum(axis=1)).ravel()[None, :]
    unions = np.maximum(qsz + tsz - dot, 1.0)
    jacc = dot / unions
    res = []
    for i in range(jacc.shape[0]):
        sims = jacc[i]
        mask = sims >= CONFIG["min_jaccard"]
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            res.append([])
            continue
        vals = sims[idxs]
        order = np.argsort(-vals)[:top_k]
        res.append([(int(idxs[j]), float(vals[j])) for j in order])
    return res


# ──────────────────────────────────────────────
# SCORING
# ──────────────────────────────────────────────
def emission_profile(max_sim: float) -> dict:
    if max_sim >= 0.35:
        return {"thr": 0.08, "cap": {"P": 10, "F": 7, "C": 7}, "guarantee": True}
    if max_sim >= 0.22:
        return {"thr": 0.10, "cap": {"P": 9, "F": 6, "C": 6}, "guarantee": True}
    if max_sim >= 0.14:
        return {"thr": 0.13, "cap": {"P": 7, "F": 5, "C": 5}, "guarantee": False}
    return {"thr": 0.16, "cap": {"P": 6, "F": 4, "C": 4}, "guarantee": False}


def score_terms_reranker(
    similar_proteins,
    ontology: str,
    train_ids,
    protein_to_terms: dict,
    ia_dict: dict,
    term_frequencies: dict,
    train_physchem: dict,
    ALLOWED: dict,
    query_features=None,
) -> dict:
    if not similar_proteins:
        return {}

    term_sum = defaultdict(float)
    term_votes = Counter()
    rank_decay = CONFIG["rank_decay"]
    sim_power = CONFIG["sim_power"]
    votes_w = CONFIG["votes_weight"]
    ia_w = CONFIG["ia_weight"]

    for rank, (tidx, sim) in enumerate(similar_proteins):
        if sim <= 0:
            continue
        pid = train_ids[tidx]
        terms = protein_to_terms.get(pid, {}).get(ontology, set())
        if not terms:
            continue
        w_rank = rank_decay ** rank
        w_sim = sim ** sim_power
        neighbor_w = w_sim * w_rank

        phys_bonus = 1.0
        if query_features is not None and pid in train_physchem:
            trf = train_physchem[pid]
            dist = float(np.linalg.norm(query_features - trf))
            max_dist = math.sqrt(len(query_features))
            psim = max(0.0, 1.0 - dist / max_dist)
            phys_bonus = 1.0 + (psim * CONFIG["cov_boost"])

        for term in terms:
            if term not in ALLOWED[ontology]:
                continue
            if term in ROOT_GENERIC:
                continue
            term_votes[term] += 1
            ia = float(ia_dict.get(term, 0.5))
            s = neighbor_w * (1.0 + ia_w * ia) * phys_bonus
            f = term_frequencies.get(term, 0.0)
            if f >= CONFIG["freq_penalty_threshold"]:
                penalty = (1.0 - min(0.95, f * CONFIG["freq_penalty_strength"])) ** 0.5
                s *= penalty
            term_sum[term] += s

    if not term_sum:
        return {}

    mv = max(term_votes.values()) if term_votes else 0
    if mv > 0:
        for term in term_sum:
            v = term_votes[term] / mv
            term_sum[term] += votes_w * v

    mx = max(term_sum.values())
    if mx > 0:
        for term in list(term_sum.keys()):
            x = term_sum[term] / mx
            sc = (CONFIG["min_score_floor"] + 0.98 * x) ** 2.05
            ia = float(ia_dict.get(term, 0.5))
            sc *= 0.68 + 0.52 * ia
            sc *= CONFIG["ontology_weights"][ontology]
            sc = float(np.clip(sc, 0.0, CONFIG["score_clip_max"]))
            term_sum[term] = sc

    return dict(term_sum)


def apply_gap_filling(term_scores: dict, ontology: str, max_sim: float, fallback_cache: dict) -> dict:
    if CONFIG["gap_fill_only_low_confidence"] and max_sim > 0.14:
        return term_scores
    if len(term_scores) >= CONFIG["gap_fill_threshold"]:
        return term_scores
    need = min(CONFIG["gap_fill_threshold"] - len(term_scores), 6)
    added = 0
    for term, _freq, _ia, comb in fallback_cache[ontology]:
        if term in term_scores:
            continue
        sc = float(np.clip(CONFIG["gap_fill_score"] * (0.45 + comb), 0.0, 0.22))
        term_scores[term] = sc
        added += 1
        if added >= need:
            break
    return term_scores


def ensure_min_terms(picked: list, ontology: str, min_n: int, fallback_cache: dict) -> list:
    if len(picked) >= min_n:
        return picked
    have = set(t for t, _ in picked)
    for term, _f, _ia, comb in fallback_cache[ontology]:
        if term in have:
            continue
        sc = float(np.clip(0.10 + 0.18 * comb, 0.0, 0.28))
        picked.append((term, sc))
        have.add(term)
        if len(picked) >= min_n:
            break
    return picked


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CAFA-6 MMseqs2+Kmer Hybrid GO Term Predictor")
    parser.add_argument("--fasta",        required=True, help="Test FASTA file")
    parser.add_argument("--train_fasta",  required=True, help="Training FASTA file")
    parser.add_argument("--train_terms",  required=True, help="Training GO terms TSV (protein_id, GO_term, ontology)")
    parser.add_argument("--ia",           required=True, help="Information Accretion TSV (GO_term, IA_score)")
    parser.add_argument("--output",       required=True, help="Output predictions TSV")
    args = parser.parse_args()

    print("=" * 60)
    print("  CAFA-6 Hybrid Predictor — LAFA Edition")
    print("=" * 60)

    # ── Load sequences ──
    print("\n[1/7] Loading sequences...")
    test_seq_df  = load_fasta(args.fasta)
    train_seq_df = load_fasta(args.train_fasta)
    test_ids  = np.array(test_seq_df["Id"].values, dtype=object)
    train_ids = np.array(train_seq_df["Id"].values, dtype=object)
    print(f"  Train: {len(train_ids):,} | Test: {len(test_ids):,}")

    # ── Load GO terms ──
    print("\n[2/7] Loading GO terms and IA scores...")
    train_terms = pd.read_csv(args.train_terms, sep="\t", header=None, names=["Id", "Term", "Ontology"])
    train_terms["Id"]   = train_terms["Id"].astype(str).map(normalize_uniprot_id)
    train_terms["Term"] = train_terms["Term"].astype(str)

    ia_df = pd.read_csv(args.ia, sep="\t", header=None, names=["Term", "IA_Score"])
    ia_df["IA_Score"] = pd.to_numeric(ia_df["IA_Score"], errors="coerce").fillna(0.5).astype(np.float32)
    ia_dict = dict(zip(ia_df["Term"].values, ia_df["IA_Score"].values))
    print(f"  GO annotations: {len(train_terms):,} | IA terms: {len(ia_dict):,}")

    # ── Build ALLOWED sets ──
    print("\n[3/7] Building allowed GO term sets...")
    ALLOWED = {}
    for ont in ["P", "F", "C"]:
        df_ont = train_terms[train_terms["Ontology"] == ont]
        vc = df_ont["Term"].value_counts()
        ALLOWED[ont] = set(vc[vc >= CONFIG["min_term_frequency"]].index)
    ALLOWED_ALL = ALLOWED["P"] | ALLOWED["F"] | ALLOWED["C"]
    print(f"  P={len(ALLOWED['P'])} | F={len(ALLOWED['F'])} | C={len(ALLOWED['C'])}")

    # ── Build protein->terms map ──
    protein_to_terms = {pid: {"P": set(), "F": set(), "C": set()} for pid in train_ids}
    for ont in ["P", "F", "C"]:
        grouped = train_terms[train_terms["Ontology"] == ont].groupby("Id")["Term"].apply(list).to_dict()
        for pid, terms in grouped.items():
            if pid in protein_to_terms:
                protein_to_terms[pid][ont] = set(terms) & ALLOWED[ont]

    # ── Term frequencies ──
    total_proteins = len(train_ids)
    term_frequencies = {}
    for ont in ["P", "F", "C"]:
        vc = train_terms[train_terms["Ontology"] == ont]["Term"].value_counts()
        for term, count in vc.items():
            term_frequencies[term] = float(count) / float(total_proteins)

    # ── Fallback cache ──
    print("\n[4/7] Building fallback cache...")
    fallback_cache = {}
    for ont in ["P", "F", "C"]:
        term_counts = train_terms[train_terms["Ontology"] == ont]["Term"].value_counts()
        bag = []
        for term, count in term_counts.head(CONFIG["fallback_top_terms"]).items():
            if term not in ALLOWED[ont] or term in ROOT_GENERIC:
                continue
            freq = float(count) / float(total_proteins)
            ia   = float(ia_dict.get(term, 0.5))
            comb = 0.75 * freq + 0.25 * ia
            bag.append((term, freq, ia, comb))
        bag.sort(key=lambda x: x[3], reverse=True)
        fallback_cache[ont] = bag

    # ── PhysChem ──
    print("\n[5/7] Computing physicochemical features...")
    train_physchem = {}
    for _, r in tqdm(train_seq_df.iterrows(), total=len(train_seq_df), desc="  Train physchem"):
        train_physchem[r["Id"]] = calculate_physicochemical_features(r["Sequence"])
    train_seq_len = {pid: int(v[0]) for pid, v in train_physchem.items()}

    # ── K-mer matrix ──
    print("\n[6/7] Building k-mer matrix...")
    k = CONFIG["k"]
    kmer_counts = Counter()
    for seq in tqdm(train_seq_df["Sequence"].values, desc="  Vocab"):
        kmer_counts.update(extract_kmers_set(seq, k))
    top_kmers = [km for km, _ in kmer_counts.most_common(CONFIG["vocab_size"])]
    kmer_to_idx = {km: i for i, km in enumerate(top_kmers)}
    V = len(kmer_to_idx)

    rows = []
    for seq in tqdm(train_seq_df["Sequence"].values, desc="  Train matrix"):
        idx, data = sequence_to_sparse_vector(seq, kmer_to_idx, k)
        rows.append(csr_matrix((data, (np.zeros(len(idx), dtype=np.int32), idx)), shape=(1, V)))
    train_kmer_matrix = vstack(rows).tocsr()
    del rows
    gc.collect()

    train_id_to_idx = {pid: i for i, pid in enumerate(train_ids)}

    # ── Prediction ──
    print("\n[7/7] Predicting GO terms...")
    submission_rows = []
    batch_size = CONFIG["batch_size"]
    n_batches = (len(test_seq_df) + batch_size - 1) // batch_size

    for b in tqdm(range(n_batches), desc="  Batches"):
        start = b * batch_size
        end   = min(start + batch_size, len(test_seq_df))
        batch_df = test_seq_df.iloc[start:end]

        vecs, qfeats = [], []
        for seq in batch_df["Sequence"].values:
            idx, data = sequence_to_sparse_vector(seq, kmer_to_idx, k)
            vecs.append(csr_matrix((data, (np.zeros(len(idx), dtype=np.int32), idx)), shape=(1, V)))
            qfeats.append(calculate_physicochemical_features(seq))

        batch_mat = vstack(vecs).tocsr()
        kmer_hits_batch = find_top_similar_batch_kmer(batch_mat, train_kmer_matrix, top_k=CONFIG["top_k_similar"])

        for i, (_, r) in enumerate(batch_df.iterrows()):
            pid    = r["Id"]
            qfeat  = qfeats[i]
            neighbors = kmer_hits_batch[i]
            max_sim = neighbors[0][1] if neighbors else 0.0
            prof = emission_profile(max_sim)

            picked_all = []
            for ont in ["P", "F", "C"]:
                thr = prof["thr"]
                cap = prof["cap"][ont]

                if neighbors:
                    ts = score_terms_reranker(
                        neighbors, ont, train_ids,
                        protein_to_terms, ia_dict, term_frequencies,
                        train_physchem, ALLOWED, qfeat,
                    )
                    ts = apply_gap_filling(ts, ont, max_sim, fallback_cache)
                    picked = [(t, float(s)) for t, s in ts.items() if s >= thr and t in ALLOWED[ont]]
                    picked.sort(key=lambda x: x[1], reverse=True)
                    picked = picked[:cap]
                    if prof["guarantee"]:
                        picked = ensure_min_terms(picked, ont, max(2, cap // 3), fallback_cache)
                else:
                    picked = []
                    for term, _f, _ia, comb in fallback_cache[ont][:CONFIG["fallback_min_terms"]]:
                        if term in ROOT_GENERIC:
                            continue
                        sc = float(np.clip(0.10 + 0.12 * comb, 0.0, 0.25))
                        picked.append((term, sc))

                for term, sc in picked:
                    picked_all.append({"Id": pid, "Term": term, "Score": float(sc)})

            # Global cap + final compression
            picked_all.sort(key=lambda x: x["Score"], reverse=True)
            picked_all = picked_all[:CONFIG["max_global_terms_per_protein"]]
            for row in picked_all:
                s = max(CONFIG["final_min_score"], row["Score"])
                s = (s ** CONFIG["final_score_power"]) * CONFIG["final_score_compress"]
                row["Score"] = round(float(np.clip(s, 0.0, CONFIG["score_clip_max"])), 3)

            submission_rows.extend(picked_all)

        if b % 10 == 0:
            gc.collect()

    # ── Write output ──
    sub = pd.DataFrame(submission_rows)
    sub = sub[sub["Term"].isin(ALLOWED_ALL)]
    sub = sub[~sub["Term"].isin(ROOT_GENERIC)]
    sub["Score"] = sub["Score"].clip(0.0, 1.0)
    sub = sub.sort_values(["Id", "Term", "Score"], ascending=[True, True, False])
    sub = sub.drop_duplicates(subset=["Id", "Term"], keep="first")
    sub = sub.sort_values(["Id", "Score"], ascending=[True, False])
    sub = sub.groupby("Id").head(CONFIG["max_global_terms_per_protein"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sub.to_csv(args.output, sep="\t", header=False, index=False)

    print(f"\n✅ Output: {args.output}")
    print(f"   Proteins: {sub['Id'].nunique():,} / {len(test_ids):,}")
    print(f"   Rows:     {len(sub):,}")
    print(f"   Avg terms/protein: {len(sub)/max(1, sub['Id'].nunique()):.2f}")
    print(f"   Score range: {sub['Score'].min():.3f} – {sub['Score'].max():.3f}")


if __name__ == "__main__":
    main()

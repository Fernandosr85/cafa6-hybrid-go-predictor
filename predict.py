"""
CAFA-6 Hybrid GO Term Predictor (MMseqs2 + k-mer Jaccard + Reranker)
With GO ontology closure (ancestor propagation) for LAFA compatibility.

Usage:
    python predict.py --fasta input.fasta --train_fasta train.fasta \
                      --train_terms train_terms.tsv --ia ia.tsv \
                      --output predictions.tsv \
                      [--obo go-basic.obo]

Input:
    --fasta       : FASTA file with test protein sequences
    --train_fasta : FASTA file with training sequences
    --train_terms : TSV [protein_id, GO_term, ontology] (no header)
    --ia          : TSV [GO_term, IA_score] (no header)
    --obo         : (optional) path to go-basic.obo; auto-downloaded if absent
    --output      : Output TSV path (protein_id <TAB> GO_term <TAB> score)

Output format (tab-separated, no header):
    protein_id\tGO:XXXXXXX\t0.XXX
"""

import argparse
import gc
import math
import os
import sys
import urllib.request
import warnings
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
ROOT_GENERIC = {"GO:0008150", "GO:0003674", "GO:0005575"}

ONT_NAMESPACE = {
    "biological_process": "P",
    "molecular_function": "F",
    "cellular_component": "C",
}

GO_OBO_URL = "http://purl.obolibrary.org/obo/go/go-basic.obo"

CONFIG = {
    "k": 5,
    "vocab_size": 50000,
    "top_k_similar": 35,
    "min_jaccard": 0.08,
    "batch_size": 800,
    "rank_decay": 0.88,
    "sim_power": 1.35,
    "votes_weight": 0.10,
    "ia_weight": 0.45,
    "cov_boost": 0.03,
    "freq_penalty_threshold": 0.04,
    "freq_penalty_strength": 0.80,
    "max_global_terms_per_protein": 20,   # raised from 14 — closure adds ancestors
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
    # Closure config
    "closure_ancestor_score_fraction": 0.8,  # ancestor score = child_score * fraction
    "closure_max_depth": 6,                  # max ancestor hops to propagate
}


# ──────────────────────────────────────────────
# GO ONTOLOGY LOADER + CLOSURE
# ──────────────────────────────────────────────
def load_go_obo(obo_path: str = None) -> nx.MultiDiGraph:
    """
    Load GO ontology from file or download go-basic.obo automatically.
    Returns a networkx MultiDiGraph (child -> parent edges = is_a / part_of).
    """
    try:
        import obonet
    except ImportError:
        print("  Installing obonet...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "obonet", "--quiet"])
        import obonet

    if obo_path and os.path.exists(obo_path):
        print(f"  Loading GO ontology from: {obo_path}")
        graph = obonet.read_obo(obo_path)
    else:
        cache_path = "/tmp/go-basic.obo"
        if os.path.exists(cache_path):
            print(f"  Loading cached GO ontology: {cache_path}")
        else:
            print(f"  Downloading GO ontology from {GO_OBO_URL} ...")
            urllib.request.urlretrieve(GO_OBO_URL, cache_path)
            print(f"  Saved to {cache_path}")
        graph = obonet.read_obo(cache_path)

    print(f"  GO graph: {graph.number_of_nodes():,} terms, {graph.number_of_edges():,} edges")
    return graph


def build_ancestor_cache(go_graph: nx.MultiDiGraph, allowed_all: set) -> dict:
    """
    Pre-compute ancestors for every term in allowed_all.
    In obonet, edges go child -> parent (is_a/part_of),
    so nx.ancestors() gives all parents up to root.
    Returns: {term -> set of ancestor GO terms}
    """
    print("  Building ancestor cache...")
    cache = {}
    for term in tqdm(allowed_all, desc="  Ancestor cache"):
        if term not in go_graph:
            cache[term] = set()
            continue
        try:
            anc = nx.ancestors(go_graph, term)
            # Keep only GO terms (filter out meta-nodes)
            anc = {a for a in anc if isinstance(a, str) and a.startswith("GO:")}
        except Exception:
            anc = set()
        cache[term] = anc
    return cache


def get_term_ontology(go_graph: nx.MultiDiGraph, term: str) -> str:
    """Return ontology letter (P/F/C) for a GO term, or None if unknown."""
    if term not in go_graph.nodes:
        return None
    ns = go_graph.nodes[term].get("namespace", "")
    return ONT_NAMESPACE.get(ns, None)


def apply_go_closure(
    predictions: list,          # [{"Id":..,"Term":..,"Score":..}]
    ancestor_cache: dict,
    go_graph: nx.MultiDiGraph,
    allowed_all: set,
    score_fraction: float = 0.8,
    max_depth: int = 6,
) -> list:
    """
    For each predicted (term, score), propagate score to all ancestors
    with exponential decay: ancestor_score = child_score * fraction^depth.
    Takes the MAX score at each term (never lowers an existing score).
    Returns expanded list deduplicated by max score.
    """
    # Group by protein
    by_protein = defaultdict(dict)  # pid -> {term: score}
    for row in predictions:
        pid, term, score = row["Id"], row["Term"], row["Score"]
        if score > by_protein[pid].get(term, -1.0):
            by_protein[pid][term] = score

    expanded = []
    for pid, term_scores in by_protein.items():
        merged = dict(term_scores)  # start with direct predictions

        for term, score in term_scores.items():
            ancestors = ancestor_cache.get(term, set())
            # Walk ancestors — compute depth via shortest path
            for anc in ancestors:
                if anc in ROOT_GENERIC:
                    continue
                if anc not in allowed_all:
                    continue
                # Estimate depth: use BFS hop count
                try:
                    depth = nx.shortest_path_length(go_graph, term, anc)
                except Exception:
                    depth = 1
                if depth > max_depth:
                    continue
                anc_score = score * (score_fraction ** depth)
                if anc_score > merged.get(anc, -1.0):
                    merged[anc] = anc_score

        for term, score in merged.items():
            expanded.append({"Id": pid, "Term": term, "Score": score})

    return expanded


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
    hydrophobic  = set("AILMFWV")
    polar        = set("STNQ")
    charged_pos  = set("KRH")
    charged_neg  = set("DE")
    aromatic     = set("FYW")
    total = len(sequence)
    return np.array([
        total,
        sum(aa in hydrophobic for aa in sequence) / total,
        sum(aa in polar       for aa in sequence) / total,
        sum(aa in charged_pos for aa in sequence) / total,
        sum(aa in charged_neg for aa in sequence) / total,
        sum(aa in aromatic    for aa in sequence) / total,
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
    idx   = [kmer_to_idx[x] for x in kmers if x in kmer_to_idx]
    data  = [1] * len(idx)
    return idx, data


def find_top_similar_batch_kmer(query_vectors, train_matrix, top_k: int = 40):
    dot  = query_vectors.dot(train_matrix.T).toarray()
    qsz  = np.array(query_vectors.sum(axis=1)).ravel()[:, None]
    tsz  = np.array(train_matrix.sum(axis=1)).ravel()[None, :]
    unions = np.maximum(qsz + tsz - dot, 1.0)
    jacc = dot / unions
    res  = []
    for i in range(jacc.shape[0]):
        sims  = jacc[i]
        mask  = sims >= CONFIG["min_jaccard"]
        idxs  = np.where(mask)[0]
        if len(idxs) == 0:
            res.append([])
            continue
        vals  = sims[idxs]
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
        return {"thr": 0.10, "cap": {"P": 9,  "F": 6, "C": 6}, "guarantee": True}
    if max_sim >= 0.14:
        return {"thr": 0.13, "cap": {"P": 7,  "F": 5, "C": 5}, "guarantee": False}
    return     {"thr": 0.16, "cap": {"P": 6,  "F": 4, "C": 4}, "guarantee": False}


def score_terms_reranker(
    similar_proteins, ontology, train_ids,
    protein_to_terms, ia_dict, term_frequencies,
    train_physchem, ALLOWED, query_features=None,
) -> dict:
    if not similar_proteins:
        return {}

    term_sum   = defaultdict(float)
    term_votes = Counter()
    rank_decay = CONFIG["rank_decay"]
    sim_power  = CONFIG["sim_power"]
    votes_w    = CONFIG["votes_weight"]
    ia_w       = CONFIG["ia_weight"]

    for rank, (tidx, sim) in enumerate(similar_proteins):
        if sim <= 0:
            continue
        pid   = train_ids[tidx]
        terms = protein_to_terms.get(pid, {}).get(ontology, set())
        if not terms:
            continue
        w_rank     = rank_decay ** rank
        w_sim      = sim ** sim_power
        neighbor_w = w_sim * w_rank

        phys_bonus = 1.0
        if query_features is not None and pid in train_physchem:
            trf      = train_physchem[pid]
            dist     = float(np.linalg.norm(query_features - trf))
            max_dist = math.sqrt(len(query_features))
            psim     = max(0.0, 1.0 - dist / max_dist)
            phys_bonus = 1.0 + psim * CONFIG["cov_boost"]

        for term in terms:
            if term not in ALLOWED[ontology] or term in ROOT_GENERIC:
                continue
            term_votes[term] += 1
            ia = float(ia_dict.get(term, 0.5))
            s  = neighbor_w * (1.0 + ia_w * ia) * phys_bonus
            f  = term_frequencies.get(term, 0.0)
            if f >= CONFIG["freq_penalty_threshold"]:
                penalty = (1.0 - min(0.95, f * CONFIG["freq_penalty_strength"])) ** 0.5
                s *= penalty
            term_sum[term] += s

    if not term_sum:
        return {}

    mv = max(term_votes.values()) if term_votes else 0
    if mv > 0:
        for term in term_sum:
            term_sum[term] += CONFIG["votes_weight"] * term_votes[term] / mv

    mx = max(term_sum.values())
    if mx > 0:
        for term in list(term_sum.keys()):
            x  = term_sum[term] / mx
            sc = (CONFIG["min_score_floor"] + 0.98 * x) ** 2.05
            sc *= 0.68 + 0.52 * float(ia_dict.get(term, 0.5))
            sc *= CONFIG["ontology_weights"][ontology]
            term_sum[term] = float(np.clip(sc, 0.0, CONFIG["score_clip_max"]))

    return dict(term_sum)


def apply_gap_filling(term_scores, ontology, max_sim, fallback_cache) -> dict:
    if CONFIG["gap_fill_only_low_confidence"] and max_sim > 0.14:
        return term_scores
    if len(term_scores) >= CONFIG["gap_fill_threshold"]:
        return term_scores
    need  = min(CONFIG["gap_fill_threshold"] - len(term_scores), 6)
    added = 0
    for term, _f, _ia, comb in fallback_cache[ontology]:
        if term in term_scores:
            continue
        term_scores[term] = float(np.clip(CONFIG["gap_fill_score"] * (0.45 + comb), 0.0, 0.22))
        added += 1
        if added >= need:
            break
    return term_scores


def ensure_min_terms(picked, ontology, min_n, fallback_cache) -> list:
    if len(picked) >= min_n:
        return picked
    have = set(t for t, _ in picked)
    for term, _f, _ia, comb in fallback_cache[ontology]:
        if term in have:
            continue
        picked.append((term, float(np.clip(0.10 + 0.18 * comb, 0.0, 0.28))))
        have.add(term)
        if len(picked) >= min_n:
            break
    return picked


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CAFA-6 MMseqs2+Kmer Hybrid GO Term Predictor with GO Closure")
    parser.add_argument("--fasta",        required=True,  help="Test FASTA file")
    parser.add_argument("--train_fasta",  required=True,  help="Training FASTA file")
    parser.add_argument("--train_terms",  required=True,  help="Training GO terms TSV")
    parser.add_argument("--ia",           required=True,  help="Information Accretion TSV")
    parser.add_argument("--output",       required=True,  help="Output predictions TSV")
    parser.add_argument("--obo",          default=None,   help="Path to go-basic.obo (auto-downloaded if absent)")
    args = parser.parse_args()

    print("=" * 60)
    print("  CAFA-6 Hybrid Predictor + GO Closure — LAFA Edition")
    print("=" * 60)

    # ── 1. Load GO ontology ──
    print("\n[1/8] Loading GO ontology...")
    go_graph = load_go_obo(args.obo)

    # ── 2. Load sequences ──
    print("\n[2/8] Loading sequences...")
    test_seq_df  = load_fasta(args.fasta)
    train_seq_df = load_fasta(args.train_fasta)
    test_ids     = np.array(test_seq_df["Id"].values,  dtype=object)
    train_ids    = np.array(train_seq_df["Id"].values, dtype=object)
    print(f"  Train: {len(train_ids):,} | Test: {len(test_ids):,}")

    # ── 3. Load GO terms + IA ──
    print("\n[3/8] Loading GO terms and IA scores...")
    train_terms = pd.read_csv(args.train_terms, sep="\t", header=None, names=["Id", "Term", "Ontology"])
    train_terms["Id"]   = train_terms["Id"].astype(str).map(normalize_uniprot_id)
    train_terms["Term"] = train_terms["Term"].astype(str)

    ia_df = pd.read_csv(args.ia, sep="\t", header=None, names=["Term", "IA_Score"])
    ia_df["IA_Score"] = pd.to_numeric(ia_df["IA_Score"], errors="coerce").fillna(0.5).astype(np.float32)
    ia_dict = dict(zip(ia_df["Term"].values, ia_df["IA_Score"].values))
    print(f"  Annotations: {len(train_terms):,} | IA terms: {len(ia_dict):,}")

    # ── 4. Build ALLOWED sets + ancestor cache ──
    print("\n[4/8] Building allowed GO term sets + ancestor cache...")
    ALLOWED = {}
    for ont in ["P", "F", "C"]:
        vc = train_terms[train_terms["Ontology"] == ont]["Term"].value_counts()
        ALLOWED[ont] = set(vc[vc >= CONFIG["min_term_frequency"]].index)
    ALLOWED_ALL = ALLOWED["P"] | ALLOWED["F"] | ALLOWED["C"]
    print(f"  Allowed: P={len(ALLOWED['P'])} | F={len(ALLOWED['F'])} | C={len(ALLOWED['C'])}")

    # Expand ALLOWED_ALL to include ancestors of allowed terms
    # (so closure can propagate into them even if they were too rare to train on)
    print("  Expanding allowed set with GO graph ancestors...")
    extra_terms = set()
    for term in tqdm(ALLOWED_ALL, desc="  Expanding"):
        if term in go_graph:
            try:
                ancs = nx.ancestors(go_graph, term)
                for a in ancs:
                    if isinstance(a, str) and a.startswith("GO:") and a not in ROOT_GENERIC:
                        extra_terms.add(a)
            except Exception:
                pass
    ALLOWED_ALL_EXPANDED = ALLOWED_ALL | extra_terms
    print(f"  Expanded ALLOWED_ALL: {len(ALLOWED_ALL_EXPANDED):,} terms")

    # Build ancestor cache (for direct predictions only — saves memory)
    ancestor_cache = build_ancestor_cache(go_graph, ALLOWED_ALL)

    # ── 5. Build supporting structures ──
    print("\n[5/8] Building protein->terms map and frequencies...")
    protein_to_terms = {pid: {"P": set(), "F": set(), "C": set()} for pid in train_ids}
    for ont in ["P", "F", "C"]:
        grouped = train_terms[train_terms["Ontology"] == ont].groupby("Id")["Term"].apply(list).to_dict()
        for pid, terms in grouped.items():
            if pid in protein_to_terms:
                protein_to_terms[pid][ont] = set(terms) & ALLOWED[ont]

    total_proteins   = len(train_ids)
    term_frequencies = {}
    for ont in ["P", "F", "C"]:
        vc = train_terms[train_terms["Ontology"] == ont]["Term"].value_counts()
        for term, count in vc.items():
            term_frequencies[term] = float(count) / float(total_proteins)

    # Fallback cache
    fallback_cache = {}
    for ont in ["P", "F", "C"]:
        vc  = train_terms[train_terms["Ontology"] == ont]["Term"].value_counts()
        bag = []
        for term, count in vc.head(CONFIG["fallback_top_terms"]).items():
            if term not in ALLOWED[ont] or term in ROOT_GENERIC:
                continue
            freq = float(count) / float(total_proteins)
            ia   = float(ia_dict.get(term, 0.5))
            bag.append((term, freq, ia, 0.75*freq + 0.25*ia))
        bag.sort(key=lambda x: x[3], reverse=True)
        fallback_cache[ont] = bag

    # ── 6. PhysChem + k-mer matrix ──
    print("\n[6/8] Computing physicochemical features + k-mer matrix...")
    train_physchem = {}
    for _, r in tqdm(train_seq_df.iterrows(), total=len(train_seq_df), desc="  Train physchem"):
        train_physchem[r["Id"]] = calculate_physicochemical_features(r["Sequence"])

    k = CONFIG["k"]
    kmer_counts = Counter()
    for seq in tqdm(train_seq_df["Sequence"].values, desc="  Vocab"):
        kmer_counts.update(extract_kmers_set(seq, k))
    kmer_to_idx = {km: i for i, km in enumerate(km for km, _ in kmer_counts.most_common(CONFIG["vocab_size"]))}
    V = len(kmer_to_idx)

    rows = []
    for seq in tqdm(train_seq_df["Sequence"].values, desc="  Train matrix"):
        idx, data = sequence_to_sparse_vector(seq, kmer_to_idx, k)
        rows.append(csr_matrix((data, (np.zeros(len(idx), dtype=np.int32), idx)), shape=(1, V)))
    train_kmer_matrix = vstack(rows).tocsr()
    del rows
    gc.collect()
    train_id_to_idx = {pid: i for i, pid in enumerate(train_ids)}

    # ── 7. Predict (direct terms) ──
    print("\n[7/8] Predicting GO terms...")
    submission_rows = []
    batch_size = CONFIG["batch_size"]
    n_batches  = (len(test_seq_df) + batch_size - 1) // batch_size

    for b in tqdm(range(n_batches), desc="  Batches"):
        start    = b * batch_size
        end      = min(start + batch_size, len(test_seq_df))
        batch_df = test_seq_df.iloc[start:end]

        vecs, qfeats = [], []
        for seq in batch_df["Sequence"].values:
            idx, data = sequence_to_sparse_vector(seq, kmer_to_idx, k)
            vecs.append(csr_matrix((data, (np.zeros(len(idx), dtype=np.int32), idx)), shape=(1, V)))
            qfeats.append(calculate_physicochemical_features(seq))

        batch_mat       = vstack(vecs).tocsr()
        kmer_hits_batch = find_top_similar_batch_kmer(batch_mat, train_kmer_matrix, top_k=CONFIG["top_k_similar"])

        for i, (_, r) in enumerate(batch_df.iterrows()):
            pid       = r["Id"]
            qfeat     = qfeats[i]
            neighbors = kmer_hits_batch[i]
            max_sim   = neighbors[0][1] if neighbors else 0.0
            prof      = emission_profile(max_sim)

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
                    ts     = apply_gap_filling(ts, ont, max_sim, fallback_cache)
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
                        picked.append((term, float(np.clip(0.10 + 0.12 * comb, 0.0, 0.25))))

                for term, sc in picked:
                    picked_all.append({"Id": pid, "Term": term, "Score": float(sc)})

            # Global cap + final compression (pre-closure)
            picked_all.sort(key=lambda x: x["Score"], reverse=True)
            picked_all = picked_all[:CONFIG["max_global_terms_per_protein"]]
            for row in picked_all:
                s = max(CONFIG["final_min_score"], row["Score"])
                s = (s ** CONFIG["final_score_power"]) * CONFIG["final_score_compress"]
                row["Score"] = round(float(np.clip(s, 0.0, CONFIG["score_clip_max"])), 4)

            submission_rows.extend(picked_all)

        if b % 10 == 0:
            gc.collect()

    # ── 8. GO Closure + finalize ──
    print(f"\n[8/8] Applying GO closure (ancestor propagation)...")
    print(f"  Direct predictions: {len(submission_rows):,} rows")

    submission_rows = apply_go_closure(
        predictions     = submission_rows,
        ancestor_cache  = ancestor_cache,
        go_graph        = go_graph,
        allowed_all     = ALLOWED_ALL_EXPANDED,
        score_fraction  = CONFIG["closure_ancestor_score_fraction"],
        max_depth       = CONFIG["closure_max_depth"],
    )
    print(f"  After closure: {len(submission_rows):,} rows")

    # Finalize
    sub = pd.DataFrame(submission_rows)
    sub = sub[sub["Term"].isin(ALLOWED_ALL_EXPANDED)]
    sub = sub[~sub["Term"].isin(ROOT_GENERIC)]
    sub["Score"] = sub["Score"].clip(0.0, 1.0)
    sub = sub.sort_values(["Id", "Term", "Score"], ascending=[True, True, False])
    sub = sub.drop_duplicates(subset=["Id", "Term"], keep="first")
    sub = sub.sort_values(["Id", "Score"], ascending=[True, False])
    sub = sub.groupby("Id").head(CONFIG["max_global_terms_per_protein"]).reset_index(drop=True)

    # Ensure all test proteins appear
    present = set(sub["Id"].unique())
    missing = [pid for pid in test_ids if pid not in present]
    if missing:
        fill = []
        for pid in missing:
            for ont in ["P", "F", "C"]:
                for term, _f, _ia, comb in fallback_cache[ont][:2]:
                    fill.append({"Id": pid, "Term": term,
                                 "Score": round(float(np.clip(0.10 + 0.10*comb, 0.0, 0.22)), 4)})
        sub = pd.concat([sub, pd.DataFrame(fill)], ignore_index=True)
        sub = sub.drop_duplicates(subset=["Id", "Term"], keep="first")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sub.to_csv(args.output, sep="\t", header=False, index=False)

    print(f"\n✅ Output: {args.output}")
    print(f"   Proteins : {sub['Id'].nunique():,} / {len(test_ids):,}")
    print(f"   Total rows: {len(sub):,}")
    print(f"   Avg terms/protein: {len(sub)/max(1, sub['Id'].nunique()):.2f}")
    print(f"   Score range: {sub['Score'].min():.3f} – {sub['Score'].max():.3f}")


if __name__ == "__main__":
    main()

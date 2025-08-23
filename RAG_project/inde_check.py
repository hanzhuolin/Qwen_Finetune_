# -*- coding: utf-8 -*-
import os
import re
import csv
import json
import math
import argparse
from typing import List, Dict, Any, Iterable, Tuple, Optional

import torch
from difflib import SequenceMatcher
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------
# Utils for eval dataset I/O
# ---------------------------
def load_eval_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Support JSONL or CSV.
    Each example must include:
      - 'query': str
      - either 'relevant_ids' (list[str] or comma-separated) OR 'relevant_texts' (list[str] or comma-separated)
    """
    data = []
    _, ext = os.path.splitext(path.lower())
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                data.append(obj)
    elif ext == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # normalize lists when provided as comma-separated
                if "relevant_ids" in row and isinstance(row["relevant_ids"], str):
                    row["relevant_ids"] = [x.strip() for x in row["relevant_ids"].split(",") if x.strip()]
                if "relevant_texts" in row and isinstance(row["relevant_texts"], str):
                    row["relevant_texts"] = [x.strip() for x in row["relevant_texts"].split(",") if x.strip()]
                data.append(row)
    else:
        raise ValueError("Unsupported eval file extension. Use .jsonl or .csv")
    return data


# ---------------------------
# Fuzzy text matching helpers
# ---------------------------
def jaccard_like(a: str, b: str) -> float:
    """A simple character-level overlap score (robust to short spans)."""
    a_set = set(a.lower())
    b_set = set(b.lower())
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def fuzzy_match_one(doc_text: str, gold_text: str, threshold: float = 0.6) -> bool:
    lt = (doc_text or "").lower()
    lg = (gold_text or "").strip().lower()
    if not lg:
        return False
    if lg in lt or lt in lg:
        return True
    sim = SequenceMatcher(None, lt, lg).ratio()
    if sim >= threshold:
        return True
    if jaccard_like(lt, lg) >= threshold:
        return True
    return False


# ---------------------------
# Metrics (with per-query details)
# ---------------------------
def compute_metrics_per_query(
    retrieved: List[Any],
    gold_ids: List[str],
    gold_texts: List[str],
    k_values: List[int],
    match_field: str,
    text_match_threshold: float
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Compute metrics and return (metrics_dict, details_dict) for one query.
    details_dict includes:
      - rel_vector (per-rank 0/1)
      - first_hit_rank
      - matched_gold_ids or matched_gold_texts
    """
    # --- normalize gold sets ---
    gold_ids_set = set([g.strip() for g in (gold_ids or []) if isinstance(g, str) and g.strip()])
    use_id = len(gold_ids_set) > 0
    gold_texts_norm = [t.strip() for t in (gold_texts or []) if isinstance(t, str) and t.strip()]
    num_rel = len(gold_ids_set) if use_id else len(gold_texts_norm)
    num_rel = max(num_rel, 1)  # avoid division by zero

    max_k = max(k_values) if k_values else len(retrieved)
    cut = retrieved[:max_k]

    rel: List[int] = []
    first_hit_rank: Optional[int] = None
    matched_gold_text_indices = set()
    matched_gold_texts: List[str] = []
    matched_gold_ids: List[str] = []

    for rank, doc in enumerate(cut, start=1):
        if use_id:
            did = None
            if isinstance(doc.metadata, dict):
                did = doc.metadata.get(match_field)
                if isinstance(did, (list, tuple)):
                    did = next((x for x in did if isinstance(x, str)), None)
            hit = int(bool(did and did in gold_ids_set))
            if hit == 1:
                matched_gold_ids.append(did)
                if first_hit_rank is None:
                    first_hit_rank = rank
            rel.append(hit)
        else:
            # text mode: each gold_text can be matched at most once
            doc_text = doc.page_content or ""
            hit = 0
            for j, g in enumerate(gold_texts_norm):
                if j in matched_gold_text_indices:
                    continue
                if fuzzy_match_one(doc_text, g, text_match_threshold):
                    matched_gold_text_indices.add(j)
                    matched_gold_texts.append(g)
                    hit = 1
                    if first_hit_rank is None:
                        first_hit_rank = rank
                    break
            rel.append(hit)

    # clamp total positives to num_rel (safety)
    total_pos = sum(rel)
    if total_pos > num_rel:
        surplus = total_pos - num_rel
        for i in range(len(rel) - 1, -1, -1):
            if surplus == 0:
                break
            if rel[i] == 1:
                rel[i] = 0
                surplus -= 1

    # --- RR & AP ---
    rr = 0.0
    ap = 0.0
    hits_so_far = 0
    for i, r in enumerate(rel, start=1):
        if r == 1 and rr == 0.0:
            rr = 1.0 / i
        if r == 1:
            hits_so_far += 1
            ap += hits_so_far / i
    ap = ap / num_rel  # AP ¡Ê [0,1]

    # --- NDCG@k with binary gains ---
    def dcg_at_k(k: int) -> float:
        return sum((rel[i - 1] / math.log2(i + 1)) for i in range(1, min(k, len(rel)) + 1))

    def idcg_at_k(k: int) -> float:
        ideal_ones = min(sum(rel), num_rel, k)  # upper bound by both positives and num_rel
        return sum((1.0 / math.log2(i + 1)) for i in range(1, ideal_ones + 1))

    out: Dict[str, float] = {"RR": rr, "AP": ap}
    for k in k_values:
        k_eff = min(k, len(rel)) if len(rel) > 0 else k
        topk = rel[:k_eff] if k_eff > 0 else []
        hits = sum(topk)
        precision = hits / max(k_eff, 1)
        recall = hits / num_rel
        dcg = dcg_at_k(k_eff)
        idcg = idcg_at_k(k_eff)
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        out[f"Hit@{k}"] = 1.0 if hits > 0 else 0.0
        out[f"Precision@{k}"] = precision
        out[f"Recall@{k}"] = recall
        out[f"NDCG@{k}"] = ndcg

    # details
    details: Dict[str, Any] = {
        "rel_vector": rel,
        "first_hit_rank": first_hit_rank,
        "matched_gold_ids": matched_gold_ids if use_id else None,
        "matched_gold_texts": matched_gold_texts if not use_id else None,
    }
    return out, details


def aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not all_metrics:
        return {}
    keys = sorted(all_metrics[0].keys())
    agg = {k: 0.0 for k in keys}
    for m in all_metrics:
        for k, v in m.items():
            agg[k] += float(v)
    n = float(len(all_metrics))
    for k in agg:
        agg[k] /= n
    # Friendly names
    agg["MRR"] = agg.get("RR", 0.0)
    agg["MAP"] = agg.get("AP", 0.0)
    return agg


# ---------------------------
# Evaluation runner (with per-query details)
# ---------------------------
def evaluate_retrieval(
    db: FAISS,
    dataset_path: str,
    k_values: List[int],
    match_field: str = "source",
    text_match_threshold: float = 0.6,
    details_output: Optional[str] = None,
    show_per_query: bool = True
) -> None:
    print("\n=== Retrieval Evaluation ===")
    eval_data = load_eval_dataset(dataset_path)
    print(f"Loaded eval set with {len(eval_data)} examples.")

    max_k = max(k_values)
    per_query_metrics: List[Dict[str, float]] = []
    per_query_details_out: List[Dict[str, Any]] = []

    for idx, ex in enumerate(eval_data, start=1):
        query = ex.get("query", "")
        if not query:
            continue

        # Normalize gold ids/texts
        gold_ids = ex.get("relevant_ids", []) or []
        gold_texts = ex.get("relevant_texts", []) or []
        if isinstance(gold_ids, str):
            gold_ids = [x.strip() for x in gold_ids.split(",") if x.strip()]
        if isinstance(gold_texts, str):
            gold_texts = [x.strip() for x in gold_texts.split(",") if x.strip()]

        # retrieve
        try:
            results = db.similarity_search(query, k=max_k)
        except Exception as e:
            print(f"[{idx}] Retrieval failed for query: {query[:60]}... Error: {e}")
            continue

        m, d = compute_metrics_per_query(
            retrieved=results,
            gold_ids=gold_ids,
            gold_texts=gold_texts,
            k_values=k_values,
            match_field=match_field,
            text_match_threshold=text_match_threshold
        )
        per_query_metrics.append(m)

        # assemble per-query detail record
        detail_record = {
            "index": idx,
            "query": query,
            "relevant_ids": gold_ids if gold_ids else None,
            "relevant_texts": gold_texts if gold_texts else None,
            "rel_vector": d["rel_vector"],
            "first_hit_rank": d["first_hit_rank"],
            "matched_gold_ids": d["matched_gold_ids"],
            "matched_gold_texts": d["matched_gold_texts"],
            "metrics": m
        }
        per_query_details_out.append(detail_record)

        if show_per_query:
            print("\n--- Query #{:d} ---".format(idx))
            print(f"Q: {query}")
            if gold_ids:
                print(f"Gold (IDs): {gold_ids}")
            if gold_texts:
                print(f"Gold (Texts): {gold_texts}")
            print(f"rel@top{max_k}: {d['rel_vector']}")
            print(f"First Hit Rank: {d['first_hit_rank']}")
            if d["matched_gold_ids"]:
                print(f"Matched gold IDs: {d['matched_gold_ids']}")
            if d["matched_gold_texts"]:
                print(f"Matched gold texts: {d['matched_gold_texts']}")
            # concise per-query metrics summary (top-1/3/5 if available)
            keys_brief = ["RR", "AP"] + [k for k in (f"Hit@1", f"Hit@3", f"Hit@5") if k in m]
            print("Metrics (brief): " + ", ".join([f"{k}={m[k]:.4f}" for k in keys_brief if k in m]))

    # aggregated
    summary = aggregate_metrics(per_query_metrics)
    if not summary:
        print("No metrics computed (empty eval or all failures).")
        return

    print("\n--- Aggregated Metrics ---")
    ordered = ["MRR", "MAP"] + \
              [f"Hit@{k}" for k in k_values] + \
              [f"Precision@{k}" for k in k_values] + \
              [f"Recall@{k}" for k in k_values] + \
              [f"NDCG@{k}" for k in k_values]

    for k in ordered:
        if k in summary:
            print(f"{k:>12}: {summary[k]:.4f}")

    # optional save details
    if details_output:
        with open(details_output, "w", encoding="utf-8") as f:
            for rec in per_query_details_out:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nPer-query details saved to: {details_output}")


# ---------------------------
# Sanity test (when no eval set)
# ---------------------------
def run_sanity_test(db, sanity_query: str = "hello"):
    print("Step 3: Validating index content and running a sanity test query...")
    docstore_size = len(db.docstore._dict)
    index_size = db.index.ntotal

    print(f"   - Total documents in docstore: {docstore_size}")
    print(f"   - Total vectors in index: {index_size}")

    if docstore_size == 0 or index_size == 0:
        print("WARNING: Index is loaded, but it contains no documents or vectors.")
        return

    if docstore_size != index_size:
        print("WARNING: Document count does not match vector count. The index might be corrupted.")

    print(f"\n   - Performing test query: '{sanity_query}'")
    results = db.similarity_search(sanity_query, k=1)

    if not results:
        print("ERROR: Test query returned no results. The index might be empty or malfunctioning.")
    else:
        print("Test query successful!")
        print("\n--- Example of the first retrieved document ---")
        print(results[0].page_content[:1000])
        print("---------------------------------------------")

    print("\n--- Conclusion: The FAISS index files appear to be healthy and usable! ---")


# ---------------------------
# Main checker
# ---------------------------
def check_faiss_index(
    index_path: str,
    embedding_model_path: str,
    eval_dataset: str = None,
    k_values: List[int] = None,
    match_field: str = "source",
    text_match_threshold: float = 0.6,
    sanity_query: str = "hello",
    details_output: Optional[str] = None,
    show_per_query: bool = True
):
    """
    Load and validate a FAISS index and its associated embedding model.
    Optionally evaluate retrieval with a labeled dataset (by IDs or texts),
    print per-query details, and save them to JSONL.
    """
    print("--- Starting FAISS Index Validation ---")

    # 1) Paths
    if not os.path.exists(index_path):
        print(f"ERROR: Index path does not exist -> {index_path}")
        return

    if not os.path.exists(embedding_model_path):
        print(f"ERROR: Embedding model path does not exist -> {embedding_model_path}")
        return

    faiss_file = os.path.join(index_path, "index.faiss")
    pkl_file = os.path.join(index_path, "index.pkl")

    if not os.path.exists(faiss_file):
        print(f"ERROR: 'index.faiss' file not found in the specified path: {index_path}")
        return

    if not os.path.exists(pkl_file):
        print(f"ERROR: 'index.pkl' file not found in the specified path: {index_path}")
        return

    print(f"All necessary paths and files have been found.")
    print("-" * 20)

    try:
        # 2) Embedding
        print(f"Step 1: Loading embedding model from '{embedding_model_path}'...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   - Using device: {device}")

        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("Embedding model loaded successfully.")
        print("-" * 20)

        # 3) FAISS index
        print(f"Step 2: Loading FAISS index from '{index_path}'...")
        db = FAISS.load_local(
            index_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        print("FAISS index loaded successfully.")
        print("-" * 20)

        # 4) Evaluate or sanity test
        if eval_dataset:
            print("\nEval dataset provided. Skipping sanity test query and proceeding to retrieval evaluation...")
            if k_values is None:
                k_values = [1, 3, 5, 10]
            evaluate_retrieval(
                db=db,
                dataset_path=eval_dataset,
                k_values=k_values,
                match_field=match_field,
                text_match_threshold=text_match_threshold,
                details_output=details_output,
                show_per_query=show_per_query
            )
        else:
            run_sanity_test(db, sanity_query=sanity_query)

    except Exception as e:
        print(f"\nA critical error occurred during the process: {e}")
        print("\n--- Conclusion: There is a problem with the index files. ---")
        print("Possible Causes:")
        print("  1. Embedding Model Mismatch: The model being loaded MUST be identical to the one used when the index was created. This is the most common cause of 'pickle' or 'deserialization' errors.")
        print("  2. Corrupted Files: The 'index.faiss' or 'index.pkl' files may have been corrupted during transfer or saving.")
        print("  3. Incompatible Library Versions: Significant version differences in libraries like `faiss` or `langchain` between index creation and loading.")


# ---------------------------
# CLI
# ---------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FAISS Index Validation & Retrieval Evaluation Tool (with per-query details)")
    parser.add_argument(
        "--index_path",
        type=str,
        required=True,
        help="The directory path containing index.faiss and index.pkl files."
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        required=True,
        help="The path to the embedding model used to create the index."
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default=None,
        help="Path to JSONL/CSV evaluation file with fields: query + (relevant_ids OR relevant_texts)."
    )
    parser.add_argument(
        "--k_values",
        type=str,
        default="1,3,5,10",
        help="Comma-separated list of k values for metrics, e.g., '1,3,5,10'."
    )
    parser.add_argument(
        "--match_field",
        type=str,
        default="source",
        help="Metadata key used to match doc IDs (when using relevant_ids)."
    )
    parser.add_argument(
        "--text_match_threshold",
        type=float,
        default=0.6,
        help="Similarity threshold for fuzzy text matching (when using relevant_texts)."
    )
    parser.add_argument(
        "--sanity_query",
        type=str,
        default="hello",
        help="Sanity test query used when no eval dataset is provided."
    )
    parser.add_argument(
        "--details_output",
        type=str,
        default=None,
        help="Optional path to save per-query details as JSONL."
    )
    parser.add_argument(
        "--no_show_per_query",
        action="store_true",
        help="If set, do not print per-query details to stdout."
    )
    args = parser.parse_args()

    k_vals = [int(x) for x in args.k_values.split(",") if x.strip()]

    check_faiss_index(
        index_path=args.index_path,
        embedding_model_path=args.embedding_path,
        eval_dataset=args.eval_dataset,
        k_values=k_vals,
        match_field=args.match_field,
        text_match_threshold=args.text_match_threshold,
        sanity_query=args.sanity_query,
        details_output=args.details_output,
        show_per_query=(not args.no_show_per_query)
    )
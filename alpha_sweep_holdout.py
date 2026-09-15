import json
import random
import numpy as np
from search import load_chunks, embed_text, cosine_similarity
from search_hybrid import build_bm25_index, tokenize
import re

random.seed(42)

with open("eval_set.json", "r") as f:
    eval_questions = json.load(f)

function_aware_chunks = load_chunks("chunks_function_aware.json")
answerable_questions = [q for q in eval_questions if "N/A" not in q["source"]]

bm25_index = build_bm25_index(function_aware_chunks)

def normalize(arr):
    if arr.max() - arr.min() == 0:
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())

# Stratified split: separate by type, shuffle each group, split each in half
by_type = {}
for q in answerable_questions:
    by_type.setdefault(q.get("type", "factual"), []).append(q)

tuning_set = []
holdout_set = []
for qtype, questions in by_type.items():
    shuffled = questions[:]
    random.shuffle(shuffled)
    midpoint = len(shuffled) // 2
    tuning_set.extend(shuffled[:midpoint] if len(shuffled) % 2 == 0 else shuffled[:midpoint + 1])
    holdout_set.extend(shuffled[midpoint:] if len(shuffled) % 2 == 0 else shuffled[midpoint + 1:])

print(f"Tuning set: {len(tuning_set)} questions ({[q.get('type') for q in tuning_set].count('multi_hop')} multi-hop)")
print(f"Holdout set: {len(holdout_set)} questions ({[q.get('type') for q in holdout_set].count('multi_hop')} multi-hop)\n")

# Precompute embeddings + BM25 scores once per question, for BOTH sets
def precompute(questions):
    entries = []
    for q in questions:
        query_embedding = embed_text(q["question"])
        semantic_scores = np.array([
            cosine_similarity(query_embedding, chunk["embedding"]) for chunk in function_aware_chunks
        ])
        keyword_scores = np.array(bm25_index.get_scores(tokenize(q["question"])))
        entries.append({
            "semantic_norm": normalize(semantic_scores),
            "keyword_norm": normalize(keyword_scores),
            "expected_files": re.findall(r'\b\w+\.py\b', q["source"]),
            "id": q["id"]
        })
    return entries

def check_at_alpha(entry, alpha, top_k=6):
    combined = alpha * entry["semantic_norm"] + (1 - alpha) * entry["keyword_norm"]
    top_indices = np.argsort(combined)[::-1][:top_k]
    retrieved = [function_aware_chunks[i]["source_file"] for i in top_indices]
    return all(ef in retrieved for ef in entry["expected_files"])

print("Precomputing tuning set...")
tuning_precomputed = precompute(tuning_set)
print("Precomputing holdout set...")
holdout_precomputed = precompute(holdout_set)

# Step 1: sweep alpha on TUNING set only, pick the best
alpha_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
print(f"\n--- TUNING (selecting alpha on {len(tuning_set)} questions) ---")
best_alpha = None
best_accuracy = -1
for alpha in alpha_values:
    correct = sum(1 for e in tuning_precomputed if check_at_alpha(e, alpha))
    accuracy = correct / len(tuning_precomputed)
    print(f"alpha={alpha}: {correct}/{len(tuning_precomputed)} = {accuracy:.2%}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_alpha = alpha

print(f"\nSelected alpha={best_alpha} (tuning accuracy: {best_accuracy:.2%})")

# Step 2: report that alpha's performance on the HOLDOUT set, which it never saw
print(f"\n--- HOLDOUT (evaluating selected alpha={best_alpha} on {len(holdout_set)} unseen questions) ---")
holdout_correct = sum(1 for e in holdout_precomputed if check_at_alpha(e, best_alpha))
holdout_accuracy = holdout_correct / len(holdout_precomputed)
print(f"alpha={best_alpha}: {holdout_correct}/{len(holdout_precomputed)} = {holdout_accuracy:.2%}")

# Also report pure semantic (alpha=1.0) on holdout for comparison
semantic_correct = sum(1 for e in holdout_precomputed if check_at_alpha(e, 1.0))
semantic_accuracy = semantic_correct / len(holdout_precomputed)
print(f"alpha=1.0 (pure semantic) on holdout: {semantic_correct}/{len(holdout_precomputed)} = {semantic_accuracy:.2%}")

print(f"\n--- SUMMARY ---")
print(f"Tuned alpha ({best_alpha}) accuracy on tuning set: {best_accuracy:.2%} (this number is optimistic, same data used to select it)")
print(f"Tuned alpha ({best_alpha}) accuracy on holdout set: {holdout_accuracy:.2%} (this is the honest number)")
print(f"Gap between tuning and holdout: {(best_accuracy - holdout_accuracy)*100:.2f} points")
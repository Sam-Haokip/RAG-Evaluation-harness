import json
import numpy as np
from search import load_chunks, embed_text, cosine_similarity
from search_hybrid import build_bm25_index, tokenize

with open("eval_set.json", "r") as f:
    eval_questions = json.load(f)

function_aware_chunks = load_chunks("chunks_function_aware.json")
answerable_questions = [q for q in eval_questions if "N/A" not in q["source"]]

bm25_index = build_bm25_index(function_aware_chunks)

def normalize(arr):
    if arr.max() - arr.min() == 0:
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())

# Embed every question ONCE, and compute BM25 scores ONCE - alpha only changes the blend
precomputed = []
for q in answerable_questions:
    query_embedding = embed_text(q["question"])
    semantic_scores = np.array([
        cosine_similarity(query_embedding, chunk["embedding"]) for chunk in function_aware_chunks
    ])
    keyword_scores = np.array(bm25_index.get_scores(tokenize(q["question"])))
    semantic_norm = normalize(semantic_scores)
    keyword_norm = normalize(keyword_scores)

    import re
    expected_files = re.findall(r'\b\w+\.py\b', q["source"])

    precomputed.append({
        "semantic_norm": semantic_norm,
        "keyword_norm": keyword_norm,
        "expected_files": expected_files
    })

def check_at_alpha(entry, alpha, top_k=6):
    combined_scores = alpha * entry["semantic_norm"] + (1 - alpha) * entry["keyword_norm"]
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    retrieved_sources = [function_aware_chunks[i]["source_file"] for i in top_indices]
    return all(ef in retrieved_sources for ef in entry["expected_files"])

alpha_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

print(f"Sweeping alpha across {len(answerable_questions)} questions (embeddings computed once)\n")

for alpha in alpha_values:
    correct = sum(1 for entry in precomputed if check_at_alpha(entry, alpha))
    accuracy = correct / len(precomputed)
    print(f"alpha={alpha}: {correct}/{len(precomputed)} = {accuracy:.2%}")
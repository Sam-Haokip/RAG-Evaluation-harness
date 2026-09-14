import numpy as np
from rank_bm25 import BM25Okapi
from search import embed_text, cosine_similarity, load_chunks

def tokenize(text):
    return text.lower().split()

def build_bm25_index(chunks):
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized_corpus)

def search_hybrid(query, chunks, bm25_index, top_k=6, alpha=0.7, tracker=None):
    # Semantic scores
    query_embedding = embed_text(query, tracker=tracker)
    semantic_scores = np.array([
        cosine_similarity(query_embedding, chunk["embedding"]) for chunk in chunks
    ])

    # Keyword scores
    keyword_scores = np.array(bm25_index.get_scores(tokenize(query)))

    # Normalize both to 0-1 range so they're comparable
    def normalize(arr):
        if arr.max() - arr.min() == 0:
            return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())

    semantic_norm = normalize(semantic_scores)
    keyword_norm = normalize(keyword_scores)

    # Weighted combination
    combined_scores = alpha * semantic_norm + (1 - alpha) * keyword_norm

    top_indices = np.argsort(combined_scores)[::-1][:top_k]

    return [(combined_scores[i], chunks[i]) for i in top_indices]


if __name__ == "__main__":
    chunks = load_chunks("chunks_function_aware.json")
    bm25 = build_bm25_index(chunks)

    test_question = "What does the utils.py function default_headers() return?"
    results = search_hybrid(test_question, chunks, bm25)

    print(f"Question: {test_question}\n")
    for score, chunk in results:
        print(f"Score: {score:.4f} | {chunk['source_file']} - {chunk.get('name', '')}")

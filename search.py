import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AICREDITS_API_KEY"),
    base_url="https://aicredits.in/v1"
)

def embed_text(text, tracker=None):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    if tracker is not None:
        tracker.record(response, "embedding")
    return response.data[0].embedding

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_chunks(filepath="chunks_with_embeddings.json"):
    with open(filepath, "r") as f:
        return json.load(f)

all_chunks = load_chunks()
print(f"Loaded {len(all_chunks)} chunks.")

def search(query, chunks=None, top_k=3, tracker=None):
    if chunks is None:
        chunks = all_chunks

    query_embedding = embed_text(query, tracker=tracker)

    scored_chunks = []
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    return scored_chunks[:top_k]

if __name__ == "__main__":
    test_question = "What does the raise_for_status method do when a request returns a 404 status code?"

    results = search(test_question)

    print(f"\nQuestion: {test_question}\n")
    for score, chunk in results:
        print(f"Score: {score:.4f} | Source: {chunk['source_file']} (chunk {chunk.get('chunk_index', '')})")
        print(chunk["text"][:200])
        print("---")

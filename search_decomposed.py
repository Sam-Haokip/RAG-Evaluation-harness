import os
from dotenv import load_dotenv
from openai import OpenAI
from search import search, cosine_similarity, embed_text

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AICREDITS_API_KEY"),
    base_url="https://aicredits.in/v1"
)

def decompose_query(question, tracker=None):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""This question is about how a codebase works internally.

First, check if the question is asking about a single, flat piece of information (like 
what a specific file contains, or what one constant equals) rather than a mechanism with 
multiple moving parts. If so, output the question itself, unchanged, as the only line - 
do not invent artificial sub-topics or general background concepts.

If the question genuinely involves multiple functions, methods, or files working together, 
identify AT MOST 3 distinct components involved and write exactly one focused search query 
per component - not rephrasings of the same idea, and not generic conceptual background.

Example - if asked "what happens when you call api.get(url)?", a good answer is exactly these 3 lines:
How the top-level get() function builds a request and hands it to a session
How the session selects a transport adapter and delegates to it
What the adapter does to actually send the request

Example - if asked "what does config.py contain?", a good answer is exactly this 1 line:
What does config.py contain?

Output ONLY the search queries themselves, one per line. No dashes, no bullets, no numbering, 
no explanation, no more than 3 lines total.

Question: {question}"""
        }]
    )
    if tracker is not None:
        tracker.record(response, "decomposition")

    raw = response.choices[0].message.content
    sub_queries = [line.strip().lstrip("-•*").strip() for line in raw.split("\n") if line.strip()]
    return sub_queries[:3]

def search_with_decomposition(question, chunks, top_k_per_query=3, tracker=None):
    sub_queries = decompose_query(question, tracker=tracker)

    all_results = {}
    for sq in sub_queries:
        results = search(sq, chunks=chunks, top_k=top_k_per_query, tracker=tracker)
        for score, chunk in results:
            key = (chunk["source_file"], chunk["name"])
            if key not in all_results or score > all_results[key][0]:
                all_results[key] = (score, chunk)

    combined = sorted(all_results.values(), key=lambda x: x[0], reverse=True)
    return combined, sub_queries

if __name__ == "__main__":
    from search import load_chunks
    function_aware_chunks = load_chunks("chunks_function_aware.json")

    test_question = "How does Session.send() delegate the actual network request, and why is it structured this way?"
    results, sub_queries = search_with_decomposition(test_question, function_aware_chunks)

    print("Sub-queries generated:")
    for sq in sub_queries:
        print(f"  - {sq}")

    print("\nCombined results:")
    for score, chunk in results:
        print(f"Score: {score:.4f} | Source: {chunk['source_file']} - {chunk.get('name', '')}")

import os
from dotenv import load_dotenv
from openai import OpenAI
from search import search, load_chunks

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AICREDITS_API_KEY"),
    base_url="https://aicredits.in/v1"
)

def generate_answer(question, chunks, top_k=6, tracker=None):
    results = search(question, chunks=chunks, top_k=top_k, tracker=tracker)

    context = "\n\n---\n\n".join(
        f"From {chunk['source_file']} ({chunk.get('name', 'unknown')}):\n{chunk['text']}"
        for score, chunk in results
    )

    prompt = f"""Answer the question using ONLY the code context below. 
If the context does not contain enough information to answer confidently, 
say clearly "I could not find this in the codebase" instead of guessing.
Context:
{context}
Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    if tracker is not None:
        tracker.record(response, "generation")

    return response.choices[0].message.content

if __name__ == "__main__":
     function_aware_chunks = load_chunks("chunks_function_aware.json")
     test_question = "What does the Session. retry_with_backoff() method do?"
     answer = generate_answer(test_question, function_aware_chunks)
     print(f"Question: {test_question}\n")
     print(f"Answer: {answer}")

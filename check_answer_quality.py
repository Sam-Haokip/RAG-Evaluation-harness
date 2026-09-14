import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from search import load_chunks, search
from generate_answer import generate_answer
import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AICREDITS_API_KEY"),
    base_url="https://aicredits.in/v1"
)

def judge_answer_lenient(question, expected_answer, generated_answer):
    prompt = f"""You are evaluating whether a generated answer is factually correct, 
based on an expected reference answer.
Question: {question}
Reference answer: {expected_answer}
Generated answer: {generated_answer}
Mark YES if the generated answer is factually correct and captures the main point, 
even if it is shorter, longer, differently worded, or omits some secondary details.
Mark NO only if it is factually wrong, contradicts the reference, or misses the core point entirely.
Reply with only YES or NO."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    verdict = response.choices[0].message.content.strip().upper()
    return "YES" in verdict

def judge_answer_strict(question, expected_answer, generated_answer):
    prompt = f"""Compare a candidate answer against a reference answer for a factual question 
about a codebase. Act as a skeptical reviewer, not a lenient one.

Question: {question}
Reference answer: {expected_answer}
Candidate answer: {generated_answer}

Does the candidate answer state the same core mechanism or fact as the reference, without 
inventing details the reference does not support? Wording, length, and omitted secondary 
details do not matter - only whether the central claim matches and nothing false was added.

Respond with a single word: CORRECT or INCORRECT."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    verdict = response.choices[0].message.content.strip().upper()
    return "CORRECT" in verdict and "INCORRECT" not in verdict

def judge_answer_against_source(question, retrieved_context, generated_answer):
    prompt = f"""You are checking whether a generated answer is factually supported by the 
source code it was retrieved from. You do NOT have access to any reference answer, and 
you should not penalize the generated answer for being more detailed, more specific, or 
differently scoped than any hypothetical reference would be.

Question: {question}

Retrieved source code context:
{retrieved_context}

Generated answer: {generated_answer}

Mark INCORRECT if any of the following apply:
- The answer states something false or contradicts the provided code.
- The answer invents behavior not present in the context.
- The answer is generic enough that it would be equally true of a different
  function or class than the one asked about - i.e., it doesn't demonstrate
  specific knowledge of the mechanism the question is actually asking about.
Test for the third case: could this exact answer be reused, unchanged, as a
response to a different but superficially similar question about this codebase,
and still sound fine? If yes, it's too vague - mark INCORRECT.

Otherwise mark CORRECT.

Respond with a single word: CORRECT or INCORRECT."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    verdict = response.choices[0].message.content.strip().upper()
    return "CORRECT" in verdict and "INCORRECT" not in verdict

with open("eval_set.json", "r") as f:
    eval_questions = json.load(f)

function_aware_chunks = load_chunks("chunks_function_aware.json")

answerable_questions = [q for q in eval_questions if "N/A" not in q["source"]]

results = []
for q in answerable_questions:
    start_time = time.time()
    generated = generate_answer(q["question"], function_aware_chunks)
    elapsed = time.time() - start_time

    retrieved = search(q["question"], chunks=function_aware_chunks, top_k=6)
    retrieved_context = "\n\n---\n\n".join(
        f"From {chunk['source_file']} ({chunk.get('name', 'unknown')}):\n{chunk['text']}"
        for score, chunk in retrieved
    )

    verdict_lenient = judge_answer_lenient(q["question"], q["answer"], generated)
    verdict_strict = judge_answer_strict(q["question"], q["answer"], generated)
    verdict_grounded = judge_answer_against_source(q["question"], retrieved_context, generated)

    results.append({
        "id": q["id"],
        "lenient": verdict_lenient,
        "strict": verdict_strict,
        "grounded": verdict_grounded,
        "latency_seconds": round(elapsed, 2)
    })

    print(f"Q{q['id']}: lenient={verdict_lenient} strict={verdict_strict} grounded={verdict_grounded} ({elapsed:.2f}s)")

lenient_correct = sum(1 for r in results if r["lenient"])
strict_correct = sum(1 for r in results if r["strict"])
grounded_correct = sum(1 for r in results if r["grounded"])
avg_latency = sum(r["latency_seconds"] for r in results) / len(results)

print(f"\n--- END-TO-END ANSWER QUALITY (THREE JUDGES) ---")
print(f"Lenient (vs reference): {lenient_correct}/{len(results)} = {lenient_correct/len(results):.2%}")
print(f"Strict (vs reference): {strict_correct}/{len(results)} = {strict_correct/len(results):.2%}")
print(f"Grounded (vs source code): {grounded_correct}/{len(results)} = {grounded_correct/len(results):.2%}")
print(f"Average latency per question: {avg_latency:.2f}s")

print("\n--- CASES WHERE STRICT AND GROUNDED DISAGREE ---")
for r in results:
    if r["strict"] != r["grounded"]:
        print(f"Q{r['id']}: strict={r['strict']}, grounded={r['grounded']}")
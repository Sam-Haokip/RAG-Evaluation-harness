import os
import time
import random
from dotenv import load_dotenv
from openai import OpenAI
from search import load_chunks, search
import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AICREDITS_API_KEY"),
    base_url="https://aicredits.in/v1"
)

def generate_no_context(question):
    prompt = f"""Answer the question about the requests Python library as best you can.

Question: {question}

Answer:"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def generate_wrong_context(question, wrong_context):
    prompt = f"""Answer the question using ONLY the code context below.
If the context does not contain enough information to answer confidently,
say clearly "I could not find this in the codebase" instead of guessing.

Context:
{wrong_context}

Question: {question}

Answer:"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

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

with open("eval_set.json", "r") as f:
    eval_questions = json.load(f)

function_aware_chunks = load_chunks("chunks_function_aware.json")
answerable_questions = [q for q in eval_questions if "N/A" not in q["source"]]

random.seed(42)

print(f"Running closed-book control on {len(answerable_questions)} questions\n")

no_context_results = []
wrong_context_results = []

for i, q in enumerate(answerable_questions):
    # No context at all
    ans_no_ctx = generate_no_context(q["question"])
    lenient_no = judge_answer_lenient(q["question"], q["answer"], ans_no_ctx)
    strict_no = judge_answer_strict(q["question"], q["answer"], ans_no_ctx)
    no_context_results.append({"lenient": lenient_no, "strict": strict_no})

    # Wrong context: retrieve chunks for a DIFFERENT question
    other_questions = [x for x in answerable_questions if x["id"] != q["id"]]
    decoy_question = other_questions[(i * 7) % len(other_questions)]["question"]
    decoy_results = search(decoy_question, chunks=function_aware_chunks, top_k=6)
    wrong_context = "\n\n---\n\n".join(
        f"From {c['source_file']} ({c.get('name', 'unknown')}):\n{c['text']}"
        for score, c in decoy_results
    )
    ans_wrong_ctx = generate_wrong_context(q["question"], wrong_context)
    lenient_wrong = judge_answer_lenient(q["question"], q["answer"], ans_wrong_ctx)
    strict_wrong = judge_answer_strict(q["question"], q["answer"], ans_wrong_ctx)
    wrong_context_results.append({"lenient": lenient_wrong, "strict": strict_wrong})

    print(f"Q{q['id']}: no-context(lenient={lenient_no}, strict={strict_no})  wrong-context(lenient={lenient_wrong}, strict={strict_wrong})")

n = len(answerable_questions)
print(f"\n--- CLOSED-BOOK CONTROL RESULTS ---")
print(f"No context   - lenient: {sum(r['lenient'] for r in no_context_results)}/{n} = {sum(r['lenient'] for r in no_context_results)/n:.2%}")
print(f"No context   - strict:  {sum(r['strict'] for r in no_context_results)}/{n} = {sum(r['strict'] for r in no_context_results)/n:.2%}")
print(f"Wrong context - lenient: {sum(r['lenient'] for r in wrong_context_results)}/{n} = {sum(r['lenient'] for r in wrong_context_results)/n:.2%}")
print(f"Wrong context - strict:  {sum(r['strict'] for r in wrong_context_results)}/{n} = {sum(r['strict'] for r in wrong_context_results)/n:.2%}")
print(f"\n(Compare against correct-context results from check_answer_quality.py: lenient 78%, strict 60%)")
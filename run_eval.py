import json
from search import search, load_chunks
from search_decomposed import search_with_decomposition
from generate_answer import generate_answer
from search_hybrid import search_hybrid, build_bm25_index
from cost_tracker import CostTracker

with open("eval_set.json", "r") as f:
    eval_questions = json.load(f)

print(f"Loaded {len(eval_questions)} eval questions.")
print("First question:", eval_questions[0]["question"])

function_aware_chunks = load_chunks("chunks_function_aware.json")

def check_retrieval(question_entry, chunks=None, top_k=3, tracker=None):
    query = question_entry["question"]
    results = search(query, chunks=chunks, top_k=top_k, tracker=tracker)
    retrieved_sources = [chunk["source_file"] for score, chunk in results]
    expected_source = question_entry["source"]

    if "N/A" in expected_source:
        found = None
    else:
        import re
        expected_files = re.findall(r'\b\w+\.py\b', expected_source)
        found = all(ef in retrieved_sources for ef in expected_files)

    return {
        "id": question_entry["id"],
        "type": question_entry["type"],
        "question": query,
        "retrieved_sources": retrieved_sources,
        "expected_source": expected_source,
        "found": found
    }

def check_retrieval_decomposed(question_entry, chunks, top_k_per_query=3, tracker=None):
    query = question_entry["question"]
    results, sub_queries = search_with_decomposition(query, chunks, top_k_per_query=top_k_per_query, tracker=tracker)
    retrieved_sources = [chunk["source_file"] for score, chunk in results]
    expected_source = question_entry["source"]

    if "N/A" in expected_source:
        found = None
    else:
        import re
        expected_files = re.findall(r'\b\w+\.py\b', expected_source)
        found = all(ef in retrieved_sources for ef in expected_files)

    return {
        "id": question_entry["id"],
        "question": query,
        "retrieved_sources": retrieved_sources,
        "expected_source": expected_source,
        "found": found
    }

def check_retrieval_hybrid(question_entry, chunks, bm25, top_k=6, tracker=None):
    query = question_entry["question"]
    results = search_hybrid(query, chunks, bm25, top_k=top_k, tracker=tracker)
    retrieved_sources = [chunk["source_file"] for score, chunk in results]
    expected_source = question_entry["source"]

    if "N/A" in expected_source:
        found = None
    else:
        import re
        expected_files = re.findall(r'\b\w+\.py\b', expected_source)
        found = all(ef in retrieved_sources for ef in expected_files)

    return {"id": question_entry["id"], "found": found,
            "question": query, "expected_source": expected_source,
            "retrieved_sources": retrieved_sources}

# ---------- BASELINE (function-aware chunking) ----------
baseline_tracker = CostTracker()
all_results = [check_retrieval(q, chunks=function_aware_chunks, top_k=6, tracker=baseline_tracker) for q in eval_questions]

answerable_results = [r for r in all_results if r["found"] is not None]
unanswerable_results = [r for r in all_results if r["found"] is None]

correct_count = sum(1 for r in answerable_results if r["found"])
total_answerable = len(answerable_results)
accuracy = correct_count / total_answerable if total_answerable > 0 else 0

print(f"\n--- FUNCTION-AWARE CHUNKING RESULTS ---")
print(f"Answerable questions: {total_answerable}")
print(f"Correctly retrieved: {correct_count}")
print(f"Retrieval accuracy: {accuracy:.2%}")
print(f"Cost: {baseline_tracker.totals()}")
print(f"\nUnanswerable questions (for later hallucination testing): {len(unanswerable_results)}")

print("\n--- FAILURES ---")
for r in answerable_results:
    if not r["found"]:
      print(f"Q{r['id']}: {r['question']}")
      print(f"  Expected: {r['expected_source']}")
      print(f"  Got: {r['retrieved_sources']}")

# ---------- QUERY DECOMPOSITION ----------
print("\n\n=== TESTING QUERY DECOMPOSITION (all 30 questions) ===")
decomposed_tracker = CostTracker()
decomposed_results = [check_retrieval_decomposed(q, function_aware_chunks, tracker=decomposed_tracker) for q in eval_questions]

decomposed_answerable = [r for r in decomposed_results if r["found"] is not None]
decomposed_correct = sum(1 for r in decomposed_answerable if r["found"])
decomposed_total = len(decomposed_answerable)
decomposed_accuracy = decomposed_correct / decomposed_total if decomposed_total > 0 else 0

print(f"Answerable questions: {decomposed_total}")
print(f"Correctly retrieved: {decomposed_correct}")
print(f"Retrieval accuracy: {decomposed_accuracy:.2%}")
print(f"Cost: {decomposed_tracker.totals()}")

print("\n--- REMAINING FAILURES ---")
for r in decomposed_answerable:
    if not r["found"]:
        print(f"Q{r['id']}: {r['question']}")
        print(f"  Expected: {r['expected_source']}")
        print(f"  Got: {r['retrieved_sources']}")

# ---------- HALLUCINATION ----------
print("\n\n=== TESTING HALLUCINATION BEHAVIOR (unanswerable questions) ===")
hallucination_tracker = CostTracker()
unanswerable_questions = [q for q in eval_questions if "N/A" in q["source"]]

hallucination_count = 0
for q in unanswerable_questions:
    answer = generate_answer(q["question"], function_aware_chunks, tracker=hallucination_tracker)
    admitted_unknown = "could not find" in answer.lower() or "does not exist" in answer.lower() or "no such method" in answer.lower()

    print(f"\nQ: {q['question']}")
    print(f"A: {answer}")
    print(f"Correctly admitted unknown: {admitted_unknown}")

    if not admitted_unknown:
        hallucination_count += 1

print(f"\n--- HALLUCINATION SUMMARY ---")
print(f"Total unanswerable questions: {len(unanswerable_questions)}")
print(f"Hallucinated (confidently wrong): {hallucination_count}")
print(f"Hallucination rate: {hallucination_count / len(unanswerable_questions):.2%}")
print(f"Cost: {hallucination_tracker.totals()}")

# ---------- HYBRID SEARCH ----------
print("\n\n=== TESTING HYBRID SEARCH (all 30 questions) ===")
bm25_index = build_bm25_index(function_aware_chunks)

hybrid_tracker = CostTracker()
hybrid_results = [check_retrieval_hybrid(q, function_aware_chunks, bm25_index, tracker=hybrid_tracker) for q in eval_questions]
hybrid_answerable = [r for r in hybrid_results if r["found"] is not None]
hybrid_correct = sum(1 for r in hybrid_answerable if r["found"])
hybrid_accuracy = hybrid_correct / len(hybrid_answerable)

print(f"Answerable questions: {len(hybrid_answerable)}")
print(f"Correctly retrieved: {hybrid_correct}")
print(f"Retrieval accuracy: {hybrid_accuracy:.2%}")
print(f"Cost: {hybrid_tracker.totals()}")

print("\n--- REMAINING FAILURES ---")
for r in hybrid_answerable:
    if not r["found"]:
        print(f"Q{r['id']}: {r['question']}")
        print(f"  Expected: {r['expected_source']}")
        print(f"  Got: {r['retrieved_sources']}")

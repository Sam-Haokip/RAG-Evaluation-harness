import json
from search import load_chunks
from generate_answer import generate_answer

with open("eval_set.json", "r") as f:
    eval_questions = json.load(f)

function_aware_chunks = load_chunks("chunks_function_aware.json")

# Look closely at a few specific "surprising" failures
ids_to_check = ["q5", "q6", "q9"]

for q in eval_questions:
    if q["id"] in ids_to_check:
        generated = generate_answer(q["question"], function_aware_chunks)
        print(f"=== {q['id']} ===")
        print(f"Question: {q['question']}")
        print(f"\nExpected answer:\n{q['answer']}")
        print(f"\nGenerated answer:\n{generated}")
        print("\n" + "="*50 + "\n")
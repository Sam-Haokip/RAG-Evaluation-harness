import os
import json
from math import comb
from dotenv import load_dotenv
from openai import OpenAI
from search import load_chunks
from generate_answer import generate_answer
from check_answer_quality import judge_answer_lenient, judge_answer_strict

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AICREDITS_API_KEY"),
    base_url="https://aicredits.in/v1"
)


def generate_no_context(question):
    """
    Closed-book control, matched to generate_answer's exact prompt template.

    This MUST stay structurally identical to generate_answer()'s prompt, with
    only the context block emptied out. An earlier version of this function
    used a different, weaker prompt ("answer as best you can"), which made the
    no-context condition free to guess while the real system was instructed to
    abstain when unsure. That mismatch is not a fair test of whether retrieval
    helps; it's a test of whether permission to guess beats an instruction to
    hold back, which produced a misleading result (no-context appeared to
    perform close to correct-context). See docs/rag-evaluation-project.docx,
    "Does retrieval even matter here?" for the full account.
    """
    prompt = f"""Answer the question using ONLY the code context below.
If the context does not contain enough information to answer confidently,
say clearly "I could not find this in the codebase" instead of guessing.

Context:
(no context provided)

Question: {question}

Answer:"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(2 * p, 1.0)


if __name__ == "__main__":
    with open("eval_set.json") as f:
        eval_questions = json.load(f)

    function_aware_chunks = load_chunks("chunks_function_aware.json")
    answerable = [q for q in eval_questions if "N/A" not in q["source"]]

    print(f"Running matched-prompt closed-book control on {len(answerable)} questions\n")

    b_lenient = c_lenient = 0
    b_strict = c_strict = 0
    correct_ctx_lenient_count = 0
    no_ctx_lenient_count = 0
    correct_ctx_strict_count = 0
    no_ctx_strict_count = 0

    for q in answerable:
        correct_ctx_answer = generate_answer(q["question"], function_aware_chunks)
        no_ctx_answer = generate_no_context(q["question"])

        cc_lenient = judge_answer_lenient(q["question"], q["answer"], correct_ctx_answer)
        nc_lenient = judge_answer_lenient(q["question"], q["answer"], no_ctx_answer)
        cc_strict = judge_answer_strict(q["question"], q["answer"], correct_ctx_answer)
        nc_strict = judge_answer_strict(q["question"], q["answer"], no_ctx_answer)

        correct_ctx_lenient_count += cc_lenient
        no_ctx_lenient_count += nc_lenient
        correct_ctx_strict_count += cc_strict
        no_ctx_strict_count += nc_strict

        if cc_lenient and not nc_lenient:
            b_lenient += 1
        elif nc_lenient and not cc_lenient:
            c_lenient += 1

        if cc_strict and not nc_strict:
            b_strict += 1
        elif nc_strict and not cc_strict:
            c_strict += 1

        print(f"{q['id']}: correct-ctx(lenient={cc_lenient},strict={cc_strict}) "
              f"no-ctx(lenient={nc_lenient},strict={nc_strict})")

    n = len(answerable)
    p_lenient = mcnemar_exact(b_lenient, c_lenient)
    p_strict = mcnemar_exact(b_strict, c_strict)

    print(f"\n--- CLOSED-BOOK CONTROL RESULTS (matched prompts) ---")
    print(f"Correct-context: lenient {correct_ctx_lenient_count}/{n} = {correct_ctx_lenient_count/n:.2%}, "
          f"strict {correct_ctx_strict_count}/{n} = {correct_ctx_strict_count/n:.2%}")
    print(f"No-context:      lenient {no_ctx_lenient_count}/{n} = {no_ctx_lenient_count/n:.2%}, "
          f"strict {no_ctx_strict_count}/{n} = {no_ctx_strict_count/n:.2%}")
    print(f"\nMcNemar's exact test (correct-context vs. no-context):")
    print(f"Lenient: correct-ctx-only-right={b_lenient}, no-ctx-only-right={c_lenient}, p={p_lenient}")
    print(f"Strict:  correct-ctx-only-right={b_strict}, no-ctx-only-right={c_strict}, p={p_strict}")
    print(f"\nExpected result (from the run reported in README.md and docs/rag-evaluation-project.docx):")
    print(f"Lenient: 20 vs 0, p < 0.0001")
    print(f"Strict:  14 vs 0, p < 0.0001")

# RAG Evaluation Harness for Codebase Q&A

Answers natural-language questions about a Python codebase (`psf/requests`) by retrieving relevant source code and generating a grounded answer from it. Built primarily as a testbed for measuring retrieval quality rigorously: most of the effort here went into the evaluation harness, not the chatbot on top of it.

**If the retrieved code doesn't contain the answer, the system says so.** Measured hallucination rate on adversarial unanswerable questions: 0%.

## Results

Eight retrieval configurations, benchmarked against the same hand-verified 36-question set (factual, multi-hop, and deliberately unanswerable questions).

| Configuration | Retrieval Accuracy | Notes |
|---|---|---|
| Fixed-line chunking (baseline) | 75.00% | Original 30-question set |
| AST-based chunking, first attempt | 66.67% | Regression, see [What I got wrong](#what-i-got-wrong) |
| AST-based chunking, fixed | 83.33% | |
| + top-k raised 3 → 6 | 90.00% | |
| Hybrid (BM25 + semantic), α=0.5 | 83.33% | Worse than semantic alone |
| Hybrid, α=0.7 | 90.00% | Believed to be the ceiling at the time |
| **Hybrid, α=0.8 (full sweep)** | **93.33%** | Actual peak, earlier conclusion was wrong |
| Query decomposition | 83.33–95.83% (varies by run) | Non-deterministic, LLM-generated sub-queries |

**Cost per question:** baseline retrieval costs roughly ₹0.00002. Query decomposition costs more than 100× that, sometimes considerably more depending on how many sub-queries a given question triggers, for comparable or sometimes zero accuracy gain. Hybrid search adds a full second retrieval signal (BM25) at **zero marginal cost**, since it's computed locally against the already-loaded corpus.

**Answer quality is not one number.** Grading the same generated answers with three independently-designed judges (lenient reference-matching, strict reference-matching, source-grounded factual checking) gave accuracies of 78%, 60%, and 83% respectively, a stable 18–24 point gap, confirmed across three repeated runs, that doesn't close with more sampling. See [below](#how-correct-is-correct) for why.

## The finding I'd actually talk about in an interview

While building an LLM-as-judge to grade generated answers, the exact same evaluation pipeline scored **37.5% or 75%** depending entirely on how strictly the grading prompt was worded, with no other change. Digging into why led to a deeper investigation: reading the disputed cases showed two distinct failure patterns hiding inside "the judge disagreed" (genuinely shallow answers, and answers that were *more* detailed than the reference and got penalized for it). That motivated a third, source-grounded judge that checks the answer against the retrieved code directly instead of a single fixed reference phrasing.

That judge's first version was also broken: it marked every answer correct, meaning it had over-corrected into never penalizing anything. Adding an explicit genericness check ("could this exact answer be reused for a different question and still sound fine?") fixed it. Full writeup in [`docs/rag-evaluation-project.docx`](docs/rag-evaluation-project.docx).

## What I got wrong

Early in the project, hybrid keyword+semantic search was tested at two weightings (α=0.5, α=0.7) and reported as **not** beating pure semantic search, a reasonable conclusion from the data available at the time. A full sweep across eight values later found a real peak at α=0.8, beating semantic-only by 3.3 points. The original two points happened to sit on the rising part of a curve, short of its actual peak.

I'm leaving the wrong conclusion in the full writeup rather than quietly correcting it. The correction is itself evidence the methodology can catch its own mistakes when pushed further, which is the whole point of building an evaluation harness in the first place.

## How correct is "correct"?

Retrieval accuracy only measures whether the right file was found, not whether the final generated answer was actually right. Grading that required building an LLM judge, and the project ended up being as much an investigation into how to evaluate an LLM's answers as into retrieval itself. See the [full writeup](docs/rag-evaluation-project.docx) for the complete judge investigation.

## Architecture

| File | Responsibility |
|---|---|
| `build_index.py` | Chunking (fixed-line and AST-based), embedding generation, index persistence |
| `search.py` | Cosine-similarity retrieval; chunk set and cost tracker are injectable |
| `search_hybrid.py` | BM25 + semantic fusion with tunable weighting |
| `search_decomposed.py` | LLM query decomposition, with pass-through detection for questions that don't have decomposable structure |
| `alpha_sweep.py` | Sweeps hybrid search's weighting parameter with embeddings computed once, reused across all values |
| `generate_answer.py` | Context assembly and answer generation with explicit abstention instruction |
| `cost_tracker.py` | Per-call token and cost tracking across every strategy |
| `check_answer_quality.py` | Three-judge end-to-end answer grading |
| `judge_variance_check.py` | Repeats judge evaluation across multiple runs to separate real disagreement from sampling noise |
| `run_eval.py` | Runs every strategy against the eval set, reports accuracy, failures, and cost |
| `eval_set.json` | 36 hand-written, individually source-verified questions |

## Running it

```bash
pip install -r requirements.txt
# add your API key to .env (see .env.example)

python build_index.py        # builds both chunking indexes (one-time)
python run_eval.py           # runs all strategies, prints accuracy/cost/failures
python alpha_sweep.py        # sweeps hybrid search's alpha parameter
python check_answer_quality.py   # three-judge end-to-end answer grading
```

## Known limitations

- 36 questions is still a modest sample; strategy differences are directional, not statistically airtight
- Abstention detection uses keyword matching, untested against a larger or adversarial question set
- Function-level retrieval precision is unmeasured: scoring checks file-level presence only
- All findings are specific to this one 20-file corpus and may not generalize to a codebase with a different size/structure profile

## Stack

Python · `ast` for structural chunking · `text-embedding-3-small` · `gpt-4o-mini` for generation, decomposition, and judging · `numpy` · `rank-bm25` · corpus: [`psf/requests`](https://github.com/psf/requests)

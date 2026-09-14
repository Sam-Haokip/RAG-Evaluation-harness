# RAG Evaluation Harness for Codebase Q&A

Answers natural-language questions about a Python codebase (`psf/requests`) by retrieving relevant source code and generating a grounded answer from it. Built primarily as a testbed for measuring retrieval quality rigorously: most of the effort here went into the evaluation harness, not the chatbot on top of it.

**If the retrieved code doesn't contain the answer, the system says so.** Measured hallucination rate on adversarial unanswerable questions: 0%.

**None of the comparative claims below are statistically significant at the current sample size.** Every retrieval configuration was checked with McNemar's exact test against the strategy it was meant to beat. Chunking strategy (fixed-line vs. function-aware) and hybrid weighting (α=0.8 vs. pure semantic) both came back p=1.0: the entire "improvement" in each case is a single question out of thirty. Query decomposition came closer (p=0.25, 3 wins to 0 losses in its favor) but still falls short of the conventional 0.05 threshold. This is the real headline of the project as it currently stands: retrieval accuracy across every tested configuration clusters in the 83–100% range, and distinguishing between them properly requires more than 36 questions. See [What I got wrong](#what-i-got-wrong) for how this was found.

## Does retrieval even matter here?

`psf/requests` is almost certainly in the training data of the model used for generation, which raises an obvious question no accuracy number above answers on its own: is the system succeeding because retrieval works, or because the model already knows this library? Two controls were built to check.

**Control 1: no context, model free to answer from memory.** Given the question with no retrieved code and no instruction to hold back, the model answered 76.67–80.00% correctly across two runs (lenient judge), 63.33–66.67% (strict judge). That's close to, and in one run higher than, the system's actual end-to-end accuracy with real retrieval. Read on its own, this looked like the worst-case outcome: retrieval adding little or nothing.

**That reading was wrong, and it stayed wrong until the test was rebuilt.** The no-context prompt used in Control 1 said "answer as best you can," while the real system's prompt says "answer using ONLY the context below, and say you couldn't find it if the context is insufficient." Comparing those two isn't a test of whether retrieval helps; it's a test of whether being told to guess freely beats being told to hold back, which is a different and less interesting question. That confound was caught, and the same instruction was applied to both conditions, with the no-context version simply given an empty context block instead of real chunks.

**Control 2, matched prompts, is the real result: correct-context beat no-context on every single discordant question.** 20 wins to 0 on the lenient judge, 14 to 0 on strict, both with p-values below 0.0001. With the confound removed, the no-context condition scored essentially 0% correct on both judges, because it correctly followed the same instruction to abstain when told there was nothing to work with.

**Two separate, true things follow from this, and neither should be collapsed into the other:**
- The deployed system genuinely depends on real retrieved content. It does not silently fall back on memorized knowledge when retrieval comes up empty; it abstains, exactly as designed.
- The benchmark itself is not contamination-proof. A model with zero retrieval and permission to guess freely gets roughly 77–80% of these questions right purely from training familiarity with `requests`. Any accuracy number in this document should be read against that floor, not against zero.

## Results

Eight retrieval configurations, benchmarked against the same hand-verified 36-question set (factual, multi-hop, and deliberately unanswerable questions). Statistical status is noted per row where a test has been run.

| Configuration | Retrieval Accuracy | Statistical status |
|---|---|---|
| Fixed-line chunking (baseline) | 93.33% | Not significant vs. function-aware (p=1.0, McNemar exact) |
| AST-based chunking, first attempt | 66.67% | Regression, see [What I got wrong](#what-i-got-wrong) |
| AST-based chunking, fixed | 83.33% | 30-question set at the time |
| Function-aware chunking + top-k raised 3 → 6 | 90.00% | Not significant vs. fixed-line baseline (p=1.0) |
| Hybrid (BM25 + semantic), α=0.5 | 83.33% | Worse than semantic alone |
| Hybrid, α=0.7 | 90.00% | Believed to be the ceiling at the time |
| Hybrid, α=0.8 (full sweep) | 93.33% | Not significant vs. α=1.0 pure semantic (p=1.0) |
| Query decomposition | 83.33–100% (varies by run) | Closest to significant of the three tested (p=0.25, this run); directionally favors decomposition (3-0) but not confirmed |

**Cost per question:** baseline retrieval costs roughly ₹0.00002. Query decomposition costs more than 100× that, sometimes considerably more depending on how many sub-queries a given question triggers, for an accuracy advantage that is directionally promising but not yet statistically confirmed. Hybrid search adds a full second retrieval signal (BM25) at **zero marginal cost**, though its accuracy advantage over semantic-only search is also not yet confirmed at this sample size.

**Answer quality is not one number.** Grading the same generated answers with three independently-designed judges (lenient reference-matching, strict reference-matching, source-grounded factual checking) gave accuracies of 78%, 60%, and 83% respectively, a stable 18–24 point gap, confirmed across three repeated runs, that doesn't close with more sampling. See [below](#how-correct-is-correct) for why.

## The finding I'd actually talk about in an interview

While building an LLM-as-judge to grade generated answers, the exact same evaluation pipeline scored **37.5% or 75%** depending entirely on how strictly the grading prompt was worded, with no other change. Digging into why led to a deeper investigation: reading the disputed cases showed two distinct failure patterns hiding inside "the judge disagreed" (genuinely shallow answers, and answers that were *more* detailed than the reference and got penalized for it). That motivated a third, source-grounded judge that checks the answer against the retrieved code directly instead of a single fixed reference phrasing.

That judge's first version was also broken: it marked every answer correct, meaning it had over-corrected into never penalizing anything. Adding an explicit genericness check ("could this exact answer be reused for a different question and still sound fine?") fixed it. Full writeup in [`docs/rag-evaluation-project.docx`](docs/rag-evaluation-project.docx).

## What I got wrong

Early in the project, hybrid keyword+semantic search was tested at two weightings (α=0.5, α=0.7) and reported as **not** beating pure semantic search, a reasonable conclusion from the data available at the time. A full sweep across eight values later found what looked like a real peak at α=0.8, beating semantic-only by 3.3 points, and I reported that as a corrected, confirmed finding.

It wasn't. Running McNemar's exact test on that exact comparison, prompted by a separate re-verification pass that also caught the sample-size issue below, returned p=1.0: the entire "peak" is one question out of thirty. I had replaced one unearned conclusion (hybrid doesn't help) with another (hybrid is confirmed better) without ever actually testing whether either one was statistically real. The honest version is neither: the data so far cannot distinguish hybrid search from pure semantic search at any weighting tested.

A second mistake surfaced in the same pass, this time in the README rather than the analysis: after expanding the eval set from 30 to 36 questions, the strategy table kept reporting the original fixed-line baseline's 75.00% score from the old 30-question set, next to newer strategies scored on 36. Re-running the baseline on the current set gave 93.33%, higher than the "improved" chunking strategy it was supposed to be worse than. Testing that comparison also came back not significant (p=1.0): a full third of the project's headline comparisons turned out to be single-question noise rather than confirmed effects.

I'm leaving both mistakes here rather than quietly fixing the numbers, for the same reason as always: the corrections are the actual evidence that the methodology works, catching its own errors when someone (in this case, an outside reviewer) pushes on it hard enough. The real lesson isn't "hybrid doesn't help" or "chunking doesn't matter." It's that 36 questions is not enough to responsibly claim any of these comparisons are settled, and every accuracy table in this project should be read with that in mind until the eval set grows.

A third mistake happened while building the closed-book control described above, and it's worth including because it's a different kind of error than the first two: not a missing test, but a misread of a test's own result. The first version of the no-context control used a different, weaker prompt than the real system (permission to guess freely, versus an instruction to abstain when unsure), and reading its output at face value looked like the worst possible outcome: retrieval barely beating a model just guessing from memory. That comparison wasn't testing what it appeared to test. Once the prompts were matched so the only real difference was presence or absence of retrieved code, the result reversed completely, in favor of retrieval by an overwhelming margin. The lesson here is distinct from the first two: a statistically valid test on a badly designed comparison can be just as misleading as no test at all, and the p-value from a flawed experiment doesn't rescue the experiment.

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

- **None of the three tested comparisons reach statistical significance at n=36.** Chunking strategy: p=1.0. Hybrid weighting: p=1.0. Query decomposition: p=0.25, the closest but still short of 0.05. A larger, properly powered eval set (80-100+ questions, per a standard rule of thumb for detecting effects of this size) is the single highest-priority next step for this project, above any new feature.
- **Resolved:** whether the system depends on real retrieval or is coasting on the generation model's training knowledge of `psf/requests` was an open question, addressed with a matched-prompt closed-book control (see [Does retrieval even matter here?](#does-retrieval-even-matter-here)). The system does depend on retrieval (p < 0.0001). The benchmark itself is not contamination-proof: a no-context model given permission to guess freely scores roughly 77-80% on these questions from training familiarity alone, and that floor should be kept in mind when reading any accuracy number in this document.
- Abstention detection uses keyword matching, untested against a larger or adversarial question set
- Function-level retrieval precision is unmeasured: scoring checks file-level presence only
- All findings are specific to this one 20-file corpus and may not generalize to a codebase with a different size/structure profile

## Stack

Python · `ast` for structural chunking · `text-embedding-3-small` · `gpt-4o-mini` for generation, decomposition, and judging · `numpy` · `rank-bm25` · corpus: [`psf/requests`](https://github.com/psf/requests)

# Evaluating Retrieval Strategies for Codebase Question Answering

*A RAG system built around a measurement harness, not the other way round*

Most retrieval-augmented generation projects stop at "it answers questions." This one starts there and asks the harder question: how would you know if it were any good? The system answers questions about a real production codebase, the Python requests library, and every design decision in it was made against a hand-built evaluation set rather than by inspection of a few cherry-picked outputs.

Eight retrieval and generation configurations were built and measured against the same evaluation set, now 36 questions. Several of them made things worse before they made things better. Those are documented here alongside the ones that worked, because the failures turned out to be the more instructive half of the project. In one case, a conclusion reached earlier in this project was later found to be wrong and is corrected below rather than quietly dropped.

## Headline results


**None of the comparative claims in this table are statistically significant at the current sample size. Every pairwise comparison below was checked with McNemar's exact test; two came back p=1.0 (the entire reported difference is a single question out of thirty) and the third, query decomposition, came closer at p=0.25 but still falls short of the conventional 0.05 threshold. This is stated here, at the top, rather than in the limitations section at the bottom, because it materially changes how every number below should be read. Full detail in “A statistical check that changed the headline” below.**

| **Metric**                                                        | **Result**                   | **Comparison point**                             |
|-------------------------------------------------------------------|------------------------------|--------------------------------------------------|
| Retrieval accuracy, function-aware chunking (36 questions)        | **90.00%**                   | 93.33% fixed-line, not significant (p=1.0)       |
| Retrieval accuracy, hybrid search at best alpha (0.8)             | **93.33%**                   | 90.00% semantic-only, not significant (p=1.0)    |
| Retrieval accuracy, query decomposition (this run)                | **100.00%**                  | 90.00% baseline, closest to significant (p=0.25) |
| End-to-end answer accuracy (single-run, earlier eval size)        | **75.00%**                   | n/a                                              |
| Answer-quality range across three independent judges, repeated 3x | **58–87%**                   | n/a                                              |
| Hallucination rate on unanswerable questions                      | **0.00%**                    | n/a                                              |
| Mean latency per question, end to end                             | **2.37s**                    | n/a                                              |
| Cost per question, baseline vs. decomposition                     | **₹0.00002 vs ₹0.002–0.006** | n/a                                              |

## What the system does


Given a natural-language question about the requests library, for example "What does raise_for_status do when a request returns a 404?", the system retrieves the relevant source code from a 20-file corpus and generates a cited answer from it. If the retrieved code does not contain the answer, it says so rather than guessing.

### Pipeline


- **Ingestion.** 20 .py source files from psf/requests are parsed and split into chunks.

- **Indexing.** Each chunk is embedded with text-embedding-3-small (1536 dimensions) and persisted to disk, so the embedding pass runs once per chunking strategy rather than once per query.

- **Retrieval.** The question is embedded and scored against every chunk by cosine similarity, optionally blended with BM25 keyword scores; the top-k chunks are returned.

- **Generation.** Retrieved chunks are assembled into a labelled context block and passed to gpt-4o-mini with an explicit instruction to abstain when the context is insufficient.

### Evaluation set


Thirty-six questions, written by hand and individually verified against the source before being accepted, in three categories:

- **Factual**: answerable from a single function or module.

- **Multi-hop**: require two or three files at once. "How does Session.send() delegate the actual network request?" needs both sessions.py and adapters.py; retrieving either alone is scored as a miss.

- **Unanswerable**: plausible-sounding questions with no answer in the corpus (Session.retry_with_backoff(), WebSocket support, GraphQL batching). These exist solely to measure hallucination.

The set was expanded from an initial 30 to 36 by targeting files with zero or thin coverage: auth.py had none, and gained five verified questions covering its digest-authentication logic and thread-local state handling; structures.py and status_codes.py each gained a question closing gaps in their coverage. Every added question follows the same verification discipline as the original set: the specific method or line referenced was located and read before the answer was written, not generated from general knowledge of the library and assumed correct.

## Strategy comparison


Every configuration below was run against the identical evaluation set. The scoring rule is strict: a multi-hop question counts as correct only if all required source files appear in the retrieved set.

| **Configuration**                                      | **Accuracy**   | **Statistical status**                                         |
|--------------------------------------------------------|----------------|----------------------------------------------------------------|
| Fixed 20-line chunks (baseline, re-run on current set) | 93.33%         | Not significant vs. function-aware, p=1.0                      |
| Function-aware chunking, first attempt                 | 66.67%         | Regression (see Finding 1)                                     |
| Function-aware chunking, module content restored       | 83.33%         | 30-question set at the time                                    |
| \+ top-k raised 3 → 6 (current 36-question set)        | 90.00%         | Not significant vs. fixed-line baseline, p=1.0                 |
| Hybrid, α=0.5 (first attempt)                          | 83.33%         | Worse than semantic alone                                      |
| Hybrid, α=0.7 (second attempt)                         | 90.00%         | Parity with semantic alone                                     |
| Hybrid, α=0.8 (full sweep)                             | 93.33%         | Not significant vs. α=1.0 pure semantic, p=1.0 (see Finding 5) |
| Query decomposition (range across runs)                | **83.33–100%** | Closest to significant, p=0.25 (see Finding 3)                 |

## A statistical check that changed the headline


This section exists because a routine consistency check turned into something bigger. While fixing an unrelated footnote (the fixed-line chunking baseline had never been re-run after the evaluation set grew from 30 to 36 questions), the re-run scored 93.33%, higher than the function-aware chunking strategy it was supposed to have been superseded by. That result forced a real question: is this a genuine reversal, or is a comparison this close simply not decidable at this sample size?

McNemar's exact test answers exactly that question for paired binary outcomes on the same question set. It was applied to all three comparisons in this project that claim one configuration beats another:

| **Comparison**                                         | **Discordant pairs** | **p-value** |
|--------------------------------------------------------|----------------------|-------------|
| Fixed-line chunking vs. function-aware chunking        | 1 vs. 0              | **1.0**     |
| Hybrid search (α=0.8) vs. pure semantic search (α=1.0) | 1 vs. 0              | **1.0**     |
| Query decomposition vs. function-aware baseline        | 3 vs. 0              | **0.25**    |

The first two results are unambiguous: a p-value of 1.0 means the observed difference is exactly as likely under the assumption of no real effect as it is under any alternative. The entire reported improvement, in both cases, is a single discordant question out of thirty. Neither the chunking-strategy conclusion nor the alpha-sweep conclusion, as previously stated in this document, is statistically supported.

The third result is more interesting and worth reading carefully rather than lumping in with the other two. Decomposition won 3 of the discordant pairs and lost 0, a pattern that is directionally clean even though it does not clear the conventional 0.05 threshold with only 3 discordant observations to work with. This is the shape of a result that a larger sample would likely resolve one way or the other, rather than a result that looks like pure noise. It is reported as “not yet significant” rather than “no effect,” because those are different claims and this project has already shown what happens when that distinction gets collapsed.

**The practical conclusion is not that any of these techniques are useless. It is that a 36-question evaluation set cannot responsibly distinguish between configurations that differ by one to three questions, and every comparative claim in this document prior to this section was written without that check having been run. It has now been run, and the results are reported here in full rather than quietly folded into the tables above, for the same reason every other correction in this project has been kept visible: the willingness to report a finding that weakens the project's own headline is the actual evidence the methodology is sound.**

## Does retrieval even matter here?


psf/requests is almost certainly present in the training data of the model used for generation, which raises a question none of the accuracy numbers above answer on their own: is the system succeeding because retrieval works, or because the model already knows this library? Two controls were built to check, and the process of building them correctly turned out to matter as much as the result.

### The first version of the test was itself flawed


Given the question with no retrieved context and a prompt that simply said “answer as best you can,” the model answered correctly on 76.67–80.00% of questions across two runs on the lenient judge, and 63.33–66.67% on the strict judge. Read at face value, this looked like close to the worst possible outcome: a no-context model matching or beating the real system's own end-to-end accuracy, suggesting retrieval was adding little.

**That reading was wrong, and the error was in the experiment's design, not the arithmetic. The no-context prompt told the model to answer freely; the real system's prompt explicitly instructs it to use only the provided context and to say it could not find the answer otherwise. Comparing those two conditions is not a clean test of whether retrieval helps. It is a test of whether permission to guess beats an instruction to hold back, a different and considerably less interesting question. A statistically well-formed test run on a badly designed comparison produces a confident, wrong answer, and this was very nearly reported as one.**

### With the confound removed, the result reversed completely


The no-context condition was rebuilt using the identical prompt template as the real system, with an empty context block in place of retrieved chunks, so the only variable that changed was the presence or absence of real retrieved code. Paired against the system's actual correct-context answers and tested with McNemar's exact test:

| **Judge**                                          | **Discordant pairs** | **p-value**   |
|----------------------------------------------------|----------------------|---------------|
| Lenient (correct-context wins vs. no-context wins) | 20 vs. 0             | **\< 0.0001** |
| Strict (correct-context wins vs. no-context wins)  | 14 vs. 0             | **\< 0.0001** |

With matched prompts, the no-context condition scored essentially 0% correct on both judges: told there was nothing to work with, it correctly abstained rather than answering from memory, exactly as the grounding instruction intends. Correct-context won every single discordant question. This is the strongest, cleanest statistical result in the entire project, in the opposite direction from what the flawed first version of the test suggested.

### Two separate conclusions, not one


- **The deployed system genuinely depends on retrieval.** It does not silently fall back on the generation model's memorized knowledge when retrieval fails to find anything; it abstains, which is the behavior it was designed to have and the behavior actually confirmed here.

- **The benchmark itself is not contamination-proof.** A model given zero context and permission to answer freely still gets roughly 77–80% of these questions right, purely from training familiarity with psf/requests. Every accuracy number in this document should be read against that floor rather than against zero, and a corpus that is not present in a model's training data would be needed for a contamination-free version of this evaluation.

### 1. A "smarter" chunker silently deleted a third of the corpus's meaning


Fixed-line chunking cuts functions in half. Replacing it with AST-based chunking, using Python's ast module to emit one chunk per function and class, should have been a clear improvement. It dropped accuracy to 66.67%.

The newly-failing questions were all module-level: "What does certs.py do?", "What does packages.py do?" The chunker walked the syntax tree for FunctionDef and ClassDef nodes and nothing else, so module docstrings, imports and top-level constants were never indexed at all. For small files that are mostly docstring, the entire file effectively vanished from the index.

Fix: track which line numbers are claimed by a function or class, then emit everything left over as a module-level chunk. Accuracy went to 83.33%, above the original baseline.

### 2. Retrieving more results papered over the multi-hop problem without solving it


Raising top-k from 3 to 6 lifted accuracy substantially. But inspecting the surviving failures showed this was partial luck rather than a full fix: for a question needing two specific files together, one would sometimes appear multiple times in the results while the other never appeared at all.

Single-vector similarity search scores each chunk in isolation. Two files that are causally linked (one calls the other) are not necessarily semantically similar to the same query, so no amount of extra slots reliably surfaces both. The problem needed a different technique, not a bigger k.

### 3. Query decomposition helps, is non-deterministic, and initially traded one failure for another


Asking the LLM to split each question into focused sub-queries, searching each independently and merging the deduplicated results, reached as high as 95.83% on some runs, the strongest single result measured for retrieval. It also does not return the same number twice: across repeated runs on the identical question set it has scored anywhere from 83.33% to 95.83%, because the decomposition step is itself an LLM generation and varies run to run.

Inspecting the actual sub-queries generated for the hardest three-file question showed why early attempts underperformed: the model produced near-synonyms of the original question (“requests.get internals”, “requests library source code”, “Python requests get method flow”) rather than splitting by component. Rewriting the prompt to explicitly request one sub-query per distinct function, method, or file fixed that question for the first time, but broke a different one: a question asking what a small metadata file contains started failing, because the model tried to invent decomposable components for a question that has none, producing generic packaging-concept sub-queries instead of just searching for the file directly.

The final fix adds an explicit branch to the prompt: if the question asks about a single flat piece of information rather than a multi-step mechanism, output the question unchanged as the only sub-query. Tested against both the three-file question and the single-file question that had regressed, both now pass, and the fix reduced total API calls (and therefore cost) rather than increasing them, since single-line pass-through skips the extra search calls a forced three-way split would have made.

**Reporting a single decomposition accuracy number would be misleading; the honest claim is a mean somewhere in the 83–100% range observed across runs, with the understanding that the technique's reliability depends on whether a given question actually has decomposable structure. Tested against the function-aware baseline with McNemar's exact test, decomposition came the closest of any strategy in this project to a statistically significant improvement (p=0.25, 3 wins to 0 losses) without actually clearing the threshold, making it the one result here worth prioritizing if the evaluation set is expanded.**

### 4. Hybrid keyword search's first two data points understated it, and the fuller sweep overstated the fix


BM25 keyword scoring combined with semantic search was tried at two weightings early in the project: α=0.5 scored 83.33% (worse than semantic alone), and α=0.7 scored 90.00% (parity, no gain). Based on those two points, hybrid search was reported as not helping.

A full sweep across eight alpha values later appeared to correct this. Accuracy rises steadily from α=0.3 through α=0.8, peaks at **93.33% at α=0.8**, 3.33 points above pure semantic search, then falls back to the semantic-only baseline at α=0.9 and 1.0. Re-running α=0.8 twice more against the same precomputed embeddings returned the identical count both times, which was reported at the time as confirmation the result was stable.

**That repeat-run check proved the wrong thing. Cosine similarity and BM25 scoring are deterministic arithmetic with no generation step involved; running the identical calculation twice will always return the identical answer. It demonstrates the code is reproducible, not that the underlying effect is real. The actual test is whether α=0.8 and α=1.0 disagree on more than a handful of the 30 questions, and applying McNemar's exact test to that comparison gives p=1.0: the entire 3.33-point gap is one question out of thirty. It is not statistically distinguishable from chance at this sample size.**

**This means the project made the same mistake twice, in opposite directions, on the same underlying question. The original two-point conclusion (hybrid doesn't help) was reported without enough data to support it. The revised eight-point conclusion (hybrid does help, confirmed) was also reported without the one test that would have actually supported it. The honest version, arrived at only after a third pass, is that pure semantic search and hybrid search at any tested weighting are not distinguishable with the data collected so far.**

A separate methodological gap sat underneath the first two: alpha=0.8 was selected by sweeping all eight values against the same 30 questions used to report its accuracy, tuning and evaluating on identical data. This was addressed with a stratified 15/15 tuning and holdout split, selecting alpha on the tuning half only and testing it on the untouched holdout half.

| **Configuration**                  | **Tuning set (n=15)** | **Holdout set (n=15)** |
|------------------------------------|-----------------------|------------------------|
| Hybrid, α=0.8 (selected on tuning) | **93.33%**            | **93.33%**             |
| Pure semantic, α=1.0 (never tuned) | 86.67%                | 93.33%                 |

**This result is genuinely mixed and both halves matter. The alpha-selection process itself is validated: 0.8's accuracy held exactly between tuning and holdout, a zero-point gap, meaning the earlier alpha sweep was not simply overfitting to the 30 questions it was evaluated on. But pure semantic search, never tuned at all, scored identically to the selected hybrid weighting on the holdout set. At 15 questions per half, this split cannot distinguish a real hybrid advantage from no advantage at all, which is consistent with, not contradictory to, the McNemar result above. The holdout split validates the methodology used to pick alpha. It does not validate the claim that hybrid search outperforms pure semantic search.**

### 5. Evaluation methodology moved the headline number by 37 points


Retrieval accuracy only measures whether the right code was found. To measure whether the final answer was correct, generated answers were graded against reference answers by a second LLM call. The first judge prompt returned 37.50% accuracy on a run that closer reading showed was not actually broken. Recalibrating the same prompt to tolerate differently-worded but factually correct answers moved the identical pipeline to 75.00%.

**Judge calibration is a design parameter, not a neutral instrument. A miscalibrated one made a working system look like it failed two questions in three. This was investigated in far more depth in a later pass; see the section below.**

### 6. Retrieval accuracy overstates system accuracy


Retrieval accuracy and end-to-end answer accuracy are not the same number, and the gap between them is generation loss: cases where the correct code was in context and the final answer still did not land. Retrieval quality is necessary but not sufficient, and reporting only the retrieval number would flatter the system.

## Cost: the third axis


Latency and accuracy say nothing about what a strategy costs to run in production. Every embedding and generation call was instrumented to report tokens and cost per call.

| **Strategy**                       | **Accuracy** | **API calls** | **Cost (₹)** |
|------------------------------------|--------------|---------------|--------------|
| Baseline (function-aware, top-k=6) | 90.00%       | 36            | 0.0009       |
| Hybrid (BM25 + semantic, α=0.8)    | **93.33%**   | 36            | 0.0009       |
| Query decomposition                | 93.33%       | 104–144       | 0.21–0.23    |

### Hybrid search's extra signal is free


Baseline and hybrid report identical cost, because BM25 keyword scoring is computed locally against the already-loaded corpus with no API call involved. Hybrid's only paid call is the same single embedding baseline makes. This means hybrid search's keyword signal is free to compute regardless of whether it turns out to help accuracy, which per Finding 4 above has not yet been statistically confirmed at this sample size.

### Decomposition's cost premium is large and its return is inconsistent


Decomposition makes multiple calls per question against baseline's roughly one (a sub-query generation call plus one embedding call per sub-query), putting its cost over 100 times higher than baseline. That premium sometimes buys no accuracy improvement over hybrid search, which costs essentially nothing. The honest characterization is that decomposition's expected value is positive but highly variable, and any single run may deliver little improvement at meaningfully higher cost.

## How much an answer is “correct” depends on who is asked


The judge-calibration finding above, the same pipeline scoring 37.5% or 75% depending on grading-prompt wording, was investigated further using three independently designed judges plus repeated runs, to separate genuine philosophical disagreement from measurement noise.

### Three judges, three different standards


- **Lenient**: passes an answer that captures the reference's main point, regardless of wording, length, or omitted secondary detail.

- **Strict**: same reference comparison, but requires the central claim to match without inventing unsupported detail; omissions the lenient judge tolerates are treated more critically.

- **Source-grounded**: has no access to the reference answer at all. Checks the generated answer directly against the retrieved code, and separately screens for genericness: would this exact answer still sound plausible as a response to a different, superficially similar question about the codebase? If so, it is marked incorrect regardless of factual accuracy.

### Reading the disagreements changed the diagnosis


An initial two-judge comparison showed 75% agreement, with every disagreement in the same direction: lenient passed, strict failed. Reading the six disputed answers against their references split them into two distinct patterns, not one:

- Genuinely shallow answers, correctly caught by the strict judge, for example, “The api.py module provides the Requests API for making HTTP requests,” which restates the question without saying anything a different module's description could not equally claim.

- Answers more detailed and more specific than the reference itself, incorrectly penalized for not matching the reference's generic phrasing, for example an answer naming resolve_redirects, SessionRedirectMixin, and merge_cookies by name, marked wrong because the reference only described the mechanism in general terms.

This motivated the source-grounded judge: grade against the actual code, not a single fixed phrasing of the answer, and add an explicit genericness test to still catch the first failure pattern.

### The first grounded design over-corrected


An initial version of the source-grounded prompt, checking only for false statements or invented behavior, returned a perfect score across every answer, including the genuinely shallow ones. A judge that never fails is not measuring anything; it had fixed the over-penalization problem by removing the ability to penalize at all.

Adding the explicit genericness criterion restored real discrimination: several previously-passing answers were correctly marked incorrect once the test was in place, without reintroducing the original over-strictness.

### Repeating each judge three times separated real disagreement from noise


A single run of any judge is a sample, not a measurement. Running the full pipeline three times end to end gave a mean and range for each grading standard:

| **Judge**               | **Mean**   | **Range (3 runs)** |
|-------------------------|------------|--------------------|
| Lenient (vs. reference) | 77.78%     | 75.00% – 83.33%    |
| Strict (vs. reference)  | **59.72%** | 58.33% – 62.50%    |
| Source-grounded         | 83.33%     | 79.17% – 87.50%    |

Run-to-run variance for each judge stayed under 9 points. The gap between judges, roughly 18 to 24 points between strict and the other two, is four times larger. **The disagreement is structural, not noise.** Strict reference-matching is a consistently harder standard than either lenient reference-matching or source-grounded factual checking; more sampling would not make the three converge.

### The honest reportable claim


There is no single accuracy number for this system's answer quality. Depending on grading philosophy, measured accuracy ranges from roughly 58% to 87%. Reporting only the most favorable figure, or only one judge's output, would misrepresent a genuinely multi-valued result as a single point estimate. The range itself, with the mechanism behind it, is the finding.

## Abstention and hallucination


Six unanswerable questions test whether the system invents behaviour for code that does not exist. All six were answered with an explicit refusal across every configuration tested: a 0.00% hallucination rate throughout.

*This result depends on an explicit instruction in the generation prompt to abstain rather than guess. It is a designed behaviour, not an emergent one. Abstention detection in the harness itself is currently simple keyword matching over the response text, which is adequate for six controlled cases but has not been stress-tested against a larger or adversarial set, a known and stated limitation rather than a hidden one.*

## Repository layout


| **File**                | **Responsibility**                                                                                                                                       |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| build_index.py          | Both chunking strategies (fixed-line and AST-based), embedding generation, persistence of the two indexes                                                |
| search.py               | Cosine-similarity retrieval; accepts an injectable chunk set and cost tracker so strategies share one code path                                          |
| search_decomposed.py    | LLM sub-query generation with pass-through detection for non-decomposable questions, per-sub-query search, score-max deduplication and merge             |
| search_hybrid.py        | BM25 index, min-max score normalisation, weighted fusion with tunable α                                                                                  |
| alpha_sweep.py          | Precomputes embeddings and BM25 scores once, sweeps α across eight values with zero additional API cost                                                  |
| generate_answer.py      | Context assembly and answer generation with abstention instruction                                                                                       |
| cost_tracker.py         | Records tokens and cost per API call across all strategies, including a manual cost formula for embedding calls that the gateway does not price natively |
| check_answer_quality.py | Three-judge end-to-end answer grading (lenient, strict, source-grounded) plus latency measurement                                                        |
| judge_variance_check.py | Repeats the three-judge evaluation across multiple runs to separate genuine disagreement from sampling noise                                             |
| run_eval.py             | Harness: runs every strategy against the eval set, scores multi-source questions strictly, reports failures and cost per strategy                        |
| eval_set.json           | The 36-question ground truth: question, reference answer, expected source files, category                                                                |

### Design notes


- Embeddings are computed once per chunking strategy and persisted, not recomputed per query. The alpha sweep applies the same principle to parameter tuning: embeddings and BM25 scores are computed once and reused across all eight alpha values, rather than re-querying the API for each.

Multi-source expectations are parsed with a regex over the source field rather than naïve comma-splitting, after an earlier version silently mishandled entries whose reference text contained a comma.

- Every strategy shares one search path via dependency injection of the chunk set and cost tracker, so comparisons differ only in the variable under test.

## Limitations


Stated plainly, because a comparison table without them would overclaim:

- None of the three tested pairwise comparisons in this project reach statistical significance at n=36 (see “A statistical check that changed the headline”). Chunking strategy and hybrid weighting both returned p=1.0; query decomposition returned p=0.25. A properly powered evaluation set, likely 80–100+ questions given the effect sizes being distinguished, is the single highest-priority next step, ranked above any new feature or technique.

- Resolved: whether the system depends on real retrieval or coasts on the generation model's training knowledge of psf/requests was tested with a matched-prompt closed-book control (see “Does retrieval even matter here?”). The system does depend on retrieval, confirmed at p \< 0.0001 on both judges. The benchmark itself is not contamination-proof: a no-context model given permission to guess freely scores roughly 77-80% on these questions from training familiarity alone, and that floor should be kept in mind when reading any accuracy number in this document.

*Resolved in this pass:* the single-judge, single-reference limitation was addressed with three independently designed judges and repeated runs. The result was not a single better judge, but a demonstrated, stable 18 to 24 point gap between grading philosophies, a more honest outcome than convergence would have been.

*Found and only partially resolved in this pass:* the hybrid-search conclusion reported earlier in the project (“never beats pure semantic”) was revised after a full alpha sweep to “hybrid does beat pure semantic, confirmed.” That revision was itself wrong: a repeat run only demonstrated the arithmetic was deterministic, not that the effect was statistically real, and McNemar's exact test on the actual comparison returned p=1.0. The current, honest state is that hybrid search and pure semantic search are not distinguishable with the data collected so far. This is flagged here explicitly as a limitation the project previously mischaracterized as resolved.

- Abstention is still detected by keyword matching over the generated text. Adequate for six controlled cases; unaddressed for a larger or adversarial set.

- Function-level retrieval precision has not been measured. Scoring checks whether the correct file was retrieved, not whether the correct function within it was, a coarser bar than it may appear from the headline numbers.

- Results are specific to a single 20-file Python corpus. Findings about which techniques help, and by how much, should not be assumed to generalise to a different codebase's file-size distribution or structure.

## Stack


Python · ast for structural chunking · text-embedding-3-small · gpt-4o-mini for generation, decomposition, and three-way judging · numpy · rank-bm25 · corpus: psf/requests (20 source files, 320 function-aware chunks)

# Evaluation Report — RAG and Machine Learning Components

**Industrial Operations Intelligence Platform**

---

## 1. Purpose and method

This report records what the platform's retrieval-augmented generation and
machine learning components actually do when measured. Every figure in it is the
output of a script committed to the repository, and every script can be re-run
to reproduce the number beside it.

Three rules governed the work and are stated up front, because they explain
several of the results below.

**Nothing was tuned to make a number larger.** Where a metric came out low, it
was recorded and explained rather than adjusted. Two changes were made to the
system during the evaluation — a decoding setting fixed for reproducibility, and
a defect corrected once measurement exposed it — and both are documented in
place, with the before and after figures shown.

**No language model grades another language model.** The only model available
locally is `llama3.2:3b`, the same model that generates the answers being
scored. A model marking its own work produces a number a reviewer is right to
discount. Every generation metric here is therefore deterministic and
rule-based: string presence, figure traceability, refusal-phrase matching.

**Unsupervised components are not given manufactured labels.** Two of the three
ML models are unsupervised over data that carries no failure labels. Inventing
labels in order to report precision and recall would measure the labelling, not
the model. Where classification metrics do not apply, this report says so and
reports what can be established instead.

### Scripts

| Component | Script | Result file |
|---|---|---|
| RAG retrieval | `evaluation/rag/retrieval_eval.py` | `retrieval_600.json` |
| RAG threshold | `evaluation/rag/threshold_analysis.py` | `threshold_analysis_600.json` |
| RAG generation | `evaluation/rag/generation_eval.py` | `generation_600_t0.json` |
| Forecasting | `evaluation/ml/forecast_eval.py` | `forecast_backtest.json` |
| Anomaly detection | `evaluation/ml/anomaly_eval.py` | `anomaly_eval.json` |
| Risk scoring | `evaluation/ml/risk_eval.py` | `risk_eval.json` |

---

## 2. Headline results

| Metric | Result | Target |
|---|---|---|
| Retrieval Hit Rate@5 | **91.67 %** | met |
| Retrieval MRR | **0.7306** | — |
| Refusal accuracy (out-of-scope declined) | **100.00 %** | met |
| Numeric groundedness | **94.74 %** | met |
| Anomaly detection recall (injected faults) | **84.92 %** | met |
| Answer correctness | **54.17 %** | not met |
| Citation rate | **41.67 %** | not met |
| Forecast skill against a mean baseline | **0.0026** | not met |
| False refusal rate | **20.83 %** | reported alongside refusal accuracy |

Four of the measures clear the eighty percent target and three do not. The three
that do not are the more informative results, and sections 4 and 5 explain each
one.

---

## 3. Evaluation corpus

The RAG evaluation uses the documentation of a simulated bottling plant, which
is consistent with the platform's data-simulator design. The plant has four
production lines and ninety days of operational data across thirteen metrics in
seven domains.

| Document | Chunks | Domains |
|---|---|---|
| Operations manual | 7 | Operations, Finance |
| Maintenance manual | 7 | Maintenance |
| Quality manual | 6 | Quality, Customers |
| Inventory policy | 5 | Inventory |
| Workforce and safety handbook | 5 | Workforce |

Thirty chunks in total, approximately 8,500 words. The corpus was deliberately
not padded to reach a larger chunk count; an earlier plan to write twenty
thousand words purely to inflate the index was abandoned as something that
would show.

**Ground truth.** Thirty questions were written by hand against the corpus:
eighteen in-scope, six distractors that resemble in-scope questions but require
a specific document, and six out-of-scope questions whose answers are genuinely
absent. No question or answer was model-generated. Relevance resolves from the
source document and expected keywords rather than chunk identifiers, so the same
ground truth scores any chunking configuration and remains valid if the corpus
is re-indexed.

**Disclosure.** The corpus documents a simulated plant. The documents were
written to be internally consistent with the operational data, which was
verified against the hub rather than assumed; where a document and the data
disagreed, the document was corrected, never the data.

---

## 4. RAG evaluation

### 4.1 Retrieval

Architecture: standard dense retrieval. Text is chunked with sentence-aware
splitting at 600 characters with 100 characters of overlap, embedded locally
with `all-MiniLM-L6-v2` into 384 dimensions, and searched by cosine distance in
pgvector with a distance cutoff of 0.75 and `top_k` of 5. There is no hybrid
search, no re-ranker and no query rewriting.

**Passage level**

| k | Hit Rate | Recall | Precision |
|---|---|---|---|
| 1 | 62.50 % | 30.87 % | 62.50 % |
| 3 | 79.17 % | 49.11 % | 34.72 % |
| 5 | **91.67 %** | 67.94 % | 28.33 % |

MRR 0.7306.

**By question type**

| Type | n | Hit@1 | Hit@5 |
|---|---|---|---|
| In-scope | 18 | 66.67 % | 88.89 % |
| Distractor | 6 | 50.00 % | 100.00 % |

Two caveats belong with these figures rather than in a footnote.

Document-level hit rate reaches 100 % at k=3, but with only five documents in
the corpus that is close to a structural certainty and carries little
information. It is reported for completeness and should not be emphasised.

Precision@k above k=1 is bounded by the ground truth, not by the retriever. Most
questions have a single gold passage, so Precision@5 cannot exceed 20 % however
well the system performs. Only Precision@1 is interpretable.

### 4.2 Distance thresholding

The retriever's ability to decline an unanswerable question rests on the
distance cutoff. Whether that mechanism can work at all was tested directly.

| Question set | n | min | median | max |
|---|---|---|---|---|
| Answerable | 24 | 0.3034 | 0.4551 | 0.6172 |
| Out-of-scope | 6 | 0.5101 | 0.6259 | 0.7349 |

The two distributions **overlap by 0.107**. No single cutoff separates them.

| Cutoff | Abstention | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|---|
| 0.50 | 100.00 % | 41.67 % | 41.67 % | 41.67 % |
| 0.55 | 83.33 % | 45.83 % | 54.17 % | 54.17 % |
| 0.60 | 66.67 % | 58.33 % | 66.67 % | 70.83 % |
| 0.65 | 33.33 % | 62.50 % | 79.17 % | 91.67 % |
| 0.70 | 16.67 % | 62.50 % | 79.17 % | 91.67 % |
| 0.75 (deployed) | 0.00 % | 62.50 % | 79.17 % | 91.67 % |

Two findings follow. First, cosine distance alone cannot perform abstention on
this corpus — that is demonstrated, not asserted. Refusal behaviour must
therefore be evaluated at the generation layer, which section 4.3 does.

Second, the deployed cutoff of 0.75 is **strictly dominated** by 0.65: the
tighter cutoff buys 33 % abstention at no cost to hit rate at any k. This is a
real improvement identified by measurement. It was deliberately not applied
before the baseline was locked, so that the baseline describes the system as
built rather than a system tuned in response to its own evaluation. It is
recorded in section 7 as future work.

### 4.3 Generation

Answers are generated by `llama3.2:3b` running locally under Ollama, with a
strict grounding instruction and an extractive fallback if the model is
unavailable.

**A reproducibility problem was found and fixed before the baseline was set.**
A three-question wiring check and the full run disagreed on question Q02 —
identical retrieval, identical top distance of 0.3617, different verdict. The
cause was that no decoding temperature was specified, so Ollama's default of 0.8
applied and the reported correctness figure was not reproducible.

The generation call was changed to `temperature: 0` with a fixed seed, and the
evaluation was run once more. Both runs are reported. It was agreed before that
run that whatever numbers it produced would stand, and no further run was made.

| Metric | Default sampling | **Deterministic (official)** |
|---|---|---|
| Refusal accuracy | 83.33 % | **100.00 %** |
| False refusal rate | 12.50 % | **20.83 %** |
| Answer correctness | 50.00 % | **54.17 %** |
| Numeric groundedness | 100.00 % | **94.74 %** |
| Citation rate | 33.33 % | **41.67 %** |

Greedy decoding did not simply improve the system; it made it **more
conservative**. Every out-of-scope question is now declined, but the number of
answerable questions wrongly declined rose from three to five. Refusal accuracy
of 100 % is meaningless without the false refusal rate beside it — a system that
refuses everything scores 100 % on the first measure — so the two are always
reported together.

Mean latency was 57.0 seconds per question on CPU. All 24 answerable questions
were answered by the model; none fell back to extraction.

**Metric definitions.**

*Refusal accuracy*: out-of-scope questions that produced the system's refusal
message. *False refusal rate*: answerable questions that produced it instead of
an answer. *Answer correctness*: every expected keyword present in the answer —
a strict measure, since a partially correct answer scores zero. *Numeric
groundedness*: every figure in the answer traceable to the retrieved context;
this is the faithfulness proxy, chosen because the corpus is largely thresholds
and the realistic hallucination is a decimal error such as `0.45` for `0.045`.
*Citation rate*: a `[Source N]` marker present.

**One flagged case, and what it says about the metric.** The groundedness check
flagged Q20 for the figure `350`. On inspection, that figure appears in the
question itself — the model echoed the premise rather than inventing a value.
This is a **false positive of the metric**, not a detected hallucination: the
check compares the answer against the retrieved context and does not consider
the question text. The limitation is recorded here rather than corrected after
the fact, and it is worth noting that a check which never fires would tell a
reviewer nothing about whether it works.

### 4.4 Failure attribution

Because retrieval and generation were evaluated over the same question set,
every failure can be attributed to a layer. This is the most useful result in
the RAG section.

Of the 24 answerable questions:

| | Generation correct | Generation failed |
|---|---|---|
| **Gold passage retrieved (top-5)** | 13 | **9** |
| **Gold passage not retrieved** | 0 | **2** |

Retrieval supplied a gold passage for 22 of 24 answerable questions (91.7 %).
Nine of those still failed at generation. The attribution is therefore **2
retrieval errors against 9 generation errors**: the generation model, not the
retriever, is the dominant bottleneck at this corpus size.

A sharper version of the same point: in **five** of the failures the gold
passage was ranked **first**. The correct text was at the top of the context
window and the answer was still wrong. No amount of retrieval improvement
addresses those cases.

The eleven failures divide into two kinds.

**False refusals (5)** — Q05, Q09, Q11, Q15, Q21. The model declined questions
whose answers were present in the retrieved context.

**Incomplete answers (6)** — Q01, Q06, Q12, Q14, Q17, Q22. The model answered on
topic but omitted the specific value the question asked for.

In every one of the eleven, the missing element is a **specific figure or
qualifying term**: `350`, `0.045`, `2.5`, `3` and `7`, `14`, `5`, `12`, `3
percent`, `single`, `planned`, `downtime`. Q06 is representative — asked for the
causes of rising energy per unit, the model produced a correct account but named
only some of the causes, omitting downtime.

Read together with 94.74 % groundedness, this gives the central finding of the
RAG evaluation:

> The model does not fabricate. It **under-extracts**. Correctness is limited by
> precision of extraction from correctly retrieved context, not by hallucination
> and not by retrieval quality.

This is a capability limit of a 3-billion-parameter model, and determinism did
not change it — the same failures recur with sampling disabled, which
establishes that the cause is the model rather than randomness.

Citation rate of 41.67 % has a separate and simpler cause: the system prompt
requests source markers but does not require them, and the four questions that
did cite were among the incorrect answers, so citation does not correlate with
correctness on this corpus.

---

## 5. Machine learning evaluation

### 5.1 Forecasting

The job fits one model per domain metric over the daily trend series, preferring
Holt-Winters with additive trend and falling back to a least-squares linear
trend on short series, and projects seven days ahead with 95 % bounds derived
from fitted residuals.

Rows in `ml.forecasts` are future-dated and can never be scored, since the
actuals do not exist. The evaluation therefore performs a **rolling-origin
backtest**: three folds per series, each holding out seven days, training on
the preceding history, and comparing the projection against the withheld values.
The production functions `_prepare_series` and `_forecast_series` are imported
and called directly, so the measurement applies to the deployed model rather
than a reimplementation.

**A configuration defect was found by running the evaluation.** The first run
logged that `statsmodels` was not installed, meaning Holt-Winters had never
executed and all 84 folds had silently used the linear fallback. The dependency
was declared explicitly — along with `scikit-learn`, which was likewise reached
only transitively — and the image rebuilt. Both runs are reported.

32 series, 84 folds scored, 4 series too short to backtest.

| | Linear fallback (defective) | **Holt-Winters (corrected)** |
|---|---|---|
| MAE | 202.51 | **232.41** |
| RMSE | 271.20 | **288.83** |
| MAPE | 21.61 % | **23.57 %** |
| sMAPE | 21.71 % | **22.98 %** |
| 95 % interval coverage | 0.9184 | **0.9252** |

**Against baselines**

| Baseline | MAE | Skill score | Folds won by model |
|---|---|---|---|
| Naive (last value) | 247.53 | +0.0611 | 59 / 84 |
| Seasonal naive (7-day) | **59.08** | **−2.9336** | 52 / 84 |
| Training mean | 233.00 | +0.0026 | 66 / 84 |

Three things follow, and none of them flatter the model.

The corrected configuration is **worse in aggregate than the defective one**.
Holt-Winters with an additive trend and no seasonal term extrapolates a slope
that the data does not sustain over a seven-day horizon; the simpler linear fit
happened to be more conservative.

Skill against the training mean is **0.0026** — arithmetically positive and
practically zero. Predicting the historical average would have performed the
same. A MAE of 232 in isolation says nothing; measured against a baseline, the
model is not currently earning its complexity.

The decisive result is the seasonal naive baseline: **MAE 59.08 against the
model's 232.41**. Simply repeating the value from seven days earlier is roughly
four times more accurate. The series carry strong weekly seasonality and the
model is configured with `seasonal=None`. Note that the model still wins 52 of
84 folds — it is not uniformly worse, but it loses very heavily on the
high-magnitude series where seasonality dominates, and those losses drive the
aggregate.

This is a **model configuration finding, not a data finding**, and it was
invisible before the backtest existed. It was deliberately not acted on: adding
a seasonal component after the evaluation reported this number would be tuning
to the measurement. It is recorded as future work in section 7.

Interval coverage of 0.9252 against a nominal 95 % is the one result that
behaves as designed — the uncertainty bounds are approximately honest even
where the point forecast is weak.

**By domain**

| Domain | Folds | MAE | sMAPE |
|---|---|---|---|
| Assets | 6 | 0.30 | 3.01 % |
| Workforce | 9 | 0.34 | 6.85 % |
| Quality | 12 | 4.78 | 22.08 % |
| Maintenance | 15 | 7.87 | 12.29 % |
| Customers | 12 | 251.45 | 23.62 % |
| Operations | 9 | 308.07 | 37.07 % |
| Finance | 12 | 376.74 | 27.55 % |
| Inventory | 9 | 1003.44 | 50.36 % |

MAE is not comparable across domains, since it carries the scale of the
underlying metric; sMAPE is. On that basis inventory is the weakest domain by a
wide margin at 50 % error, and operations follows at 37 %.

### 5.2 Anomaly detection

The job flags unusual readings per series using an isolation forest with
`contamination=0.1` and a fixed random state, falling back to a z-score at a
2.5 threshold when a series has fewer than eight points.

The detector is unsupervised and the operational data has no fault labels.
Two separate things are therefore measured, and they are reported separately
because they answer different questions.

**Behaviour on unmodified data — descriptive only, no accuracy claimed.**

298 observations flagged out of 3,167 across 32 series; flag rate 0.0941. All
via the isolation forest path. Severity: 251 medium, 47 high.

The flag rate of 9.41 % is essentially the `contamination` parameter reflected
back. This is worth stating plainly: the isolation forest flags approximately
ten percent of any series it is given, whatever the data looks like. The flag
rate is a property of the configuration, not a measurement of the data.

**Synthetic injection benchmark — labels are manufactured, and that is stated.**

Real series are taken from the hub, faults of known shape are injected at
recorded positions, and the production detectors are asked to find them. The
injected positions are the ground truth, so precision and recall are exact for
this benchmark. They describe behaviour on injected faults of known shape; they
are not a claim about real-world faults.

252 anomalies injected.

| Metric | Value |
|---|---|
| Recall | **0.8492** (214 of 252 recovered) |
| Precision (raw) | **0.2413** |
| Precision (adjusted for contamination) | 0.8992 |
| F1 (adjusted) | 0.8735 |

**The raw precision of 0.2413 is the measured value.** The adjusted figure
requires the explanation that follows and should never be quoted without it.
Because `contamination=0.1` forces the detector to flag about ten percent of
every series regardless of content, and faults were injected at a lower rate
than that, precision is bounded above by the ratio between the two — no
detector, however good, could score higher under this configuration. The
adjusted figure divides out that structural ceiling. It is a legitimate
adjustment, but it is a derived number, and the raw value is what was observed.

**By injection magnitude**

| Magnitude | Recall | Precision (adj.) | F1 |
|---|---|---|---|
| 2σ | 0.6310 | 0.8548 | 0.7260 |
| 3σ | 0.9167 | 0.9390 | 0.9277 |
| 5σ | **1.0000** | 0.8936 | 0.9438 |

This is the most operationally useful result in the ML section. Detection is
**effectively perfect for large deviations and unreliable for small ones**: a
five-sigma excursion is always caught, a two-sigma excursion is missed about
thirty-seven percent of the time. Users should expect the alerting to surface
severe events dependably and marginal events erratically.

### 5.3 Risk scoring

The job produces a 0–100 relative degradation risk per entity for the assets and
maintenance domains, from three min-max normalised components: adverse trend
(weight 0.45), volatility as coefficient of variation (0.25), and accumulated
anomaly severity (0.30).

**Classification metrics are not reported.** The scored entities carry no
failure labels. Precision, recall and F1 would require labels to be invented,
and would then measure the invention. Four things that can be established
without labels are measured instead.

**A defect was found and corrected.** The scoring function keyed anomaly weights
by `agg["dataset_id"].iloc[0]` — the first row's dataset identifier — and then
applied that single identifier to every entity in the group. In an incremental
run scoped to one dataset this is harmless. In a full-batch run spanning several
datasets, every entity outside the first dataset received an anomaly weight of
zero, silently removing thirty percent of the score. The lookup was changed to
use each row's own dataset identifier. The formula and the weights were not
touched. The production job was re-run afterwards, scoring 314 entities across
two domains.

**Distribution** (all datasets, after the fix)

| Domain | Entities | Range | Median | Bands |
|---|---|---|---|---|
| Assets | 155 | 2.34 – 50.52 | 19.25 | 152 low, 3 medium |
| Maintenance | 159 | 0.04 – 75.83 | 12.67 | 145 low, 13 medium, 1 high |

Neither domain uses the upper part of the 0–100 range. This is a direct
consequence of the min-max design: a component only reaches 1.0 for the single
most extreme entity in the population, and reaching a high composite score
requires being extreme on all three at once.

**Weight sensitivity.** Each component was zeroed in turn and the entities
rescored, with rank agreement measured by Kendall's tau. A tau near 1 means the
ordering barely moved, which means the component is not contributing.

| Component | Weight | Assets τ | Maintenance τ |
|---|---|---|---|
| Trend | 0.45 | 0.7005 | 0.4235 |
| Volatility | 0.25 | **0.1239** | 0.7156 |
| Anomaly | 0.30 | **0.9728** | **0.9820** |

Two results stand out and both are problems.

Volatility carries the *smallest* weight but dominates the assets ranking
(τ 0.124 — removing it almost completely reorders the domain). The stated
weights do not describe the actual influence, because min-max normalisation
rescales each component by the spread of its own population before the weights
are applied.

The anomaly component is **inert**: at τ 0.97 and 0.98, removing a component
carrying thirty percent of the nominal weight barely changes the ordering. The
cause is upstream. Anomaly rows are attributed to entities through
`analytics.daily_trend`, which has no entity dimension, so the detection job
resolves an entity per `(dataset, domain, metric)` group by taking the first
match. Most anomaly rows are therefore attributed to the wrong entity, the
weights concentrate on a few entities, and min-max normalisation flattens the
rest to zero.

Fixing this properly requires adding an entity dimension to the Phase 2
analytics tables, which is a schema change beyond the scope of this evaluation.
It is documented here as a known limitation rather than patched, and it is
recorded in section 7.

**Population stability** (leave-one-out: each entity dropped, remainder
rescored)

| Domain | Runs | Mean score shift | Max shift | Rank agreement τ |
|---|---|---|---|---|
| Assets | 155 | 0.0115 | 1.0585 | 0.9977 |
| Maintenance | 159 | 0.0326 | 5.0653 | 0.9991 |

At this population size the ranking is stable. That result does not generalise
downward, as the single-dataset case below shows.

**Convergent validity.** The ranking was compared against observables the score
never reads.

| Domain | Comparison | n | Spearman | Kendall τ |
|---|---|---|---|---|
| Assets | Mean downtime | 155 | 0.3352 | 0.2333 |
| Maintenance | Mean downtime | 159 | **−0.1115** | −0.1226 |
| Maintenance | Breakdowns | 4 | 0.2582 | 0.2357 |

Assets shows weak positive agreement. Maintenance shows weak **negative**
agreement: the risk ranking disagrees with observed downtime.

**Single-dataset view.** Because the platform accepts arbitrary uploads, no
dataset is canonical, and pooling every dataset into one min-max normalisation
mixes entities from unrelated businesses. Scoped to the four-line bottling
plant:

| Entity | Risk score | Band | Observed mean downtime |
|---|---|---|---|
| L-01 | 55.00 | medium | 7.33 min (best) |
| L-03 | 45.00 | medium | **90.50 min (worst)** |
| L-04 | 13.74 | low | 17.52 min |
| L-02 | 5.90 | low | 19.76 min |

Spearman against downtime: **−0.40**.

The line with the least downtime is ranked highest risk, and the line with
4.6 times the downtime of any other is ranked second. Leave-one-out mean shift
rises to 5.58 points with a maximum of 14.80 — an order of magnitude less stable
than the pooled population, because with four entities every score is defined
almost entirely by the other three.

This inversion is **by design, not by defect**. The score is built from trend
slope and coefficient of variation: it ranks by *deterioration and instability*,
not by *current condition*. L-01 operates at low absolute downtime, so small
absolute variations produce a large coefficient of variation. Both readings are
defensible, but they are not the same question, and the dashboard currently
presents the score without making clear which one it answers.

---

## 6. Limitations

1. The RAG corpus is five documents and thirty chunks. Document-level retrieval
   metrics are close to structurally guaranteed at this size, and Precision@k
   above k=1 is bounded by the single-gold-passage ground truth rather than by
   retriever performance.
2. The numeric groundedness check compares answers against retrieved context
   only. A figure restated from the question is flagged as ungrounded, which
   accounts for the one flagged case in the official run.
3. Answer correctness requires all expected keywords. A materially correct but
   partial answer scores zero, so the 54.17 % figure is a strict lower bound on
   answer quality.
4. Anomaly detection precision and recall are measured against injected faults
   of known shape. They characterise behaviour on synthetic excursions and do
   not transfer to real-world fault rates.
5. Raw anomaly precision is bounded by the `contamination` setting. The adjusted
   figure removes that ceiling arithmetically and should not be cited without
   the explanation in section 5.2.
6. The anomaly component of the risk score is effectively inert because
   `analytics.daily_trend` has no entity dimension and anomaly rows are
   attributed to entities approximately. This is a known upstream limitation.
7. Risk scores are min-max normalised across whichever population is scored
   together. They are relative, not absolute, and are not comparable across runs
   with different populations.
8. Rank statistics on small populations are coarse. Where a domain has four
   entities, a single position change moves a correlation coefficient
   substantially, and the four-entity figures should be read accordingly.
9. Generation was measured on a single deterministic run. Variance across seeds
   was not characterised.
10. All generation timings are CPU inference on a 3-billion-parameter local
    model and are not representative of hosted inference.

---

## 7. Findings not acted on

Each of these was identified by measurement and deliberately left unchanged, so
that the reported baseline describes the system as built rather than a system
tuned in response to its own evaluation.

| Finding | Evidence | Change indicated |
|---|---|---|
| Retrieval cutoff 0.75 is dominated by 0.65 | 33 % abstention at zero cost to Hit@k | Lower the cutoff and re-measure |
| Forecasting ignores weekly seasonality | Seasonal naive MAE 59.08 vs model 232.41 | Add a seasonal component, or adopt the seasonal naive as the baseline model |
| Holt-Winters underperforms the linear fallback | MAE 232.41 vs 202.51 | Select per series by backtest rather than by history length |
| Citations are requested but not required | 41.67 % citation rate | Make the source marker mandatory in the prompt |
| Anomaly component of risk score is inert | τ 0.97 / 0.98 under ablation | Add an entity dimension to `analytics.daily_trend` |
| Stated risk weights do not match actual influence | Volatility at weight 0.25 dominates at τ 0.124 | Normalise components on a common scale before weighting |
| Risk score ranks deterioration, not condition | Spearman −0.40 against downtime on a four-entity population | Label the metric explicitly, or add an absolute-condition component |

---

## 8. Conclusion

The retrieval layer performs well: a gold passage reaches the top five for
91.67 % of answerable questions, and failure attribution shows retrieval
accounts for only 2 of 11 answerable-question failures.

The generation layer is the binding constraint. It does not fabricate —
groundedness is 94.74 %, and the single flagged case is a limitation of the
check rather than a hallucination — but it under-extracts, omitting specific
figures from context that was correctly retrieved and frequently ranked first.
Correctness of 54.17 % reflects a capability limit of a 3-billion-parameter
local model, established as such by the fact that eliminating sampling
randomness did not change it.

Anomaly detection recovers 84.92 % of injected faults, with a clear and useful
degradation profile: reliable above three sigma, unreliable at two.

Forecasting does not currently justify its complexity. Skill against a mean
baseline is 0.0026, and a seven-day seasonal naive forecast is roughly four
times more accurate. The cause is identified — a missing seasonal component —
and left uncorrected so that the reported figure remains a measurement rather
than a target.

Risk scoring is stable at population scale and unstable at small scale, and its
ranking disagrees with observed downtime on the single-plant view because it
measures deterioration rather than condition. Two defects were found through
evaluation — one silent dependency fallback and one incorrect dataset key — and
both were corrected.

The evaluation's principal value is not the four metrics that met the target. It
is that measuring the system exposed two defects, one inert model component, one
dominated configuration parameter, and one case where a metric and its intended
meaning had diverged. None of these were visible from the system's own output.

# Native five-category expert/gate recovery: development audit

This is a **development-only** record. The baseline stays Qwen2.5-VL-7B;
neither a 32B model nor confirmation-set labels entered these comparisons.
The 1,500-question confirmation split remains sealed. The five categories
retain native answers: Boolean presence, color word, spatial entity, integer
count, and OCR free text.

## Frozen 500-question comparison

| Frozen proposal set and gate | Correct / 500 | Success | Harm | Net | Relative error reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-7B raw baseline | 326 | — | — | — | — |
| Original ridge gate, original proposals | 327 | 2 | 1 | +1 | 0.57% |
| sklearn nested gate, original proposals | 333 | 10 | 3 | +7 | 4.02% |
| sklearn nested gate, SAM3-cropped SigLIP2 color added | 335 | 13 | 4 | +9 | 5.17% |
| CatBoost included in nested gate, original proposals | 340 | 18 | 4 | +14 | 8.05% |
| CatBoost included in nested gate, SigLIP2 color added | **341** | **19** | **4** | **+15** | **8.62%** |
| CatBoost included, BLIP-VQA added instead | 328 | 15 | 13 | +2 | 1.15% |

The best development run is
`/mnt/data/misaya/vpc_performance/native_five_v1/dev_sklearn_catboost_gate_v5_v1`.
Its image-group five-fold out-of-fold predictions yield a McNemar exact
two-sided p-value of 0.00260 and an image-group bootstrap relative-error-
reduction interval of 3.52%–13.87%. These statistics do **not** correct for
the repeated expert/gate comparisons made on the same development split, so
they are not an independent confirmation claim. The 20% target would require
at least +35 net corrections (361/500 correct); the observed best is +15.

The best run improves all five category slices: presence 78→82, color 45→50,
spatial 46→48, count 80→82, OCR 77→79. These are small development slices
of 100 each, not five independent benchmark claims.

## Expert diagnosis

The original proposal set covered only 45 of 174 baseline errors even under
an illegal gold-label oracle. Official [SigLIP2](https://huggingface.co/google/siglip2-base-patch16-224)
on SAM3 object crops supplied 27 color disagreements but no additional
oracle-rescuable baseline error; its unconditional accept rule had 4 successes
and 9 harms. Official [BLIP-VQA](https://huggingface.co/Salesforce/blip-vqa-base)
increased the oracle coverage to 67, but unconditional acceptance had 37
successes and 117 harms, and adding it degraded the cross-fitted gate.
Regenerating the *unchanged* Qwen answer with token logprobs matched all 500
frozen answers and exposed logprobs for all 500, yet adding those confidence
features did not improve the cross-fitted gate. The regenerated answers were
never used as replacement predictions.

The improved gate uses released sklearn regressors and
[CatBoost](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor)
through a fixed model/threshold menu, nested image-group selection, and a
harm cost of 1.25. A label-free application script replays the frozen final
model on 500 development inputs (technical replay only; no resubstitution
accuracy is reported). The CatBoost dependency is installed under
`/mnt/data/misaya/python_libs/catboost`, not the system disk.

## Decision

This recovery improves the original prototype but **does not satisfy** the
teacher's ≥20% aggregate target on the native-answer composite. More model
stacking on the same 500 development labels is not a trustworthy shortcut.
Meeting the target on this fixed composite would require demonstrably
stronger, category-relevant correct proposals and a separately trained gate,
followed by one untouched confirmation audit. The earlier MME Yes/No VPC
result is a different task and must not be used to claim native-answer success.

## Follow-up: answer-type-specific threshold test (same development set)

We tested whether a shared acceptance threshold caused the low gain. For each
outer image-group fold, the gate selects a pooled model and threshold on inner
folds. A type-specific threshold is allowed only when its inner-fold utility
exceeds the pooled threshold by at least two units. The fixed menu is 0,
0.025, 0.05, 0.1, 0.2, 0.3, 0.4. No outer-fold labels choose thresholds;
proposal files and baseline answers remain frozen.

| Frozen proposals | Prior best correct / 500 | Type-threshold correct / 500 | Success / harm / net | RER |
| --- | ---: | ---: | ---: | ---: |
| V5 with SigLIP2 color | 341 | 340 | 18 / 4 / +14 | 8.05% |
| V6 with BLIP-VQA | 328 | 328 | 15 / 13 / +2 | 1.15% |

The final fitted type thresholds collapsed to the pooled threshold in all
five answer types in both runs. The V6 frozen pool has 487 disagreement
proposals across 311 questions. Its illegal gold-label oracle could rescue
at most 67 of 174 baseline errors, distributed as presence 16, color 17,
spatial 12, count 14, OCR 8. BLIP itself has 37 potential successes but 117
potential harms under unconditional acceptance. Thus type-specific thresholds
did not solve the gate bottleneck and the 20% target remains unachieved.
Further model selection on these same labels would increase development-set
selection bias; stronger proposals and disjoint gate-training data are the
defensible next route.

## Follow-up: three-outcome gate (success / neutral / harm)

The next fixed experiment trains a CatBoost multiclass model on frozen
candidate outcomes (harm, no correctness change, success), then scores each
candidate by `P(success) - 1.25 * P(harm)`. A single threshold is selected by
inner image-group folds in each outer training partition. The outer fold is
scored once. No model-family sweep or test-fold threshold tuning is used.

| Proposal pool | Correct / 500 | Success / harm / net | RER | Exact McNemar p | Image bootstrap 95% RER interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| V5 SigLIP2 pool | 335 | 11 / 2 / +9 | 5.17% | 0.0225 | 1.28%–9.29% |
| V6 BLIP pool | 334 | 18 / 10 / +8 | 4.60% | 0.1849 | −1.23%–10.24% |

Both outcomes are below the earlier 341/500 best and far below 361/500 needed
for 20% relative error reduction. This gate formulation therefore does not
close the gap. The 500-question set has already guided several expert and gate
choices, so all these numbers remain development results even where a
fold-based p-value is small. No claim of independent confirmation is warranted.

The new run directories are
`/mnt/data/misaya/vpc_performance/native_five_v1/dev_three_outcome_gate_v5_v1`
and `/mnt/data/misaya/vpc_performance/native_five_v1/dev_three_outcome_gate_v6_v2`.
The prediction SHA256 digests are
`48369080f63f85b7cb9925a13964636cdc8333ac47d7cff899e4aaf6005a339c` and
`20f0a26fe76663e757abd40518f6f265f52d3ac4d0913f92fdc22a0f31a2da05`.

## Category and expert diagnosis for reaching the aggregate target

The relevant deployable comparison is the best frozen V5 proposal pool plus
its five-fold out-of-fold CatBoost gate. Each category has 100 questions.

| Category | Qwen correct | Best gate correct | Gate success / harm / net | V5 oracle-rescuable questions |
| --- | ---: | ---: | ---: | ---: |
| Presence | 78 | 82 | 4 / 0 / +4 | 10 |
| Color | 45 | 50 | 7 / 2 / +5 | 12 |
| Spatial relation | 46 | 48 | 3 / 1 / +2 | 8 |
| Count | 80 | 82 | 2 / 0 / +2 | 7 |
| OCR | 77 | 79 | 3 / 1 / +2 | 8 |
| **Total** | **326** | **341** | **19 / 4 / +15** | **45** |

"Oracle-rescuable" means at least one frozen expert candidate is correct on a
baseline-wrong question, using labels only for this upper-bound diagnosis. It
does not mean the gate can identify it. The current gate captures 19 of those
45 cases. At least 20 more net corrections are needed for 361/500. With the
V5 candidate pool, even a no-new-harm gate would have to capture at least 20
of the remaining 26 oracle-rescuable questions (77%). The mathematical ceiling
is 371/500 if all 45 are recovered with no harm; no observed gate is close to
that ceiling.

The accepted routes in the best gate are selective: four SAM3 presence
corrections; ten Qwen3-VL color and three SAM3-mask color corrections; eleven
Qwen3-VL spatial and one geometry correction; two Qwen3-VL count corrections;
and three Qwen3-VL plus two PP-OCRv6/Qwen3-verified OCR corrections. Total 36
accepted out of 500. Net gains are positive in every category, but only +2 to
+5 each.

The actual outcomes for each accepted route are:

| Accepted route | Accepted | Success | Harm | Net |
| --- | ---: | ---: | ---: | ---: |
| SAM3 presence | 4 | 4 | 0 | +4 |
| Qwen3-VL color | 10 | 6 | 2 | +4 |
| SAM3-mask color | 3 | 1 | 0 | +1 |
| Qwen3-VL spatial | 11 | 3 | 0 | +3 |
| Geometry spatial | 1 | 0 | 1 | −1 |
| Qwen3-VL count | 2 | 2 | 0 | +2 |
| PP-OCRv6/Qwen3-verified OCR | 2 | 0 | 1 | −1 |
| Qwen3-VL OCR | 3 | 3 | 0 | +3 |

Removing only the two currently negative routes would recover two net points
on this already-inspected development set (diagnostic only), still far below
the needed +20. Also, the PP-OCR verifier's proposal-level 3/1 summary does
not transfer to the two decisions accepted by this gate: that accepted subset
was 0/1 success/harm. Any expansion must evaluate the gate selection out of
fold.

Per-expert success/harm counts below are proposal-level outcomes if every
proposal from that expert were accepted. Counts use the V6 BLIP proposal pool,
except the SigLIP2 line, which comes from the earlier V5 audit. Several
proposals can refer to the same question, so these counts must not be summed
as unique questions.

| Category | Expert | Candidate proposals | Potential success | Potential harm | Diagnosis |
| --- | --- | ---: | ---: | ---: | --- |
| Presence | SAM3 presence | 4 | 4 | 0 | High precision, tiny coverage |
| Presence | Qwen3-VL-8B | 25 | 10 | 14 | Useful candidates, too many harmful ones |
| Presence | BLIP-VQA | 21 | 10 | 11 | Mixed; gate did not use it in the best run |
| Color | Qwen3-VL-8B | 25 | 9 | 4 | Best proposal balance among these color sources |
| Color | SAM3 mask color | 28 | 5 | 9 | Harm-prone |
| Color | SigLIP2 crop color | 27 | 4 | 9 | Added no new oracle-rescuable question in its audit |
| Color | BLIP-VQA | 35 | 11 | 8 | Some headroom, needs selective gate |
| Spatial | Qwen3-VL-8B | 46 | 8 | 12 | More rescue potential than geometry, still noisy |
| Spatial | Grounded geometry | 17 | 3 | 6 | Weak precision and limited coverage |
| Spatial | BLIP-VQA | 54 | 6 | 16 | Harm-prone |
| Count | CountAnything | 51 | 4 | 35 | Main failure source; do not accept directly |
| Count | Qwen3-VL-8B | 17 | 5 | 6 | Better balance, but low coverage |
| Count | BLIP-VQA | 30 | 9 | 17 | Noisy |
| OCR | PP-OCRv6 + Qwen3 verified | 5 | 3 | 1 | Strongest OCR candidate quality, very low coverage |
| OCR | PP-OCRv6 single-line | 5 | 0 | 2 | Not useful in current form |
| OCR | Qwen3-VL-8B | 38 | 8 | 16 | Mixed |
| OCR | BLIP-VQA | 86 | 1 | 65 | Disable for OCR; overwhelmingly harmful |

The most actionable expert work is therefore category-specific: expand the
verified OCR route while retaining its verification; replace or repair the
count proposals before trying to raise count routing; preserve the high
precision SAM3 presence route and increase its coverage; and prioritize better
spatial evidence proposals. Color already contributes the largest realized
net gain, so changing its gate alone is unlikely to supply the missing +20.
The existing best gate already rejects most harmful experts. Better gating
alone has limited room unless it can recover most of the 26 remaining V5
oracle-rescuable questions with almost no additional harm.

The remote run directories are
`/mnt/data/misaya/vpc_performance/native_five_v1/dev_type_threshold_gate_v5_v1`
and `dev_type_threshold_gate_v6_v1`. Their prediction SHA256 digests are
`19b82d0ff9bf149b75affc27719e329513de456ac6bec5d4fd55a85eef38abcd`
and `fce6552d7f9c66949e2c2c4e9f1a3196b05278f552e8205d80adb04c658741e6`.

## Follow-up: verified CountGD, OCR phrase, and target-crop color (2026-09-28)

The author-released CountGD checkpoint was verified against the original
model definition (1,146/1,146 parameter keys; zero missing, unexpected, or
shape-mismatched keys). On the frozen 100 native count questions, its default
0.23 detection threshold produced 34 exact answers versus 80 from raw
Qwen2.5-VL-7B. Adding CountGD to the V5 proposal pool yielded 336/500 with
16 successes and 6 harms under the nested gate, below V5's 341/500. The
partial Hugging Face conversion with 38 missing keys is retained only as an
invalid diagnostic and must not be cited as CountGD performance.

An exploratory OCR phrase/cross-route candidate source generated five new
proposals but **zero additional oracle-rescuable baseline errors**; accepting
them all would cause four harms. The new source is disabled by default and
must not enter a frozen deployment gate.

A question-targeted color crop was generated from frozen SAM3 boxes without
answer labels: 38 of 100 color questions had one usable target crop, 14 had
ambiguous instances, and 48 had no matching box. The same released
Qwen3-VL-8B expert ran on all 38 crops with artifact hashes frozen. The
development-only candidate audit found 15 disagreements with raw Qwen:
five potential successes, three potential harms, and only **one** newly
oracle-rescuable error beyond V5. This is candidate quality, not a realized
accuracy gain, and it is not integrated into V5.

The next defensible work is to construct a genuinely disjoint training pool
from training-split GQA/TallyQA/TextVQA annotations, audit image IDs and
pixel near-duplicates against both existing splits, then train/select one
gate before a single 1,500-question confirmation pass. The official GQA
archive download stalled; a disclosed third-party training Parquet mirror
was used to freeze the training manifest described below. The builder never
reads the confirmation labels. The sealed confirmation split remains
untouched; no ≥20% native-answer claim has been established.

The default proposal builder was replayed after adding an opt-in exploratory
OCR flag. Its 500-row proposal SHA256 is
`a105cdc8ff46af7d4d498ee0b0fc653b870e0a32448f6e6c06673d8fc3b855d4`,
byte-for-byte identical to frozen V5. Thus the failed OCR branch does not
silently alter the current best candidate pool.

## Disjoint training-pool manifest (2026-09-28)

The official GQA questions archive did not transfer from its source host, so
the GQA balanced **training** questions came from the third-party
`alexwww94/GQA` Parquet mirror, repository commit
`78362012630761a31ea0c469d0e5708b22896412`. The 183,979,778-byte
Parquet SHA256 is
`3643608e61ce3b94812decb37d50a0dea3b6b771bfa08ffc8fd93b187da57399`.
This is a provenance limitation and must not be described as the official
archive. TallyQA and TextVQA annotations came from their training splits;
TextVQA train JSON SHA256 is
`95f5c407db56cba56a177799dcd685a7cc0ec7c0d851b59910acf7786d31b68a`.

`train_pool_v1` is now frozen under
`/mnt/data/misaya/vpc_performance/native_five_v1/`: 2,500 native-answer
questions, 500 in each of presence, color, spatial relation, count and OCR,
on 2,500 distinct images. The builder read only the development and
confirmation `question_only.jsonl` and `image_requests.jsonl` plus their
frozen hashes, excluding 1,200 GQA, 400 TallyQA and 400 TextVQA withheld
image IDs. It did **not** open `sealed_labels.jsonl`. The manifest's
question/request/training-label SHA256s are respectively
`30da8f126d93eb8c212a3f5b7c4e63118a7b389aa6daa47e13fc408491dfc7211`,
`25123e58fed143aad1a7cfb851c4e4c8a53c3962c400245a8f807de33c42de7c4`,
and `1936092d714374249d816903d9d2b280947cdc41656441557d6484d099352405`.

This is a manifest, **not yet a usable training set**: image retrieval and
pixel near-duplicate checks against historical and withheld images remain
mandatory. No gate retraining or confirmation evaluation has occurred from
this pool, and the best verified development result is still V5's 341/500.

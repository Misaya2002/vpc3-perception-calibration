# Native-answer five-category evaluation (started 2026-09-26)

## Why this is separate from the binary confirmation

`independent_five_v1` asks yes/no verification questions in all five perceptual
categories.  Its TallyQA count and TextOCR word-presence questions were derived
from source annotations.  It cannot establish performance on native count or
free-text OCR answers, and its results must never be presented as such.  Keep
that frozen run and its failed pre-inference pixel audits intact as an auxiliary
verification experiment.

## Native tasks and sources

The new manifest retains source question wording and answer space:

| Category | Source | Native answer |
| --- | --- | --- |
| Object presence | GQA balanced validation | yes/no |
| Color | GQA balanced validation color query | color name |
| Spatial relation | GQA balanced validation relational query with spatial wording | related object name |
| Count | TallyQA official test question | integer |
| OCR/text reasoning | TextVQA v0.5.1 validation question | free text, ten human references |

This is a predeclared **multi-source composite**, not an official single-benchmark
score.  GQA validation images have appeared in other misaya projects.  TextVQA
validation is not the official test-std leaderboard.  Image-level development/
confirmation separation, prior-image byte and perceptual overlap checks, source
hashes, and image availability checks must pass before inference.  No image or
question may be replaced based on a model answer or evaluation label.

`scripts/build_native_five_manifest.py` selects 100 development and 300
confirmation questions **per category**, deterministically by SHA-256 rank and
with no image shared across those two splits.  It excludes the image requests
of the ongoing binary run without opening that run's sealed labels.  The output
contains a `question_only.jsonl` for inference and a separate
`sealed_labels.jsonl`.  The latter may be used for development, but confirmation
labels remain closed until the native method, predictions, and hashes are
frozen.  Selection uses source annotations only, not VPC predictions.

## VPC3 primary metric and scoring

For GQA and TallyQA, use normalized exact answer correctness; count answers
are integers.  Normalization is NFKC/lowercase, punctuation and articles
removed, and English number words zero–twenty mapped to digits. For TextVQA,
the composite accepts a normalized free-text answer if its vote
count among ten references reaches `min(3, maximum vote count)`; this avoids an
impossible-to-answer low-agreement item.  Also report ten-annotator VQA-style
soft accuracy `min(matching votes/3, 1)` separately. TextVQA predictions remain
free text; its local consensus score is **not** the official leaderboard score. The VPC3
primary aggregate is absolute accuracy gain in percentage points:
`100 * (calibrated correct - baseline correct) / question count`. Report the
original and calibrated accuracy, this percentage-point difference, each
category's accuracy and difference, correction successes, harms, net,
McNemar's paired test, and an image-group bootstrap interval for the accuracy
difference. Relative error reduction is not a primary VPC3 target. Do not
select a scoring rule or source weighting after seeing confirmation predictions.
`scripts/audit_native_five.py` checks question/prediction hashes and all 1,500
paired predictions before opening the confirmation labels.

## Method boundary and confirmation gate

The existing VPC V2 code normalizes every answer to yes/no; it is **not** a
native-answer calibrator and must not be applied unchanged here. VPC3 now has
a native-answer batch application output with structured expert evidence and
an accuracy-gain audit. Candidate generation, evidence, decision rule,
normalization and thresholds still need to be frozen together before
confirmation inference. The current application scripts consume precomputed
expert proposals; they are not yet a single end-to-end service. No accuracy
claim on the confirmation set is authorized merely by constructing this
manifest.

## Pre-inference preparation log

`manifest_v1` froze 500 development and 1,500 confirmation native questions,
one per image, with zero image overlap between splits and no labels in the
question-only files.  All 500 development images were obtained and validated.
Seven original-source downloads timed out on the server; exactly those six
TallyQA and one TextVQA images were fetched from the same public URLs through
the local computer, transferred to `/mnt/data/misaya`, byte-hash checked and
validated.  The failed download report was retained.  No questions changed.

The first development pixel audit failed **before model inference**: 96 images
matched prior misaya research fingerprints and one candidate had a within-set
near duplicate.  Its audit is retained.  `manifest_v2` was then built with the
same hash ordering and source rules, excluding only those 97 flagged image
keys.  This is a data-integrity replacement, not outcome-based selection.
Neither the native development nor confirmation labels have been used to tune
a native method; confirmation labels must stay sealed until prediction freeze.

The successive pre-inference development audits found 32, 12, 3 and 1 further
historical overlaps in V2–V5, and one more in V6.  All failed audits and
manifests remain intact.  `manifest_v7` has 500 complete development images;
its pixel audit **passed**, with zero historical and zero within-set matches
against the specified caches plus the current binary candidate images.
The SHA-ranked replacement rules, per-category counts and native wording did
not change.  The fair-format Qwen baseline V2 was frozen on all 500 development
questions before development labels were opened: 326/500 correct (65.2%);
presence 78/100, color 45/100, spatial 46/100, count 80/100, OCR 77/100.
The earlier baseline V1 (322/500) remains as a prompt-format diagnostic.

Two label-free, image/question/answer rechecks were then frozen and audited
on the same **development** questions.  The generic prompt V1 gave 221/500
(44.2%), success 8, harm 113, net -105.  Inspection found it returned Yes/No
on 76 of 100 TallyQA count questions, so it is invalid as a native-answer
method.  The question-typed prompt V2 and deterministic output-type gate
removed that format error but still gave 314/500 (62.8%), success 23,
harm 35, net -12.  Its count branch was 65/100 versus the 80/100 frozen
baseline; the other four branches were 78, 48, 47 and 76/100.  Thus generic
self-rechecking is rejected, and neither result supports the teacher's 20%
requirement.  No confirmation prediction or confirmation-label audit has
started.  Development findings may inform an independent-evidence method,
but any final rule and score must be frozen before confirmation.

## Independent-evidence development status (26 September 2026)

The finalized answer-type gate preserves the native answer space. It uses
question-only routing first and allows an image/question tool selector to
resolve ambiguous *evidence needs* only; the selector may not turn an open
answer into yes/no. The frozen `dev_router_final_v1` routes 500/500 questions.
OCR evidence was extracted on 100 development images without exposing answer
keys. On the development OCR subset, 9 of 22 baseline errors with an OCR run
had a gold answer exactly present in a recognized line (13 had a gold string
somewhere in concatenated OCR context). This is oracle coverage, not
correction accuracy.

The original conservative SAM3-style No→Yes presence specialist, applied only
after the native gate, changed four answers and achieved success 4/harm 0 on
the 500-question development set. For native color, SAM3 produced masks for
58/100 images. Among 55 baseline color errors, seven had the gold color among
mask candidates and six had it as the highest-confidence candidate; nine of
45 baseline-correct questions had a wrong highest-confidence mask color.
This ruled out an unconditional color override. A development-selected
threshold of 0.60 for a single palette-valued baseline answer was then frozen
in `predict_native_presence_color_v1.py` and audited: 326/500 → 333/500,
success 8/harm 1, net +7, accuracy 65.2%→66.6% (+1.4 percentage points),
exact McNemar p = 0.0391. Presence contributes +4; color contributes +3 net; spatial,
native count, and OCR remain untouched. This is a **development candidate**,
not a five-category successful final method and far below 20% overall.

The strict simple-count branch excludes qualified, action, attribute and
relational questions using only question text. It extracted SAM3 evidence for
54 of 100 native count questions. Among these 54, baseline errors were nine;
SAM3's NMS count equaled the reference in only one. On 45 baseline-correct
questions, the tool count disagreed with the reference on 13. Therefore this
count candidate must not be used to overwrite native answers. All original
development-only diagnostics, request manifests, evidence and hashes are
retained under `/mnt/data/misaya/vpc_performance/native_five_v1/`.

The next method gate is *not* more threshold search on these 500 labels.
Spatial open-entity answers (54 baseline errors), qualified count targets and
OCR retrieval need independently grounded candidate generation/verification.
Before any native confirmation run, the final single deployed method must
produce real native outputs for all five types, freeze every parameter and
prediction, and retain a separate no-change outcome when evidence is weak.

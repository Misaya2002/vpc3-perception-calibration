---
tags:
- visual-question-answering
- answer-calibration
- catboost
---

# VPC3 frozen native-answer gate

This repository contains the small **trained gate only**. It does not contain
Qwen3-VL, SAM3, OCR, datasets, or the VPC3 source code. Run it with the matching
code release at `Misaya2002/vpc3-perception-calibration`.

Files:

- `gate_model.joblib`: final CatBoost gate trained on the 500-question development set.
- `gate_model.json`: feature order, decision threshold (0.1), and model hash.
- `artifact_manifest.json`: SHA256 checksums and training environment versions.

Use Python 3.10.20, NumPy 1.26.4, scikit-learn 1.7.2, joblib 1.5.3, and
CatBoost 1.2.10. Download the exact repository revision and verify both file
hashes against `artifact_manifest.json` before loading the joblib file. Only
load this serialized model from a trusted source.

The gate scores the *incremental utility of adopting an expert candidate*.
Its score is not an answer-correctness probability. The reported 326/500 to
341/500 development result was obtained from five-fold out-of-fold predictions
across the development set; it is not an independent evaluation of this final
serialized model on new data. New-image performance must be measured separately.

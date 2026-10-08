# SHL Hiring Assessment 2026 - Grammar Scoring Engine for Spoken Audio

Kaggle user: `saurabingh` | Task: predict a 0-5 grammar score (MOS Likert) from a 45-60 s spoken clip | 769 training clips, 216 test clips | Metrics: RMSE (leaderboard score, lower is better) and Pearson correlation.

This repository holds three successive submissions. Each notebook is self-documented (report, training RMSE, visualisations). The final submission is **Version 3**.

| Version | Notebook | Approach | Honest CV RMSE | Public leaderboard (RMSE) |
|---|---|---|---|---|
| 1 | `notebookd736d18f9e.ipynb` | Whisper encoder embedding -> Ridge + SVR | 0.537 (5-fold, all 769 clips) | 0.4277 |
| 2 | `v2.ipynb` | 4 frozen views -> Ridge + SVR per view -> non-negative stack | 0.4895 (nested, genuine clips) | 0.3616 |
| 3 (final) | `shl_v3_submission.ipynb` | Same 4 views, 3 tuned regressors each (12 models) -> non-negative stack | 0.4803 (nested, genuine clips) | **0.3546** |

CV numbers are not directly comparable between version 1 and the others: version 1 scored all 769 clips, including the 37 garbled zero-score ones (see below), while versions 2 and 3 score only the 732 genuine clips. The public leaderboard uses roughly 60 % of the test set, so small differences between submissions are within noise.

---

## Key data observation

37 training clips have label `0`. They all have file IDs 5037-5073, their transcripts are off-topic or garbled, and they form a group that is easy to separate from the rest. The test file IDs are 0-215, so the test set almost certainly contains none of them. Versions 2 and 3 therefore report every CV score on the 732 genuine clips only (version 3 also trains on those only), because scoring all rows would flatter the metrics.

---

## Version 1 - Whisper encoder baseline

* **Features:** Whisper-large-v3-turbo *encoder* only. Audio is resampled to 16 kHz mono and processed in 30 s windows; padding frames are excluded. The hidden states of layers 16, 24 and 32 are mean-pooled over time and concatenated (3 x 1280 = 3840 dimensions).
* **Models:** `StandardScaler` -> `RidgeCV` and an RBF `SVR`; predictions are clipped to [0, 5] and averaged.
* **Evaluation:** 5-fold CV RMSE 0.537, Pearson 0.903 (all 769 clips). Training RMSE 0.237.
* **Public leaderboard:** 0.4277.

## Version 2 - four-view stack

Grammar score in this data behaves like a holistic spoken-English proficiency score, so four different views of each clip are combined:

| View | Model | Embedding |
|---|---|---|
| `whisper` | Whisper-large-v3-turbo encoder | mean over time, layers 16, 24, 32 |
| `qwen` | Qwen2.5-1.5B on the Whisper transcript | mean over transcript tokens, layers 16-22 |
| `wavlm` | WavLM-large | mean + std over time, layers 13, 18, 21 |
| `xlsr` | wav2vec2 XLS-R-300m | mean + std over time, layers 8, 13, 17 |

* **Transcripts:** Whisper (English, greedy, 30 s chunks). For Qwen a fixed prefix is prepended and excluded from pooling, because the first token of a causal LM has extreme "attention-sink" activations.
* **Layers** were chosen from per-layer ridge sweeps (late layers were best for every model).
* **Models:** Ridge + SVR per view, then a **non-negative linear stack** (NNLS) of the four view predictions.
* **Why it works:** the audio encoders carry fluency and pronunciation, which dominate the label; the transcript view is weaker alone but its errors are the least correlated with the audio views, so it helps the blend the most.
* **Evaluation:** 5-fold x 3-repeat CV with identical splits for every model; the stacker is scored with an outer 10-fold CV (nested). Nested CV RMSE 0.4895 on genuine clips; training RMSE 0.187 on genuine clips.
* **Public leaderboard:** 0.3616.

## Version 3 - tuned 12-model stack (final)

Same four views as version 2, with these changes:

* Three regressors per view instead of two: `RidgeCV`, a **tuned RBF-SVR** and a **tuned RBF kernel ridge** -> 12 level-1 models. SVR / kernel-ridge hyper-parameters come from a grid sweep (scored on a different CV split seed from the one used for the final numbers).
* **Trained on the 732 genuine clips only.**
* **Transductive standardisation:** feature mean / std are computed on train + test features (no labels involved).
* Predictions clipped to [1, 5]; NNLS stack over the 12 columns, scored with nested CV.
* **Evaluation:** nested CV RMSE 0.4803 on genuine clips (training RMSE on genuine clips is printed in the notebook). Reported training RMSE over all 769 rows is higher because the 37 zero clips are deliberately excluded from training.
* **Public leaderboard:** **0.3546**.

---

## What did not help

A further experiment (verbatim, disfluency-preserving Whisper transcripts and larger Qwen2.5-3B / 7B embeddings) reached a nested CV RMSE of 0.4787 versus 0.4803, a gain far below the noise level (~0.005), so it was not submitted. The mechanism worked (fillers per clip rose from about 0.2 to 2.7), but the new text views had residual correlation of 0.91-0.96 with the existing Qwen view, i.e. they carried the same information. Hand-crafted features (CoLA acceptability, ASR confidence, pauses, LLM surprisal) and per-layer stacking also added nothing measurable.

## Limitations

* Only 732 genuine training clips; CV differences below about 0.005-0.01 RMSE are noise, and the public leaderboard (about 130 clips) has a standard error of roughly 0.02.
* ASR tends to repair ungrammatical speech, which weakens the transcript-based view.
* Predictions regress toward the middle of the scale (clips truly scored 2.0 are over-predicted, scores of 4.5-5.0 under-predicted).
* Possible next steps: a sequence-aware pooling head over frame-level features, or fine-tuning an audio encoder.

## Reproducing

All notebooks are written for Kaggle (competition data under `/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final`, outputs to `/kaggle/working`).

* **Version 1 and 2:** run top to bottom on a GPU session with internet on; they extract embeddings themselves (a few minutes per backbone on a T4).
* **Version 3:** needs the cached backbone features (`F_*_alllayers.npy`, `qwen_*.npy`, `wavlm_*.npy`, `xlsr_*.npy`). These are produced by the research notebook (`new-version-3-r`, step "CELL 1b"), which builds every cache from the raw audio. Attach that notebook's output as an input; version 3 itself then runs on CPU in a few minutes.
* Final predictions: `submission.csv` (216 rows, columns `filename,label`).

## Environment

Python 3, PyTorch, Hugging Face `transformers`, `librosa`, `scikit-learn`, `scipy`, `numpy`, `pandas`, `matplotlib`, `joblib`. Models used: `openai/whisper-large-v3-turbo`, `Qwen/Qwen2.5-1.5B`, `microsoft/wavlm-large`, `facebook/wav2vec2-xls-r-300m`.
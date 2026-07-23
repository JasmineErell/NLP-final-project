# Project Progress

## Project

**Title:** Fine-Tuning BERT for Masked Note Prediction in Symbolic Music  
**Dataset:** Cleaned Nottingham Folk Music dataset in ABC notation

## Current Stage

```text
ABC parsing and tokenization completed
→ Full corpus processed
→ Song-level splits and train-only vocabularies created
→ Fixed masked evaluation examples created
→ Unigram baseline implemented and evaluated
→ Standard left-to-right n-gram implemented
→ Next: evaluate bigram, trigram, and 4-gram
```

---

## Completed Work

### 1. Parsing and preprocessing

Implemented:

- Splitting ABC collection files into individual songs using `X:`.
- Extracting title, rhythm, meter, default note length, key, and parts.
- Separating the musical body from the header.
- Detecting quoted chord annotations.
- Creating structured chord–melody segments.
- Saving atomic and factorized token sequences.
- Generating corpus statistics, token frequencies, examples, and error logs.
- Running the complete pipeline through `src/run_pipeline.py`.

### 2. Token representations

#### Atomic representation

Each complete chord–melody segment is one token:

```text
CHORD_G__MELODY_G2A2 B2c2 |
```

It follows the approved proposal closely, but produces a highly sparse vocabulary.

#### Factorized representation

The same segment becomes reusable tokens:

```text
[SEGMENT_START]
[CHORD_G]
[EVENT_G2]
[EVENT_A2]
[GROUP_BOUNDARY]
[EVENT_B2]
[EVENT_c2]
[BAR]
[SEGMENT_END]
```

The factorized representation is currently the main representation for modeling.

### 3. Factorized tokenizer validation

The tokenizer was corrected and manually checked against generated examples.

Confirmed support for:

- Fractional durations such as `/2` and `/4`.
- Accidentals and octave marks.
- Rests.
- Bars and repeats.
- Tuplets such as `[TUPLET_3]`.
- Alternative endings: `[ENDING_1]` and `[ENDING_2]`.
- Key and meter changes.
- Part markers such as `[PART_A]`.
- Simultaneous notes using `[MULTI_START]` and `[MULTI_END]`.

The previous false event tokens created from structural metadata were removed.

---

## Full Corpus and Split Results

### Song-level split

| Split | Songs |
|---|---:|
| Training | 816 |
| Validation | 102 |
| Test | 103 |
| **Total** | **1,021** |

Configuration:

```text
Seed: 42
Split: approximately 80/10/10
Vocabulary source: training set only
```

### Vocabulary statistics

| Metric | Atomic | Factorized |
|---|---:|---:|
| Vocabulary size including special tokens | 12,397 | 446 |
| Unique training tokens | 12,392 | 441 |
| Training singleton rate | 63.48% | 17.91% |

### Prediction-target OOV

OOV means that a target appears in validation or test but never appeared in training.

| Representation | Validation OOV | Test OOV |
|---|---:|---:|
| Atomic | 61.09% | 63.26% |
| Factorized | 0.29% | 0.26% |

Main conclusion:

> The atomic task has too many impossible held-out targets, while more than 99.7% of factorized targets are represented in the training vocabulary.

The atomic representation remains useful as a sparsity analysis. The factorized representation is the main modeling representation.

---

## Fixed Masked Evaluation Sets

Fixed single-mask examples were generated with a 15% selection rate and seed 42.

| Representation | Split | Examples | Known | Unknown |
|---|---|---:|---:|---:|
| Atomic | Validation | 409 | 160 | 249 |
| Atomic | Test | 453 | 166 | 287 |
| Factorized | Validation | 1,552 | 1,548 | 4 |
| Factorized | Test | 1,609 | 1,606 | 3 |

The same fixed examples will be used for all comparable models.

---

## Model Infrastructure

Current structure:

```text
src/
├── models/
│   ├── __init__.py
│   ├── common.py
│   ├── unigram.py
│   ├── ngram.py
│   ├── lstm.py
│   ├── bert_mlm.py
│   └── autoregressive.py
└── evaluate_unigram_ngram.py
```

### Unigram baseline

The unigram model:

- Counts eligible target tokens in the training data.
- Ignores musical context.
- Uses the same global frequency ranking for every mask.
- Reports Top-1, Top-3, Top-5, MRR, and known-target metrics.
- Saves metrics, counts, and per-example predictions.

#### Factorized validation results

| Metric | Result |
|---|---:|
| Top-1 accuracy | 6.64% |
| Top-3 accuracy | 18.43% |
| Top-5 accuracy | 27.71% |
| MRR | 18.90% |
| Known-target Top-1 | 6.65% |

This is the frequency-only baseline that context-aware models should outperform.

### N-gram baseline

The first implementation used tokens on both sides of the mask. It was useful as an exploratory local-context experiment, but it was not a standard forward n-gram.

The implementation has now been replaced with a standard **left-to-right n-gram**:

```text
Unigram = no previous context
Bigram  = 1 previous token
Trigram = 2 previous tokens
4-gram  = 3 previous tokens
```

The updated model:

- Uses only tokens before the masked position.
- Backs off to shorter left contexts when the full context was unseen.
- Finally backs off to unigram frequencies.
- Uses the same fixed validation examples as the unigram baseline.

The earlier bidirectional-radius results are not treated as the final n-gram results.

---

## Current Methodological Decisions

- Split complete songs to prevent leakage.
- Build vocabularies from training songs only.
- Keep validation and test masks fixed.
- Use factorized events for the primary model comparison.
- Keep atomic results mainly as evidence of sparsity.
- Use standard left-to-right n-grams as the classical baseline.
- Use validation data for model selection.
- Do not use the test set until all design choices are finalized.
- Report both overall and known-target metrics.
- Use Top-1 accuracy as the primary baseline-selection metric, supported by Top-3, Top-5, and MRR.

---

## Progress Checklist

- [x] Repository setup
- [x] Dataset added and inspected
- [x] ABC collection splitting implemented
- [x] Metadata and musical-body extraction implemented
- [x] Chord–melody segmentation implemented
- [x] Atomic tokenization implemented
- [x] Factorized tokenization implemented
- [x] Factorized tokenizer corrected and manually validated
- [x] Full corpus preprocessing completed
- [x] Song-level 80/10/10 split created
- [x] Train-only vocabularies created
- [x] Validation and test OOV calculated
- [x] Fixed masked evaluation sets created
- [x] Unigram baseline implemented
- [x] Unigram validation evaluation completed
- [x] Standard left-to-right n-gram implemented
- [x] Shared baseline evaluation script created
- [ ] Evaluate bigram, trigram, and 4-gram on validation
- [ ] Select the final n-gram order
- [ ] Decide BERT sequence length and windowing strategy
- [ ] Implement neural training dataset and batching
- [ ] Train the LSTM baseline
- [ ] Train the small BERT MLM
- [ ] Evaluate finalized models on the test set
- [ ] Add musical plausibility analysis
- [ ] Add qualitative error analysis
- [ ] Complete the final report

---

## Immediate Next Task

Run the updated left-to-right baseline comparison:

```bash
python3 -m src.evaluate_unigram_ngram
```

This evaluates:

```text
Unigram
Bigram
Trigram
4-gram
```

on the factorized validation set and saves:

```text
results/baseline_comparison/
├── validation_left_to_right_summary.json
└── validation_left_to_right_summary.csv
```

After selecting the best n-gram order, the next major task is preparing sequence windows and the training pipeline for the LSTM and BERT models.

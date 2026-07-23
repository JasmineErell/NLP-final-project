# Fine-Tuning BERT for Masked Note Prediction in Symbolic Music

## Project Overview

This project investigates whether a BERT-style masked language model can learn meaningful musical structure from symbolic music.

Natural language processing models learn patterns and dependencies in sequences of discrete symbols. Although these models are usually applied to words, symbolic music can also be represented as a sequence containing notes, rests, durations, chords, bar lines, and metadata. In this project, we treat music as a language and train a small BERT model to recover masked musical tokens using both their left and right context.

The main research question is:

> Can a BERT-style model learn musical structure well enough to predict masked musical units better than simpler frequency-based and local-context models?

The project uses the Nottingham folk music dataset in ABC notation.

---

## Students

- Jasmine Erell — ID: 322435090
- Yuval Nadam — ID: 315231662

Submitted as a final project for the NLP course at Reichman University.

---

## Research Questions

The project focuses on the following questions:

1. Does BERT outperform a unigram baseline?
2. Does BERT outperform an n-gram model?
3. Does bidirectional context improve masked musical-token prediction?
4. How does the choice of musical representation affect performance?
5. When BERT predicts the wrong token, is the prediction still musically plausible?
6. Does BERT learn broader musical dependencies rather than only memorizing frequent local patterns?

---

## Dataset

We use the Nottingham folk music database in ABC notation.

ABC notation is suitable for this project because it is:

- Text-based
- Compact
- Human-readable
- Easy to process using NLP tools
- Rich enough to represent pitch, duration, rhythm, rests, chords, key, and time signature

Each song is processed independently and converted into a sequence of discrete musical tokens.

### Example ABC sequence

```text
"G"G2A2 B2c2 | "G"dd2d d2d2 | "Em"dd2d d2d2 | "D7"dd2d d2d2
```

---

## Musical Representations

The project begins with a simple note-duration representation and later compares it with a chord-melody segment representation.

### 1. Note-Duration Tokens

Each note or rest is represented together with its duration.

```text
[KEY_G] [TIME_4/4] G_1/4 A_1/4 B_1/2 REST_1/4
```

This representation is used for the initial experiments because it produces a manageable vocabulary and is relatively easy to parse and verify.

### 2. Chord-Melody Segment Tokens

Each chord defines the beginning of a token. The token includes the chord and all melody events that follow it until the next chord.

```text
CHORD_G__MELODY_G2_A2_B2_c2
CHORD_G__MELODY_dd2d_d2d2
CHORD_Em__MELODY_dd2d_d2d2
CHORD_D7__MELODY_dd2d_d2d2
```

This representation combines harmonic and melodic information, but it may create a much larger and sparser vocabulary.

---

## Project Pipeline

```text
Raw ABC files
      ↓
Dataset inspection
      ↓
ABC parsing and normalization
      ↓
Musical tokenization
      ↓
Vocabulary construction
      ↓
Song-level train/validation/test split
      ↓
Masked-example generation
      ↓
Unigram baseline
      ↓
N-gram baselines
      ↓
Small BERT masked language model
      ↓
Quantitative evaluation
      ↓
Musical plausibility and error analysis
```

---

## Repository Structure

```text
music-bert-project/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── src/
│   ├── inspect_data.py
│   ├── preprocess_abc.py
│   ├── tokenizer.py
│   ├── create_splits.py
│   ├── create_masks.py
│   ├── unigram_baseline.py
│   ├── ngram_baseline.py
│   ├── bert_dataset.py
│   ├── train_bert.py
│   ├── evaluate.py
│   └── musical_analysis.py
│
├── configs/
│   └── bert_small.json
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_tokenization_analysis.ipynb
│   └── 03_results_analysis.ipynb
│
└── results/
    ├── tables/
    ├── figures/
    ├── predictions/
    └── checkpoints/
```

---

## Milestones

### Milestone 1: Repository Setup

Goals:

- Create the repository structure
- Add the dataset to `data/raw`
- Add a `requirements.txt`
- Define the research questions
- Set a fixed random seed

Deliverable:

- A reproducible repository with a clear structure

### Milestone 2: Dataset Inspection

Goals:

- Read and print sample ABC songs
- Identify metadata fields
- Identify the musical-body format
- Count successfully parsed and rejected songs
- Measure song-length statistics
- Check how often chord annotations appear

Deliverable:

- A dataset-statistics table
- Several raw and parsed examples

### Milestone 3: ABC Preprocessing

Goals:

- Parse notes
- Parse rests
- Parse durations
- Parse accidentals and octaves
- Extract key and time signature
- Handle malformed songs
- Log skipped files

Deliverable:

- One normalized token sequence per song

### Milestone 4: Tokenization

Goals:

- Implement note-duration tokens
- Implement chord-melody segment tokens
- Build the vocabulary using training data only
- Add special tokens

Special tokens may include:

```text
[PAD]
[UNK]
[MASK]
[CLS]
[SEP]
```

Deliverable:

- Tokenized songs
- Vocabulary files
- Vocabulary-size and frequency statistics

### Milestone 5: Dataset Split

Goals:

- Remove exact duplicates
- Split complete songs into training, validation, and test sets
- Prevent segments from the same song from appearing in different splits

Recommended split:

```text
80% training
10% validation
10% test
```

Use a fixed seed:

```python
RANDOM_SEED = 42
```

Deliverable:

- `train_ids.txt`
- `validation_ids.txt`
- `test_ids.txt`

### Milestone 6: Evaluation Infrastructure

Goals:

- Create fixed masked examples for validation and testing
- Use the same examples for all models
- Implement accuracy, Top-k accuracy, cross-entropy, and perplexity

Deliverable:

- A reusable evaluation pipeline

### Milestone 7: Unigram Baseline

The unigram model ignores context and predicts tokens according to their frequency in the training set.

Deliverable:

- Accuracy
- Top-3 accuracy
- Top-5 accuracy
- Cross-entropy
- Perplexity

### Milestone 8: N-Gram Baselines

Models:

- Bigram
- Trigram
- Optional five-gram
- Optional bidirectional n-gram

The forward n-gram uses only left context. The bidirectional version combines evidence from the left and right context.

Deliverable:

- Comparison of local-context baselines

### Milestone 9: Small BERT Model

The main model is a small BERT-style masked language model trained using a custom musical vocabulary.

Initial configuration:

```text
Maximum sequence length: 128
Hidden size: 256
Transformer layers: 4
Attention heads: 4
Feed-forward size: 1024
Dropout: 0.1
Mask probability: 0.15
```

A smaller debugging configuration may be used first:

```text
Hidden size: 128
Transformer layers: 2
Attention heads: 4
```

Deliverable:

- Best validation checkpoint
- Training and validation loss curves
- Test-set predictions

### Milestone 10: Musical Plausibility Analysis

Goals:

- Check whether predicted notes are in the song's key
- Check whether predicted notes are chord tones
- Compare predicted and target durations
- Measure pitch distance
- Inspect rhythmically valid substitutions
- Analyze Top-5 predictions

Deliverable:

- Quantitative plausibility metrics
- Qualitative examples of correct and incorrect predictions

---

## Baselines and Models

### Unigram Baseline

Predicts the most frequent training token for every masked position.

Purpose:

- Measures how much of the task can be solved using token frequency alone

### N-Gram Model

Predicts the masked token using a limited local context.

Purpose:

- Tests whether short musical patterns are sufficient
- Provides a classical language-model baseline

### Small BERT Model

Uses bidirectional self-attention to predict masked musical tokens.

Purpose:

- Tests whether broader left and right context improves prediction
- Tests whether Transformer-based NLP methods transfer to symbolic music

### Optional LSTM Baseline

An LSTM may be added if time permits.

Purpose:

- Provides a neural sequential baseline
- Compares recurrent and Transformer-based modeling

---

## Masking Procedure

During training, approximately 15% of eligible musical tokens are selected.

A standard BERT-style masking strategy may be used:

- 80% are replaced with `[MASK]`
- 10% are replaced with a random token
- 10% remain unchanged

Metadata tokens such as key and time signature are not masked in the initial experiments.

Validation and test masking must be fixed using a random seed so that all models are evaluated on exactly the same masked positions.

---

## Experimental Plan

### Required Experiments

| ID | Representation | Model | Purpose |
|---|---|---|---|
| E1 | Note-duration | Unigram | Frequency-only baseline |
| E2 | Note-duration | Forward n-gram | Local left-context baseline |
| E3 | Note-duration | Bidirectional n-gram | Local two-sided baseline |
| E4 | Note-duration | Small BERT | Main model |
| E5 | Chord-melody | Unigram | Representation difficulty |
| E6 | Chord-melody | N-gram | Local-context comparison |
| E7 | Chord-melody | Small BERT | Tokenization comparison |

Experiments E1-E4 form the minimum complete project.

Experiments E5-E7 should be performed after the note-duration pipeline is stable.

### Optional Experiments

Possible additional experiments include:

- With and without key metadata
- With and without time-signature metadata
- Notes only versus notes with durations
- Different sequence lengths
- Different masking percentages
- Full bidirectional context versus left context only
- Performance on frequent versus rare tokens
- Original songs versus transposed songs

---

## Evaluation Metrics

### Masked-Token Accuracy

Percentage of masked positions where the exact original token is predicted.

### Top-3 Accuracy

Percentage of examples where the correct token appears among the three most probable predictions.

### Top-5 Accuracy

Percentage of examples where the correct token appears among the five most probable predictions.

### Cross-Entropy Loss

Measures the negative log-probability assigned to the correct masked token.

### Masked-Token Perplexity

Calculated as:

```text
Perplexity = exp(average cross-entropy)
```

This value is reported specifically for masked-token prediction.

### Musical Plausibility Metrics

Possible measures include:

- In-key prediction rate
- Chord-tone prediction rate
- Duration match rate
- Mean pitch distance
- Rhythmic validity
- Melody overlap
- Edit distance between predicted and target segments

---

## Main Results Table

The final report will include a table similar to the following:

| Model | Representation | Accuracy | Top-3 | Top-5 | Loss | Perplexity |
|---|---|---:|---:|---:|---:|---:|
| Unigram | Note-duration | | | | | |
| Forward n-gram | Note-duration | | | | | |
| Bidirectional n-gram | Note-duration | | | | | |
| BERT | Note-duration | | | | | |
| Unigram | Chord-melody | | | | | |
| N-gram | Chord-melody | | | | | |
| BERT | Chord-melody | | | | | |

---

## Additional Analysis

### Token-Frequency Analysis

Performance will be measured separately for:

```text
Frequent tokens: more than 100 training occurrences
Medium-frequency tokens: 10-100 training occurrences
Rare tokens: fewer than 10 training occurrences
```

### Error Analysis

For selected test examples, the following information will be saved:

- Song ID
- Masked position
- Musical context
- Correct token
- Top-5 unigram predictions
- Top-5 n-gram predictions
- Top-5 BERT predictions
- Prediction probabilities
- Musical plausibility notes

### Suggested Figures

- Dataset song-length distribution
- Vocabulary-frequency distribution
- Training and validation loss
- Accuracy by token frequency
- Top-1, Top-3, and Top-5 comparison
- Note-duration versus chord-melody vocabulary size
- Model comparison bar chart

---

## Reproducibility

All experiments should use fixed random seeds.

Every training run should record:

```text
run_id
representation
vocabulary_size
maximum_sequence_length
hidden_size
number_of_layers
number_of_heads
learning_rate
batch_size
number_of_epochs
training_time
best_validation_loss
random_seed
checkpoint_path
```

The test set must not be used for:

- Hyperparameter tuning
- Model selection
- Early stopping

The best model is selected according to validation loss and evaluated once on the test set.

---

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Possible dependencies include:

```text
torch
transformers
datasets
numpy
pandas
scikit-learn
matplotlib
tqdm
music21
```

The final dependency list will be updated as the project develops.

---

## Suggested Execution Order

```bash
python src/inspect_data.py
python src/preprocess_abc.py
python src/create_splits.py
python src/create_masks.py
python src/unigram_baseline.py
python src/ngram_baseline.py
python src/train_bert.py
python src/evaluate.py
python src/musical_analysis.py
```

Command-line arguments and configuration files will be added during implementation.

---

## First Practical Checkpoint

Before training any model, the following pipeline must work correctly:

```text
Raw ABC song
      ↓
Parsed metadata
      ↓
Normalized musical events
      ↓
Token sequence
      ↓
Manual verification against the original song
```

The first goal is to correctly convert and manually verify at least 10 songs.

Only after this step is reliable should preprocessing be applied to the full dataset.

---

## Expected Challenges

Potential technical challenges include:

- Parsing irregular ABC notation
- Handling inherited note durations
- Handling accidentals and octaves
- Detecting malformed tunes
- Managing a large chord-melody vocabulary
- Preventing train-test leakage
- Evaluating predictions with multiple musically valid alternatives
- Comparing models with different context assumptions
- Training BERT on a relatively small dataset

These challenges are part of the research process and will be documented in the final report.

---

## Final Report Structure

### 1. Introduction

- Background
- Motivation
- Research questions
- Symbolic music as language

### 1.1 Related Work

- Symbolic music modeling
- Transformers for music
- Masked language modeling
- Music tokenization

### 2. Methodology

- Dataset
- Preprocessing
- Tokenization
- Data splits
- Masking
- Baselines
- BERT architecture
- Evaluation metrics
- Technical environment

### 3. Experimental Results

- Dataset statistics
- Baseline results
- BERT results
- Tokenization comparison
- Context experiment
- Token-frequency analysis
- Musical plausibility
- Error analysis

### 4. Discussion

- Main findings
- Interpretation
- Limitations
- Future work

### 5. Code

A link to the final GitHub repository or Colab notebooks will be provided.

---

## Status

- [ ] Repository setup
- [ ] Dataset downloaded
- [ ] Dataset inspection completed
- [ ] ABC parser implemented
- [ ] Note-duration tokenizer implemented
- [ ] Chord-melody tokenizer implemented
- [ ] Train/validation/test split created
- [ ] Fixed masked test set created
- [ ] Unigram baseline completed
- [ ] N-gram baseline completed
- [ ] BERT model trained
- [ ] Quantitative evaluation completed
- [ ] Musical plausibility analysis completed
- [ ] Final report completed

---

## License and Dataset Attribution

Dataset attribution and licensing information will be added according to the original Nottingham dataset source.


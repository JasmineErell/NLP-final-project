# Project Progress

## What We Completed

- Created the project repository and added the cleaned Nottingham ABC dataset.
- Inspected the ABC format and identified the main fields: `X`, `T`, `R`, `M`, `L`, `K`, and `P`.
- Implemented song splitting using the `X:` field.
- Implemented metadata extraction and musical-body extraction.
- Implemented chord-melody segment extraction.
- Added structured segment storage containing the chord, raw melody, normalized melody, and atomic token.
- Implemented two tokenization strategies: atomic and factorized.
- Updated preprocessing to save both representations.
- Ran a pilot on 10 songs with zero parsing errors.
- Generated corpus statistics, token frequencies, examples, and processed JSONL files.

## Pilot Results

| Metric | Atomic | Factorized |
|---|---:|---:|
| Total tokens | 265 | 2,222 |
| Unique tokens | 211 | 120 |
| Singleton rate | 81.99% | 20.00% |
| Average sequence length | 26.5 | 222.2 |
| Maximum sequence length | 47 | 366 |

## Tokenization Conflict

### Version A — Atomic

Example:

```text
CHORD_G__MELODY_G2A2 B2c2 |
```

Advantages:

- Closely matches the approved proposal.
- Keeps the complete chord-melody unit together.
- Produces short sequences.

Problems:

- Creates a very sparse vocabulary.
- Most tokens appear only once.
- Validation and test OOV rates may be high.
- The model may memorize exact segments instead of learning reusable patterns.

### Version B — Factorized

Example:

```text
[SEGMENT_START]
[CHORD_G]
[EVENT_G2]
[EVENT_A2]
[EVENT_B2]
[EVENT_c2]
[BAR]
[SEGMENT_END]
```

Advantages:

- Reuses notes, durations, chords, and structural tokens.
- Produces a much smaller and less sparse vocabulary.
- Should generalize better.
- Makes musical plausibility analysis easier.

Problems:

- Produces much longer sequences.
- May require sequence windows or a larger BERT maximum length.
- Predicts individual musical events rather than full chord-melody segments.

## Current Decision

Keep both representations:

- Use Version A as a direct implementation of the approved proposal.
- Use Version B as the main alternative designed to reduce sparsity.
- Compare their vocabulary sizes, OOV rates, and model performance.

## Next Steps

1. Manually inspect `data/processed/tokenization_examples.txt`.
2. Confirm that chords, notes, durations, rests, accidentals, bars, and repeats are preserved correctly.
3. Run preprocessing on the full Nottingham dataset.
4. Review full-corpus vocabulary and sequence-length statistics.
5. Update `create_splits.py` to create one song-level 80/10/10 split for both representations.
6. Build separate training vocabularies for atomic and factorized tokens.
7. Calculate validation and test OOV rates for both representations.
8. Update `create_masked_examples.py` to create fixed evaluation masks for both versions.
9. Implement the unigram baseline.
10. Implement the n-gram baseline.
11. Decide how to handle long factorized sequences.
12. Train and evaluate the BERT model.
13. Add musical plausibility and error analysis.

## Immediate Next Task

```text
Validate the 10 pilot examples, then run preprocessing on the full dataset.
```

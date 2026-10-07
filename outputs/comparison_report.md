# Contract Clause Classification Report

Each method decides, for each test contract and clause type, whether the clause appears anywhere in the contract.  Labels come from CUAD answer spans.

## Data

- Test contracts: 56
- Clause types: Governing Law, Anti-Assignment, Cap On Liability, Uncapped Liability, Audit Rights, Termination For Convenience, Change Of Control, Exclusivity, Non-Compete, Insurance, License Grant, Warranty Duration

## Models and measured cost

|  | Fine-tuned roberta-base | Zero-shot claude-opus-5-5, multi-label |
|---|---|---|
| Model | roberta-base | anthropic/claude-opus-5-5 |
| Contracts | 56 | 56 |
| LLM calls | none | 130 |
| Calls answered from the cache | none | 18 |
| Failed calls | none | 0 |
| Average latency per contract | 583 ms | 7,052 ms |
| Fastest and slowest contract | 11 ms and 2,271 ms | 1,329 ms and 35,225 ms |
| Cost per contract | not priced | $0.0598 |
| Total cost | not priced | $3.3475 |

## Aggregate metrics

|  | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Fine-tuned roberta-base | 0.7683 | 0.7877 | 0.7361 | 0.8750 |
| Zero-shot claude-opus-5-5, multi-label | 0.7571 | 0.9342 | 0.8175 | 0.8914 |

## Per-clause comparison

| Clause type | Method | Precision | Recall | F1 | Accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| Governing Law | Fine-tuned roberta-base | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 45 | 0 | 11 | 0 |
| Governing Law | Zero-shot claude-opus-5-5, multi-label | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 45 | 0 | 11 | 0 |
| Anti-Assignment | Fine-tuned roberta-base | 0.8947 | 0.9444 | 0.9189 | 0.8929 | 34 | 4 | 16 | 2 |
| Anti-Assignment | Zero-shot claude-opus-5-5, multi-label | 0.9459 | 0.9722 | 0.9589 | 0.9464 | 35 | 2 | 18 | 1 |
| Cap On Liability | Fine-tuned roberta-base | 0.8462 | 0.9167 | 0.8800 | 0.8929 | 22 | 4 | 28 | 2 |
| Cap On Liability | Zero-shot claude-opus-5-5, multi-label | 0.8889 | 0.6667 | 0.7619 | 0.8214 | 16 | 2 | 30 | 8 |
| Uncapped Liability | Fine-tuned roberta-base | 0.5385 | 0.7778 | 0.6364 | 0.8571 | 7 | 6 | 41 | 2 |
| Uncapped Liability | Zero-shot claude-opus-5-5, multi-label | 0.3913 | 1.0000 | 0.5625 | 0.7500 | 9 | 14 | 33 | 0 |
| Audit Rights | Fine-tuned roberta-base | 0.7500 | 0.8333 | 0.7895 | 0.8571 | 15 | 5 | 33 | 3 |
| Audit Rights | Zero-shot claude-opus-5-5, multi-label | 0.8500 | 0.9444 | 0.8947 | 0.9286 | 17 | 3 | 35 | 1 |
| Termination For Convenience | Fine-tuned roberta-base | 0.5152 | 0.8095 | 0.6296 | 0.6429 | 17 | 16 | 19 | 4 |
| Termination For Convenience | Zero-shot claude-opus-5-5, multi-label | 0.7407 | 0.9524 | 0.8333 | 0.8571 | 20 | 7 | 28 | 1 |
| Change Of Control | Fine-tuned roberta-base | 1.0000 | 0.1111 | 0.2000 | 0.8571 | 1 | 0 | 47 | 8 |
| Change Of Control | Zero-shot claude-opus-5-5, multi-label | 0.4091 | 1.0000 | 0.5806 | 0.7679 | 9 | 13 | 34 | 0 |
| Exclusivity | Fine-tuned roberta-base | 0.5909 | 0.8125 | 0.6842 | 0.7857 | 13 | 9 | 31 | 3 |
| Exclusivity | Zero-shot claude-opus-5-5, multi-label | 0.6667 | 0.8750 | 0.7568 | 0.8393 | 14 | 7 | 33 | 2 |
| Non-Compete | Fine-tuned roberta-base | 0.7143 | 0.5000 | 0.5882 | 0.8750 | 5 | 2 | 44 | 5 |
| Non-Compete | Zero-shot claude-opus-5-5, multi-label | 0.7143 | 1.0000 | 0.8333 | 0.9286 | 10 | 4 | 42 | 0 |
| Insurance | Fine-tuned roberta-base | 1.0000 | 0.9474 | 0.9730 | 0.9821 | 18 | 0 | 37 | 1 |
| Insurance | Zero-shot claude-opus-5-5, multi-label | 0.9500 | 1.0000 | 0.9744 | 0.9821 | 19 | 1 | 36 | 0 |
| License Grant | Fine-tuned roberta-base | 0.9259 | 1.0000 | 0.9615 | 0.9643 | 25 | 2 | 29 | 0 |
| License Grant | Zero-shot claude-opus-5-5, multi-label | 0.8621 | 1.0000 | 0.9259 | 0.9286 | 25 | 4 | 27 | 0 |
| Warranty Duration | Fine-tuned roberta-base | 0.4444 | 0.8000 | 0.5714 | 0.8929 | 4 | 5 | 46 | 1 |
| Warranty Duration | Zero-shot claude-opus-5-5, multi-label | 0.6667 | 0.8000 | 0.7273 | 0.9464 | 4 | 2 | 49 | 1 |

## Configuration

- Fine-tuned roberta-base: threshold 0.5, max_length 512, window_stride 128, epochs 3, learning_rate 2e-05, batch_size 8, created_at 2026-10-07T01:59:50+00:00, training_seconds 2119.3, train_windows 6972, validation_windows 1054
- Zero-shot claude-opus-5-5, multi-label: mode multi, prompt_version 2, chunk_chars 24000, chunk_overlap 1000, concurrency 4, provider anthropic, effort low, max_tokens 2048, created_at 2026-10-06T12:28:03+00:00

## Notes

- Fine-tuned roberta-base is not priced.  Its latency is measured compute time, and --cost-per-hour or FT_COST_PER_HOUR turns that time into a cost.
- Zero-shot claude-opus-5-5, multi-label: 18 of 130 calls were answered from the response cache.  Their latency and cost are those measured when they were made.

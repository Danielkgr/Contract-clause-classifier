# Contract Clause Classification Report

Each method decides, for each test contract and clause type, whether the clause appears anywhere in the contract.  Labels come from CUAD answer spans.

## Data

- Test contracts: 56
- Clause types: Governing Law, Anti-Assignment, Cap On Liability, Uncapped Liability, Audit Rights, Termination For Convenience, Change Of Control, Exclusivity, Non-Compete, Insurance, License Grant, Warranty Duration

## Models and measured cost

|  | Zero-shot claude-opus-5-5, multi-label |
|---|---|
| Model | anthropic/claude-opus-5-5 |
| Contracts | 56 |
| LLM calls | 130 |
| Calls answered from the cache | 18 |
| Failed calls | 0 |
| Average latency per contract | 7,052 ms |
| Fastest and slowest contract | 1,329 ms and 35,225 ms |
| Cost per contract | $0.0598 |
| Total cost | $3.3475 |

## Aggregate metrics

|  | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Zero-shot claude-opus-5-5, multi-label | 0.7571 | 0.9342 | 0.8175 | 0.8914 |

## Per-clause comparison

| Clause type | Method | Precision | Recall | F1 | Accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| Governing Law | Zero-shot claude-opus-5-5, multi-label | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 45 | 0 | 11 | 0 |
| Anti-Assignment | Zero-shot claude-opus-5-5, multi-label | 0.9459 | 0.9722 | 0.9589 | 0.9464 | 35 | 2 | 18 | 1 |
| Cap On Liability | Zero-shot claude-opus-5-5, multi-label | 0.8889 | 0.6667 | 0.7619 | 0.8214 | 16 | 2 | 30 | 8 |
| Uncapped Liability | Zero-shot claude-opus-5-5, multi-label | 0.3913 | 1.0000 | 0.5625 | 0.7500 | 9 | 14 | 33 | 0 |
| Audit Rights | Zero-shot claude-opus-5-5, multi-label | 0.8500 | 0.9444 | 0.8947 | 0.9286 | 17 | 3 | 35 | 1 |
| Termination For Convenience | Zero-shot claude-opus-5-5, multi-label | 0.7407 | 0.9524 | 0.8333 | 0.8571 | 20 | 7 | 28 | 1 |
| Change Of Control | Zero-shot claude-opus-5-5, multi-label | 0.4091 | 1.0000 | 0.5806 | 0.7679 | 9 | 13 | 34 | 0 |
| Exclusivity | Zero-shot claude-opus-5-5, multi-label | 0.6667 | 0.8750 | 0.7568 | 0.8393 | 14 | 7 | 33 | 2 |
| Non-Compete | Zero-shot claude-opus-5-5, multi-label | 0.7143 | 1.0000 | 0.8333 | 0.9286 | 10 | 4 | 42 | 0 |
| Insurance | Zero-shot claude-opus-5-5, multi-label | 0.9500 | 1.0000 | 0.9744 | 0.9821 | 19 | 1 | 36 | 0 |
| License Grant | Zero-shot claude-opus-5-5, multi-label | 0.8621 | 1.0000 | 0.9259 | 0.9286 | 25 | 4 | 27 | 0 |
| Warranty Duration | Zero-shot claude-opus-5-5, multi-label | 0.6667 | 0.8000 | 0.7273 | 0.9464 | 4 | 2 | 49 | 1 |

## Configuration

- Zero-shot claude-opus-5-5, multi-label: mode multi, prompt_version 2, chunk_chars 24000, chunk_overlap 1000, concurrency 4, provider anthropic, effort low, max_tokens 2048, created_at 2026-10-06T12:28:03+00:00

## Notes

- Only one arm has results, so there is nothing to compare it with yet.
- Zero-shot claude-opus-5-5, multi-label: 18 of 130 calls were answered from the response cache.  Their latency and cost are those measured when they were made.

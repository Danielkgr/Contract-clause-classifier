# Contract Clause Classifier Comparison

56 test contracts, 12 clause types.

## Models

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

## Notes

- Only one arm has results, so there is nothing to compare it with yet.
- Zero-shot claude-opus-5-5, multi-label: 18 of 130 calls were answered from the response cache.  Their latency and cost are those measured when they were made.

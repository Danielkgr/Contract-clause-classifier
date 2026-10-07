# Contract Clause Classifier Comparison

56 test contracts, 12 clause types.

## Models

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

## Notes

- Fine-tuned roberta-base is not priced.  Its latency is measured compute time, and --cost-per-hour or FT_COST_PER_HOUR turns that time into a cost.
- Zero-shot claude-opus-5-5, multi-label: 18 of 130 calls were answered from the response cache.  Their latency and cost are those measured when they were made.

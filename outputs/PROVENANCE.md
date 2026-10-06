# Provenance: zero-shot run on Claude

`zero_shot-claude-opus-5-5-multi.json` in this folder comes from one zero-shot run against the live Anthropic API.  The report files beside it were written from that file alone.

| Item | Value |
|---|---|
| Commands | `python compare_classifiers.py zero-shot --max-contracts 5`, then `python compare_classifiers.py zero-shot` |
| Run at | 2026-10-06, between 12:26 and 12:28:03 UTC |
| Code revision | `ee5ea53`, "Merge pull request #2 from Danielkgr/claude-arm-and-cli-cleanup", in a fresh clone |
| Data | `CUAD_v1/CUAD_v1.json` from the Hugging Face dataset `theatticusproject/cuad` at revision `a3c393f5d103fd0c516374e4fdff676c8176dcb1` |
| Data SHA-256 | `ed0b77d85bdf4014d7495800e8e4a70565b48ee6f8a2e5dca9cf8655dbf10eae` |
| Contracts | The 56 contracts of the test split, 12 clause types |
| Model | `claude-opus-5-5`, `low` effort, `max_tokens` of 2,048, the model's default temperature, no server-side fallbacks |
| Prompt | Multi-label mode, prompt version 2, chunks of 24,000 characters overlapping by 1,000, 4 contracts at a time |
| SDK and Python | `anthropic` 1.11.0, Python 3.14.4 |

The first command asked about the first five test contracts and saved its 18 responses in the response cache.  The second asked about all 56 and read those 18 from the cache, so no call was paid for twice.  A cached call keeps the latency and cost measured when it was made, so the totals below cover every call exactly once.  The second command overwrote the first command's output file, and both commands' console output is in `eval-logs/cuad-zero-shot.log`.

## Usage and cost

| Measure | Value |
|---|---|
| Calls | 130, all answered, 0 failed, 0 retried |
| Uncached input tokens | 809,822 |
| Cache read tokens | 99,792 |
| Cache write tokens | 3,168 |
| Output tokens | 3,620 |
| Cost | $3.3475, of which the first five contracts were $0.5460 |
| Cost per contract | $0.0598 |
| Latency per contract | Average 7,052 ms, fastest 1,329 ms, slowest 35,225 ms |

The cost is computed from the usage figures each response reported, at the list prices for Claude Opus 5.5 in `utils/pricing.py`.  The invoice is the authority.  The estimate before the run was $3.15.

## The report files

`comparison_report.md`, `summary.md`, `comparison_metrics.csv`, and `comparison_plot.png` were written on 7 October 2026 by `python compare_classifiers.py compare`, at the same revision, from the saved zero-shot file.  That step makes no API call.  The fine-tuned arm has not been run, so each file covers the zero-shot arm alone, and the aggregate precision, recall, F1, and accuracy are macro averages over the 12 clause types.

This was the first and only scored run on Claude.  No prompt, clause definition, or setting was changed after it.

To reproduce it, check out revision `ee5ea53`, install `requirements-dev.txt`, set `ANTHROPIC_API_KEY`, delete `.cache/llm/` so no earlier answer is reused, and run the two commands above.  Passing `--no-cache` to both would pay for the first five contracts twice.  The model's answers can differ between runs, and so can the scores.

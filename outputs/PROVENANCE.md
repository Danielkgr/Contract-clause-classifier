# Provenance: the zero-shot and fine-tuned runs

`zero_shot-claude-opus-5-5-multi.json` in this folder comes from one zero-shot run against the live Anthropic API, and `fine_tuned-roberta-base.json` from one fine-tuning run on a local GPU.  The report files beside them were written from those two files alone.

## The zero-shot run

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

### Usage and cost

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

## The fine-tuned run

| Item | Value |
|---|---|
| Command | `python compare_classifiers.py fine-tune --cuad <path to CUAD_v1.json>` |
| Run at | 2026-10-07, 12:23 to 12:59 AEDT (01:23 to 01:59 UTC) |
| Code revision | `6f91c07`, a fresh clone.  It changes no code or requirement since `ee5ea53`, the zero-shot run's revision, only documentation and outputs. |
| Data | The same CUAD file and hash as the zero-shot run.  401 training contracts, 53 validation contracts, 56 test contracts. |
| Base model | `roberta-base` from Hugging Face, revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b` |
| Settings | The defaults in `config.py`.  3 epochs, batch size 8, learning rate 2e-5, weight decay 0.01, 500 warm-up steps, 512-token windows with a stride of 128, one all-negative window kept per window with a clause, and a threshold of 0.5.  Nothing was tuned. |
| Hardware | AMD Radeon RX 7900 XTX, 24 GB.  Its memory was clear when training started. |
| Software | PyTorch 2.14.1 built for ROCm 7.14, transformers 5.19.0, accelerate 1.15.0, scikit-learn 1.9.1, Python 3.12 |

| Outcome | Value |
|---|---|
| Training windows | 6,972, half with a clause, from the 401 training contracts |
| Validation windows | 1,054 |
| Training time | 2,119 seconds, 2,616 steps |
| Model kept | Checkpoint 2,000, with a validation loss of 0.0689.  The Trainer marked step 2,500 as best, at 0.0664, but `config.py` saves a checkpoint every 1,000 steps and evaluates every 500, so step 2,500 was never saved and `load_best_model_at_end` reloaded the last saved checkpoint before it.  The scored model's weights match checkpoint 2,000 byte for byte. |
| Test latency | Average 583 ms per contract, fastest 11 ms, slowest 2,271 ms, to tokenise and score every window with the model already loaded |
| Cost | Not priced.  No per-call fee applies, and `FT_COST_PER_HOUR` was not set. |

Setting `SAVE_STEPS` equal to `EVAL_STEPS` would let the Trainer keep the step it marks as best.  That was not changed for this run, so it ran on the documented defaults.  The trained model is not committed.  The console output of the run is in `eval-logs/cuad-fine-tune.log`, unedited, including the progress bars and the lines of the wrapper script that started it.

## The report files

`comparison_report.md`, `summary.md`, `comparison_metrics.csv`, and `comparison_plot.png` were written on 7 October 2026 by `python compare_classifiers.py compare`, from the two saved arm files.  That step makes no API call and uses no GPU.  Each file scores both arms on the same 56 contracts, and the aggregate precision, recall, F1, and accuracy are macro averages over the 12 clause types.

Each arm was run once and scored once.  No prompt, clause definition, or setting was changed after either result was seen.

To reproduce the zero-shot run, check out revision `ee5ea53`, install `requirements-dev.txt`, set `ANTHROPIC_API_KEY`, delete `.cache/llm/` so no earlier answer is reused, and run the two zero-shot commands above.  Passing `--no-cache` to both would pay for the first five contracts twice.  To reproduce the fine-tuned run, install `requirements.txt` with a PyTorch build for your GPU and run the fine-tune command.  The model's answers can differ between runs, a different GPU or seed can change the fine-tuned model, and so the scores can change too.

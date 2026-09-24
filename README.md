<div align="center">

# Contract Clause Classifier

### A zero-shot LLM against a fine-tuned transformer on contract clauses, compared on accuracy, cost, and latency

![status prototype](https://img.shields.io/badge/status-prototype-9a6700?style=for-the-badge) ![no results yet](https://img.shields.io/badge/results-none_yet-9a6700?style=for-the-badge) ![12 clause types](https://img.shields.io/badge/clause_types-12-0969da?style=for-the-badge) ![28 tests](https://img.shields.io/badge/tests-28-0969da?style=for-the-badge) ![MIT licence](https://img.shields.io/badge/licence-MIT-57606a?style=for-the-badge)

</div>

<br>

> Prompting a general model and fine-tuning a small one are both reasonable ways to find a clause in a contract.  The choice turns on cost, latency, and where the contract text is allowed to go, as much as on accuracy.  This harness runs both approaches over the same [CUAD](https://huggingface.co/datasets/theatticusproject/cuad) contracts and reports precision, recall, F1, accuracy, cost per document, and latency for each clause type.

<br>

## What it is

An evaluation harness.  It fine-tunes a RoBERTa classifier, prompts an LLM zero-shot, and asks both the same question of every test contract, whether each clause type appears anywhere in it.  Both are scored against CUAD's labels, and a report compares accuracy, latency, and cost.  The report is framed as a build-or-buy decision for a team choosing between the two.

> [!IMPORTANT]
> It is not a deployed classifier.  It does not serve predictions, store contracts, or ship a production model.

It is a working prototype.  The whole pipeline has run end to end on CUAD with a small test model and a stand-in for the LLM, which checks the plumbing and says nothing about accuracy.

<br>

## Results

No comparison run has been executed for this repository, so there are no figures here yet.  A full run fine-tunes RoBERTa on the 401 training contracts and sends the zero-shot arm to a paid LLM API.  When a run is complete, its artefacts (`comparison_metrics.csv`, `comparison_report.md`, and the plot) will be committed next to this README.

Until then, the guidance in [Choosing between the two](#choosing-between-the-two) is qualitative.  It rests on the cost and latency profile of each approach rather than on anything this code has measured.

<br>

## How it works

CUAD contracts are long.  The median runs to 33,000 characters and the longest to 338,000, and the clauses sit throughout.  Across the 510 contracts there are 2,510 cases of a contract containing one of the 12 default clause types, and in only 5 of them does the clause begin within the first 512 characters.  Both arms therefore read the whole contract.

### Pipeline

| Stage | Code | What happens |
|---|---|---|
| **Load** | `load_cuad_dataset()` | Reads CUAD v1 (`CUAD_v1.json`) from `data/`, or downloads it from [Hugging Face](https://huggingface.co/datasets/theatticusproject/cuad).  CUAD has no splits, so each contract goes to train, validation, or test (401, 53, and 56 contracts) by a stable hash of its title.  If the download fails, it falls back to local CSV, JSON, or Parquet files in `data/`. |
| **Wrap** | `ContractData` | Holds each contract's title, its full text, whether each clause type is present, and the character offsets of every CUAD answer span. |
| **Fine-tune** | `utils/classifier.py` | Trains one multi-label `roberta-base` model on overlapping windows of the training contracts, labelled from the answer spans. |
| **Zero-shot** | `utils/llm_client.py` | Asks the LLM whether a chunk of the contract contains a clause type, answering only YES or NO, through OpenAI directly or any LiteLLM-compatible provider. |
| **Evaluate** | `utils/evaluation.py` | Runs both arms over the test contracts and times each contract.  The CLI and the notebook share this code. |
| **Report** | `compare_classifiers.py` | Writes the metrics table, the plot, a summary, and the full report to `outputs/`. |

### The fine-tuned arm

| Aspect | Detail |
|---|---|
| **Model** | One `AutoModelForSequenceClassification` with a sigmoid output for each clause type, so it scores all of them in one pass |
| **Windows** | Each contract is tokenised whole and cut into 512-token windows that overlap by 128 tokens, so every part of it is read |
| **Labels** | A window is positive for a clause type when it overlaps one of that type's CUAD answer spans |
| **Sampling** | Every window with a clause is kept, plus an equal number of windows without one, drawn at random with a fixed seed |
| **Validation** | Windows from the 53 validation contracts, so no contract appears in both training and validation |
| **Prediction** | A contract contains a clause type when any of its windows scores 0.5 or more |

### The zero-shot arm

The client sends the contract in chunks of 24,000 characters that overlap by 1,000, and asks about one clause type at a time.  A clause counts as present at the first chunk that gets YES, and the remaining chunks are skipped for that clause.  If a call fails before a YES, that contract and clause type are left out of the metrics and counted, rather than scored as NO.

### Latency and cost

Both arms are timed per contract.  For the LLM, that is the sum of its calls, and cost is the token count multiplied by the per-million prices in `config.py`, which warns when a model has no listed price.  For the fine-tuned model, it is the time to tokenise and score every window.  The fine-tuned arm has no per-call fee, so its cost is reported as not priced unless `FT_COST_PER_HOUR` is set, in which case the measured time is multiplied by that rate.

### Choosing between the two

These are the trade-offs the report is built to test.  Until a run is published they are expectations, not findings.

| Consideration | Zero-shot LLM | Fine-tuned model |
|---|---|---|
| **Volume** | A few documents a day | Hundreds of documents a day |
| **Setup** | No training and no ML infrastructure | A training run and a GPU |
| **Running cost** | Per-token API fees on every call | A one-off training cost, then no per-call fee |
| **Latency** | An API round trip for every clause and chunk | Local inference over every window of the contract |
| **Data handling** | Contract text goes to the provider | Contract text stays in-house |
| **Best fit** | A prototype or proof of concept | A long-term deployed service |

<br>

## Quick start

```bash
pip install -r requirements.txt
python compare_classifiers.py
```

The first run downloads `CUAD_v1.json` (about 40 MB) from Hugging Face.  To work offline, save that file to `data/CUAD_v1.json` and the loader will use it.

Option 1 in the menu is a quick test that asks the LLM about 5 test contracts, and adds the fine-tuned model only if one has already been saved.  It is the cheapest way to see the pipeline work.  Every setting can be changed from the menu, so a `.env` file is optional.  To keep settings between sessions, copy `.env.example` to `.env` and fill it in.

```bash
pip install -r requirements-dev.txt
pytest
```

The 27 tests cover CUAD parsing, splitting, and answer spans, chunking and window labelling, both evaluation arms with stand-in models, cost units, and the handling of failed calls.  They need only pandas, scikit-learn, and pytest.  One of them checks the real CUAD file when `data/CUAD_v1.json` is present.  A 28th test trains, saves, reloads, and runs the real classifier with a tiny model.  It needs torch and transformers, downloads the model, and runs only with `RUN_SMOKE=1 pytest`.

<br>

## Usage

### Main menu

```text
═══════════════════ Main Menu ════════════════════
  1. Quick test (5 contracts, zero-shot)
  2. Full comparison (train + evaluate)
  3. Train only
  4. Evaluate only (use existing model)
  5. Configure settings
  6. View configuration
  7. Export / Import settings
  8. Exit

Choose an option [1]:
```

| # | Action | What it does |
|:--:|---|---|
| **1** | Quick test | Asks the LLM about 5 test contracts, adding any saved fine-tuned model, and skips training |
| **2** | Full comparison | Downloads the data, trains the fine-tuned model, and evaluates both classifiers |
| **3** | Train only | Trains a new fine-tuned model on the train split, checked against the validation split |
| **4** | Evaluate only | Evaluates the LLM and the saved model in `models/fine_tuned/`, without training |
| **5** | Configure settings | Opens the configuration menu described below |
| **6** | View configuration | Shows every active setting, read only |
| **7** | Export or import | Writes the current configuration to `.env.local`, or shows the environment table |
| **8** | Exit | Leaves the program |

### Configuration menu

| Submenu | What it sets |
|---|---|
| **LLM settings** | Provider (`openai`, `anthropic`, `google`, or custom), model name, API key, base URL, temperature, and max tokens |
| **Training settings** | Model name, epochs, batch size, learning rate, max token length, weight decay, warmup steps, eval steps, and save steps |
| **Clause types** | A checklist of active clause types, with options to add another CUAD category, remove one, or restore the 12 defaults |
| **Output directory** | Where results are saved.  The directory is created if it does not exist. |

### Flags

Passing any flag skips the menu and every prompt.

```bash
# Quick test on 5 test contracts
python compare_classifiers.py --quick-test

# Train, then evaluate on at most N test contracts
python compare_classifiers.py --max-samples 20

# Write results somewhere other than outputs/
python compare_classifiers.py --output ./custom_output
```

<br>

## Reference

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `openai` | LLM backend, such as `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-3.5-turbo` | Model identifier |
| `LLM_API_KEY` | Required | API key for the chosen provider |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature |
| `LLM_MAX_TOKENS` | `500` | Maximum completion tokens |
| `LLM_BASE_URL` | Optional | Custom base URL, for Azure or Ollama for example |
| `LLM_CHUNK_CHARS` | `24000` | Characters of contract per LLM call |
| `LLM_CHUNK_OVERLAP` | `1000` | Characters shared by consecutive chunks |
| `TRAIN_MODEL` | `roberta-base` | Hugging Face model to fine-tune |
| `BATCH_SIZE` | `8` | Training and inference batch size |
| `LR` | `2e-5` | Learning rate |
| `NUM_EPOCHS` | `3` | Training epochs |
| `MAX_LENGTH` | `512` | Window length in tokens, capped at what the model accepts |
| `WEIGHT_DECAY` | `0.01` | AdamW weight decay |
| `WARMUP_STEPS` | `500` | Learning-rate warmup steps |
| `EVAL_STEPS` | `500` | Evaluation frequency during training |
| `SAVE_STEPS` | `1000` | Checkpoint frequency |
| `WINDOW_STRIDE` | `128` | Tokens shared by consecutive windows |
| `NEGATIVE_WINDOW_RATIO` | `1.0` | Windows without a clause kept for each window with one |
| `FT_THRESHOLD` | `0.5` | Score at which a window counts as containing a clause |
| `FT_COST_PER_HOUR` | Unset | USD per hour for the inference machine.  Unset leaves the fine-tuned arm unpriced. |

### Default clause types

The defaults are 12 of CUAD's 41 categories, a mix of common and rarer clauses.  Any other category can be added from the menu, spelled as it is in `CUAD_CATEGORIES` in `config.py`.  A name that is not a CUAD category stops the load with an error that lists the valid ones, rather than labelling every contract as absent.

| Clause type | What it covers |
|---|---|
| **Governing Law** | Which state or country's law governs the contract |
| **Anti-Assignment** | Consent or notice needed before the contract can be assigned |
| **Cap On Liability** | A cap on a party's liability for breach |
| **Uncapped Liability** | Liability left uncapped, for all breaches or a particular kind |
| **Audit Rights** | A right to audit the other party's books, records, or premises |
| **Termination For Convenience** | A right to terminate without cause |
| **Change Of Control** | Rights triggered by a change of control, merger, or asset sale |
| **Exclusivity** | An exclusive dealing commitment with the other party |
| **Non-Compete** | A restriction on competing with the other party |
| **Insurance** | A requirement to maintain insurance |
| **License Grant** | A licence granted by one party to the other |
| **Warranty Duration** | How long a warranty lasts |

### Output artefacts

Every artefact is written to `outputs/`.

| File | Contents |
|---|---|
| `comparison_metrics.csv` | Metrics for each clause type and each method, in long format |
| `comparison_plot.png` | A two-by-two bar chart of precision, recall, F1, and accuracy |
| `summary.md` | A short summary with the key numbers |
| `comparison_report.md` | The full report, with measured latency and cost, per-clause results with confusion counts, and the settings used |

### Metrics module

| Class or function | Purpose |
|---|---|
| `ClassificationMetrics` | Precision, recall, F1, accuracy, and the TP, FP, TN, and FN counts |
| `InferenceStats` | Total, average, minimum, and maximum latency per contract in milliseconds, cost (or none when unpriced), and token counts |
| `aggregate_metrics()` | Mean, minimum, and maximum across clause types |
| `print_comparison_table()` | The comparison table printed to the terminal |

### Stack

| Area | Libraries |
|---|---|
| **CLI** | `rich` |
| **Machine learning** | `torch`, `transformers`, `accelerate` |
| **Data** | `huggingface_hub`, `datasets`, `pandas` |
| **Evaluation** | `scikit-learn`, `numpy` |
| **LLM APIs** | `openai`, `litellm`, `tiktoken` |
| **Charts** | `matplotlib`, `seaborn` |
| **Utilities** | `python-dotenv`, `tqdm`, `requests` |
| **Tests** | `pytest` |

<br>

## Layout

```text
Contract-clause-classifier/
  compare_classifiers.py     Entry point, with the interactive CLI and the pipeline
  config.py                  Configuration as dataclasses, read from the environment
  evaluation.ipynb           Jupyter notebook for interactive exploration
  requirements.txt           Python dependencies
  requirements-dev.txt       Test dependencies (pandas, scikit-learn, and pytest)
  .env.example               Environment variable template (optional)
  utils/
    __init__.py              Public API exports
    llm_client.py            Zero-shot LLM client (OpenAI and LiteLLM)
    data_loader.py           CUAD loading and splitting, from data/ or Hugging Face
    classifier.py            Windowed multi-label transformer (FineTunedClassifier)
    chunking.py              Contract chunks and window labels from answer spans
    evaluation.py            Contract-level evaluation of both arms, shared by the CLI and notebook
    metrics.py               ClassificationMetrics, InferenceStats, aggregation helpers
  tests/                     Loader, chunking, evaluation, cost, and failed-call tests, plus a smoke test
  data/                      Optional local copy of CUAD_v1.json, or fallback CSV, JSON, or Parquet
  models/                    Saved fine-tuned checkpoints
  outputs/                   Results, created on the first run
```

<br>

## Licence

MIT.  See [LICENSE](LICENSE).

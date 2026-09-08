# 📄 Contract Clause Classifier

> A side-by-side comparison of **zero-shot LLM** versus **fine-tuned transformer** models for automated contract clause classification on the [CUAD dataset](https://huggingface.co/datasets/cuad).

Evaluate precision, recall, F1, accuracy, inference cost, and latency across all 12 standard CUAD clause types - and get engineering recommendations on when each approach makes sense.

**What it does not do:** it is an evaluation harness, not a deployed classifier - it does not serve predictions, store contracts, or ship a production model. No benchmark results have been published for this codebase yet (see [Results](#results)).

**Maturity:** working prototype. The full pipeline (load → train → evaluate → report) is implemented; a comparison run has not yet been executed for this repository, so no measured numbers are published.

---

## Results

No benchmark results are published in this repository yet. Producing them requires a full run - fine-tuning a single RoBERTa model on stacked per-clause labels plus zero-shot LLM evaluation against a paid LLM API - and no such run has been executed for this codebase. When a real run is completed, its artifacts (`comparison_metrics.csv`, `comparison_report.md`, and the plot) will be committed alongside this README. In the meantime no figures are presented: the recommendations in [Engineering Recommendations](#-engineering-recommendations) below are qualitative, based on the cost and latency profiles of the two approaches rather than on measured results from this code.

---

## ✨ Features

| Feature | Description |
| --- | --- |
| **Interactive CLI** | Styled menu-driven interface (rich) - choose options with numbers, no flags required. Every setting configurable in the CLI. |
| **Zero-Shot LLM Inference** | Prompt-based classification with OpenAI GPT or any LiteLLM-compatible provider - no training needed. |
| **Fine-Tuned Transformer** | A single `roberta-base` model trained on stacked per-clause binary labels via HuggingFace `Transformers`, evaluated per clause type. |
| **Comprehensive Benchmarks** | Precision, Recall, F1, Accuracy, cost-per-document, and latency metrics - both aggregate and per-clause-type. |
| **Automated Reporting** | Generates CSV metric tables, bar-chart visualizations, a summary memo, and a full engineering report. |
| **CLI Configuration** | All settings (LLM provider/model/API key/temperature/tokens, training params, clause types, output path) configurable via the menu - no `.env` edits needed. |
| **Legacy Flags** | `--quick-test`, `--max-samples N`, `--output DIR` still work and auto-skip the menu. |

---

## 🖥️ Usage

### Start the Interactive CLI

```bash
python compare_classifiers.py
```

You'll see a styled main menu:

```
═══════════════════ Main Menu ════════════════════
  1. Quick test (~50 samples)
  2. Full comparison (train + evaluate)
  3. Train only
  4. Evaluate only (use existing model)
  5. Configure settings
  6. View configuration
  7. Export / Import settings
  8. Exit

Choose an option [1]:
```

### Menu Actions

| # | Action | What it does |
|---|--------|-------------|
| **1** | Quick test | Runs a fast evaluation (~50 samples) with zero-shot LLM only - skips training |
| **2** | Full comparison | Downloads data, trains the fine-tuned model, evaluates both classifiers end-to-end |
| **3** | Train only | Loads the CUAD training set and trains a new fine-tuned model (no evaluation) |
| **4** | Evaluate only | Uses an existing model from `models/fine_tuned/` - skips training step |
| **5** | Configure settings | Opens the configuration submenu (see below) |
| **6** | View configuration | Shows a read-only summary of all active settings |
| **7** | Export / Import | Write current config to `.env.local` or view the environment table |

#### Configuration Sub-Menu

Accessed via option **5**, the configuration menu lets you change every setting:

```
═══════════ Configuration Menu ═══════════
  1. LLM Settings (provider, model, API key, temp, tokens, base URL)
  2. Training Settings (model, epochs, lr, batch size, max length, …)
  3. Clause Types (select / add / remove)
  4. Output Directory
  0. Back to main menu
```

**LLM Settings** - Configure provider (`openai` / `anthropic` / `google` / custom), model name, API key, base URL, temperature, and max tokens.

**Training Settings** - Configure training model name, epochs, batch size, learning rate, max token length, weight decay, warmup steps, eval steps, save steps.

**Clause Types** - Interactive checker-list of active clause types plus options to add custom clause types, remove individual ones, or load all 12 CUAD defaults.

**Output Directory** - Change where results are saved (creates the directory if needed).

### Legacy Flags (Non-Interactive Mode)

Pass any flag and the menu is skipped automatically:

```bash
# Quick test — ~50 samples, skips fine-tuned model training
python compare_classifiers.py --quick-test

# Limit evaluation to N samples
python compare_classifiers.py --max-samples 200

# Custom output directory
python compare_classifiers.py --output ./custom_output

# Combine flags
python compare_classifiers.py --quick-test --max-samples 50 --output ./results
```

---

## 📊 Supported Clause Types (CUAD)

The classifier evaluates all 12 standard CUAD clauses:

- **Agreement Effectiveness** - When the agreement becomes effective
- **Agreement Termination** - How the agreement can be terminated
- **Anti-Assignment** - Restrictions on transferring rights / obligations
- **Arbitration** - Dispute resolution through arbitration
- **Attorneys' Fees** - Payment of legal fees by the losing party
- **Notice** - How notices must be delivered
- **Governing Law** - Which jurisdiction's laws apply
- **Indemnification** - Compensation for losses
- **Jurisdiction** - Which courts have authority
- **Severability** - Invalid provisions don't void the agreement
- **Waiver** - Waiver of rights
- **Warranty** - Product / service guarantees

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Set Environment Variables

All settings are configurable via the interactive CLI menu - the `.env` file is **optional**. If you do want a persistent environment:

```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

#### `.env` Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | LLM backend (`openai`, `anthropic`, …) |
| `LLM_MODEL` | `gpt-3.5-turbo` | Model identifier |
| `LLM_API_KEY` | _(required)_ | Your API key for the chosen provider |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature |
| `LLM_MAX_TOKENS` | `500` | Max completion tokens |
| `LLM_BASE_URL` | _(optional)_ | Custom base URL (e.g. Azure / Ollama) |
| `TRAIN_MODEL` | `roberta-base` | HuggingFace model name for fine-tuning |
| `BATCH_SIZE` | `8` | Training and inference batch size |
| `LR` | `2e-5` | Learning rate |
| `NUM_EPOCHS` | `3` | Number of training epochs |
| `MAX_LENGTH` | `512` | Max token length (tokenizer pad / truncate) |
| `WEIGHT_DECAY` | `0.01` | AdamW weight decay |
| `WARMUP_STEPS` | `500` | Learning-rate warmup steps |
| `EVAL_STEPS` | `500` | Evaluation frequency during training |
| `SAVE_STEPS` | `1000` | Checkpoint save frequency |

### 3. Run the Comparison

```bash
# Interactive (default) — styled menu, all settings configurable in-app
python compare_classifiers.py

# Legacy flag — skips menu, runs quick test
python compare_classifiers.py --quick-test
```

---

## 📁 Project Structure

```
Contract-clause-classifier/
├── compare_classifiers.py        # Entry point — interactive CLI + pipeline
├── config.py                     # Dataclass-driven env-var configuration
├── evaluation.ipynb              # Interactive Jupyter notebook
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template (optional)
├── utils/
│   ├── __init__.py               # Public API exports
│   ├── llm_client.py             # Zero-shot LLM client (OpenAI + LiteLLM)
│   ├── data_loader.py            # CUAD dataset loading (HF → local fallback)
│   ├── classifier.py             # Fine-tuned transformer (ClauseDataset, FineTunedClassifier)
│   └── metrics.py                # ClassificationMetrics, InferenceStats, aggregation helpers
├── data/                         # Local dataset cache (CSV / JSON / Parquet)
├── models/                       # Saved fine-tuned model checkpoints
└── outputs/                      # Generated results (created on run)
    ├── comparison_metrics.csv     # Per-clause-type, per-method metrics
    ├── comparison_plot.png        # 2×2 bar chart — precision / recall / F1 / accuracy
    ├── summary.md                 # Quick-text summary
    └── comparison_report.md       # Full engineering report
```

---

## 🧠 Architecture Overview

### Data Flow

1. **Load** - `load_cuad_dataset()` fetches from HuggingFace (`lexnecn/contract-understanding-annotated-dataset`), with a fallback to local CSV / JSON / Parquet in `data/`.
2. **Schema** - Each document is wrapped as a `ContractData(contract_id, text, clauses_dict)`, where `clauses` maps clause type names to booleans.
3. **Preprocess** - `preprocess_data()` creates one (text, label) row per contract per clause type for binary classification.

### Fine-Tuned Classifier (`utils/classifier.py`)

| Aspect | Detail |
|---|---|
| **Model** | `AutoModelForSequenceClassification` - 2 output labels (present / absent) |
| **Dataset** | Custom `ClauseDataset(Dataset)` with tokenization, padding, and truncation |
| **Training** | HuggingFace `Trainer`; one model trained on stacked per-clause labels, evaluated per clause type |
| **Metrics** | Precision, Recall, F1, Accuracy tracked on the validation split |
| **Prediction** | Batched inference with timing → predictions, probabilities, avg latency ms |

### Zero-Shot LLM (`utils/llm_client.py`)

- Supports **OpenAI** (direct) and any **LiteLLM-compatible** provider.
- Sends a concise prompt asking the model to reply with only **YES** or **NO**.
- Calculates cost from token counts × per-million pricing in `config.llm.COSTS`.

### Metrics (`utils/metrics.py`)

| Class / Function | Purpose |
|---|---|
| `ClassificationMetrics` | Precision, Recall, F1, Accuracy, TP / FP / TN / FN |
| `InferenceStats` | Latency (total / avg / min / max ms), cost, token counts |
| `aggregate_metrics()` | Mean ± min / max across clause types |
| `print_comparison_table()` | Formatted CLI comparison output |

---

## 📈 Output Artifacts

All artifacts are written to the `outputs/` directory.

| File | Content |
|---|---|
| `comparison_metrics.csv` | Per-clause-type, per-method metrics in long format |
| `comparison_plot.png` | 2×2 bar chart comparing precision / recall / F1 / accuracy |
| `summary.md` | Concise text summary with key numbers |
| `comparison_report.md` | Full engineering report - methodology, tables, cost analysis, recommendations |

---

## 💰 Cost Analysis

### Zero-Shot LLM

| Aspect | Detail |
|---|---|
| **Pricing** | Pay-per-token based on provider pricing |
| **Example (GPT-3.5-turbo)** | ~$0.0005 / 1M input tokens, ~$0.0015 / 1M output tokens |

### Fine-Tuned Model

| Aspect | Detail |
|---|---|
| **Training** | One-time GPU cost (approx. $0.001 / second of GPU time) |
| **Inference** | Local execution - no ongoing API fees |

---

## 🎯 Engineering Recommendations

### Use Zero-Shot LLM When…

- ✅ Low volume processing (few documents per day)
- ✅ Rapid prototyping or proof-of-concept
- ✅ Limited ML expertise or infrastructure
- ✅ Data privacy is not a primary concern

### Use Fine-Tuned Model When…

- ✅ High-volume processing (hundreds of documents per day)
- ✅ Operational cost optimization matters
- ✅ Strict latency requirements (< 50 ms response time)
- ✅ Data must stay in-house for compliance
- ✅ Long-term, deployable solution is needed

---

## 🛠 Tech Stack

| Category | Library |
|---|---|
| **CLI** | `rich` (Panel, Prompt, Live, Table, Tree) |
| **ML / DL** | `torch`, `transformers`, `accelerate` |
| **Dataset** | `datasets` (HuggingFace) |
| **Evaluation** | `scikit-learn`, `numpy` |
| **LLM APIs** | `openai`, `litellm` |
| **Text Processing** | `tiktoken` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Utilities** | `pandas`, `python-dotenv`, `tqdm`, `requests` |

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

# 📄 Contract Clause Classifier

> A side-by-side comparison of **zero-shot LLM** versus **fine-tuned transformer** models for automated contract clause classification on the [CUAD dataset](https://huggingface.co/datasets/cuad).

Evaluate precision, recall, F1, accuracy, inference cost, and latency across all 12 standard CUAD clause types — and get engineering recommendations on when each approach makes sense.

---

## ✨ Features

| Feature | Description |
| --- | --- |
| **Zero-Shot LLM Inference** | Prompt-based classification with OpenAI GPT or any LiteLLM-compatible provider — no training needed. |
| **Fine-Tuned Transformer** | A `roberta-base` model trained per-clause-type (one-vs-rest binary classification) via HuggingFace `Transformers`. |
| **Comprehensive Benchmarks** | Precision, Recall, F1, Accuracy, cost-per-document, and latency metrics — both aggregate and per-clause-type. |
| **Automated Reporting** | Generates CSV metric tables, bar-chart visualizations, a summary memo, and a full engineering report. |

---

## 📊 Supported Clause Types (CUAD)

The classifier evaluates all 12 standard CUAD clauses:

- **Agreement Effectiveness** — When the agreement becomes effective
- **Agreement Termination** — How the agreement can be terminated
- **Anti-Assignment** — Restrictions on transferring rights / obligations
- **Arbitration** — Dispute resolution through arbitration
- **Attorneys' Fees** — Payment of legal fees by the losing party
- **Notice** — How notices must be delivered
- **Governing Law** — Which jurisdiction's laws apply
- **Indemnification** — Compensation for losses
- **Jurisdiction** — Which courts have authority
- **Severability** — Invalid provisions don't void the agreement
- **Waiver** — Waiver of rights
- **Warranty** — Product / service guarantees

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

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
# Quick test — ~50 samples, skips fine-tuned model training
python compare_classifiers.py --quick-test

# Full comparison — trains RoBERTa on the CUAD train set and benchmarks both
python compare_classifiers.py
```

#### CLI Options

| Flag | Type | Description |
|---|---|---|
| `--max-samples` | `int` | Limit evaluation to N samples |
| `--quick-test` | `flag` | Run a fast, limited-sample test (skips training) |
| `--output` | `str` | Custom output directory for results |

---

## 📁 Project Structure

```
Contract-clause-classifier/
├── compare_classifiers.py        # Entry point — full comparison pipeline
├── config.py                     # Dataclass-driven env-var configuration
├── evaluation.ipynb              # Interactive Jupyter notebook
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
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
    ├── cost_latency_comparison.png
    ├── summary.md                 # Quick-text summary
    └── comparison_report.md       # Full engineering report
```

---

## 🧠 Architecture Overview

### Data Flow

1. **Load** — `load_cuad_dataset()` fetches from HuggingFace (`lexnecn/contract-understanding-annotated-dataset`), with a fallback to local CSV / JSON / Parquet in `data/`.
2. **Schema** — Each document is wrapped as a `ContractData(contract_id, text, clauses_dict)`, where `clauses` maps clause type names to booleans.
3. **Preprocess** — `preprocess_data()` creates one (text, label) row per contract per clause type for binary classification.

### Fine-Tuned Classifier (`utils/classifier.py`)

| Aspect | Detail |
|---|---|
| **Model** | `AutoModelForSequenceClassification` — 2 output labels (present / absent) |
| **Dataset** | Custom `ClauseDataset(Dataset)` with tokenization, padding, and truncation |
| **Training** | HuggingFace `Trainer`; one model per clause type (one-vs-rest) |
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
| `comparison_report.md` | Full engineering report — methodology, tables, cost analysis, recommendations |

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
| **Inference** | Local execution — no ongoing API fees |

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
| **ML / DL** | `torch`, `transformers`, `accelerate` |
| **Dataset** | `datasets` (HuggingFace) |
| **Evaluation** | `scikit-learn`, `numpy` |
| **LLM APIs** | `openai`, `litellm` |
| **Text Processing** | `tiktoken` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Utilities** | `pandas`, `python-dotenv`, `tqdm`, `requests` |

---

## 📄 License

This project is provided for educational and demonstration purposes.

# Contract Clause Classifier

Compare zero-shot LLM against a fine-tuned classifier for contract clause classification on the CUAD dataset.

## Features

- **Zero-Shot LLM**: Prompt-based classification without training
- **Fine-Tuned Model**: Custom-trained transformer model
- **Comprehensive Metrics**: Precision, Recall, F1, Accuracy, Cost, Latency
- **Comparison Report**: Visualizations and engineering judgment

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

### 3. Run Comparison (Python Script)

```bash
python compare_classifiers.py --quick-test
```

### 4. Run Comparison (Jupyter Notebook)

```bash
jupyter notebook evaluation.ipynb
```

## Usage

### Python Script

```bash
# Full comparison (uses full dataset)
python compare_classifiers.py

# Quick test (uses smaller samples)
python compare_classifiers.py --quick-test

# Limit number of samples
python compare_classifiers.py --max-samples 100

# Custom output directory
python compare_classifiers.py --output /path/to/output
```

### Jupyter Notebook

Open `evaluation.ipynb` in Jupyter Notebook or JupyterLab to run the comparison interactively.

## Configuration

### Environment Variables (.env)

```env
# LLM Provider
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
LLM_API_KEY=your_api_key
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=500

# Training
TRAIN_MODEL=roberta-base
BATCH_SIZE=8
NUM_EPOCHS=3
LR=2e-5
MAX_LENGTH=512
```

### Clause Types

The classifier supports these CUAD clause types:
- Agreement Effectiveness
- Agreement Termination
- Anti-Assignment
- Arbitration
- Attorneys' Fees
- Notice
- Governing Law
- Indemnification
- Jurisdiction
- Severability
- Waiver
- Warranty

## Output

The comparison generates:

1. **comparison_metrics.csv**: Detailed metrics per clause type
2. **comparison_plot.png**: Visualization of precision/recall/F1/accuracy
3. **cost_latency_comparison.png**: Cost and latency comparison
4. **summary.md**: Text summary of results
5. **comparison_report.md**: Comprehensive engineering report

## Cost Analysis

### Zero-Shot LLM Costs
- Pay-per-token based on provider pricing
- Example (OpenAI GPT-3.5-turbo): ~$0.0005/1M input tokens, $0.0015/1M output tokens

### Fine-Tuned Model Costs
- Training: One-time GPU cost
- Inference: Local execution (no API costs)
- Estimated: ~$0.001/second GPU time

## Engineering Judgment: When is Fine-Tuning Worth It?

### Use Zero-Shot LLM When:
- Low volume (few docs/day)
- Quick prototype/POC needed
- Limited ML expertise
- Data privacy not critical

### Use Fine-Tuned Model When:
- High volume processing (>100 docs/day)
- Cost optimization important
- Strict latency requirements (<50ms)
- Data privacy required
- Long-term deployment

## Project Structure

```
Contract-clause-classifier/
├── compare_classifiers.py      # Main comparison script
├── evaluation.ipynb           # Jupyter notebook
├── requirements.txt           # Python dependencies
├── config.py                  # Configuration management
├── .env.example               # Environment variables template
├── utils/
│   ├── __init__.py
│   ├── llm_client.py          # Zero-shot LLM client
│   ├── data_loader.py         # CUAD dataset loading
│   ├── classifier.py          # Fine-tuned classifier
│   └── metrics.py             # Evaluation metrics
└── outputs/                   # Generated reports (created on run)
    ├── comparison_metrics.csv
    ├── comparison_plot.png
    ├── summary.md
    └── comparison_report.md
```

## License

This project is for educational and demonstration purposes.

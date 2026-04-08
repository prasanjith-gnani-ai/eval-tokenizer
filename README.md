# Indic Tokenizer Training and Evaluation Suite

A comprehensive pipeline for training, evaluating, and extending multilingual tokenizers for Indic languages using the Sangraha dataset.

## Overview

This suite provides:
- **Training**: BPE tokenizer training with MUTANT-style script-aware pre-tokenization
- **Evaluation**: Multi-metric evaluation across 23 Indic languages
- **Comparison**: Side-by-side comparison of multiple tokenizers
- **Extension**: Finding novel tokens for extending base models like Nemotron
- **Merging**: Integrating novel tokens into existing tokenizers

## Project Structure

### Core Scripts
- `train_tokenizer.py`: Train BPE tokenizer on Sangraha dataset with MUTANT pre-tokenization
- `evaluate_tokenizer.py`: Evaluate tokenizer metrics on Indic languages
- `compare_tokenizer.py`: Compare multiple tokenizers with plots and tables
- `tokenizer_merge.py`: Merge novel tokens into base tokenizer vocabularies
- `add_english.py`: Download and add English data from FineWeb-edu to Sangraha
- `check.py`: Utility script for dataset and environment checks

### Data and Outputs
- `sangraha_data/`: Downloaded Sangraha training data (Indic + English)
- `indic_tokenizer_output/`: Trained Indic BPE tokenizer and evaluation results
- `nemotron_indic_tokenizer/`: Extended Nemotron tokenizer with Indic tokens
- `results/`: Evaluation results (CSV/JSON) for different tokenizers
- `plots/`: Comparison plots and visualizations
- `requirements.txt`: Python dependencies

## Installation

```bash
# Clone or navigate to tokenizer directory
cd tokenizer

# Install dependencies
pip install -r requirements.txt

# For datatrove (optional, for large-scale data processing)
pip install datatrove
```

## Usage

### 1. Prepare Training Data

Download Sangraha dataset and add English:

```bash
# Download Indic data (already done if using existing)
python train_tokenizer.py  # Uncomment download_sangraha() if needed

# Add English data from FineWeb-edu
python add_english.py
```

### 2. Train Tokenizer

```bash
python train_tokenizer.py
```

Trains BPE tokenizer with:
- 32k vocabulary
- MUTANT script-aware pre-tokenization
- Balanced Indic + English data

### 3. Evaluate Tokenizer

Edit `MODEL_ID` in `evaluate_tokenizer.py` to your trained tokenizer:

```python
MODEL_ID = "indic_tokenizer_output/indic_bpe_tokenizer.json"
```

Run evaluation:

```bash
python evaluate_tokenizer.py
```

### 4. Find Novel Tokens for Extension

```bash
# In train_tokenizer.py, uncomment the find_novel_tokens call
python train_tokenizer.py
```

Generates `novel_tokens.json` for extending base models.

### 5. Merge Tokens into Base Model

```bash
python tokenizer_merge.py
```

Creates extended tokenizer in `nemotron_indic_tokenizer/`.

### 6. Compare Multiple Tokenizers

After evaluating multiple models:

```bash
python compare_tokenizer.py --results_dir results/ --output_dir plots/
```

## Metrics

### Training Metrics
- **Fertility**: Tokens per word (lower = better, ideal ~1.0)
- **Vocabulary Coverage**: % of script Unicode points in vocab

### Evaluation Metrics
- **Fertility**: Avg tokens per word across languages
- **NSL**: Normalized Sequence Length vs. reference (mT5)
- **Bytes Per Token**: Compression efficiency
- **Continuation Rate**: % words split into multiple tokens
- **UNK Rate**: % unknown tokens
- **Vocab Coverage**: Script character coverage
- **Avg Token Length**: Mean token character length

## Key Features

### MUTANT Pre-tokenization
Script-aware regex that prevents cross-script token merges, improving fertility for rare scripts like Santali (Ol Chiki).

### Balanced Multilingual Data
23 Indic languages + English, maintaining equal representation per language.

### Comprehensive Evaluation
Multi-metric evaluation with publication-quality visualizations.

### Model Extension
Automated pipeline for extending English models (like Nemotron) with Indic tokens.

## Requirements

- Python 3.8+
- 16GB+ RAM recommended for training
- Dependencies: transformers, datasets, tokenizers, pandas, matplotlib, seaborn, tqdm, regex

## Citation

Based on the MUTANT paper (arxiv 2511.03237) for script-aware tokenization.

## Troubleshooting

- **Import errors**: Ensure compatible transformers/huggingface-hub versions
- **Memory issues**: Reduce `MAX_TRAIN_DOCS` or use streaming
- **Dataset download**: Check internet connection for HuggingFace datasets
- **Evaluation fails**: Verify tokenizer file paths and formats
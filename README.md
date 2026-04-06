# Indic Tokenizer Evaluation Suite

Evaluates HuggingFace tokenizers on the Sangraha dataset across 23 Indic languages.

## Metrics
- **Fertility**: Avg tokens per word (lower = better)
- **NSL**: Normalized Sequence Length vs. reference tokenizer
- **Bytes Per Token**: Compression efficiency (higher = better)
- **Continuation Rate**: % words split into ≥2 sub-tokens (lower = better)
- **UNK Rate**: Fraction of <unk> tokens (lower = better)
- **Vocab Coverage**: % Unicode codepoints in vocabulary
- **Avg Token Length**: Mean character length of tokens

## Installation
```bash
pip install -r tokenizer/requirements.txt
```

## Usage

### Single Model Evaluation
Edit configuration in `tokenizer/evaluate_tokenizer.py`:
- `MODEL_ID`: Tokenizer to evaluate
- `REF_MODEL_ID`: Reference tokenizer (default: google/mt5-base)
- `SUBSET`: "verified" or "unverified"
- `SAMPLES_PER_LANG`: Documents per language (1000 recommended)

Run evaluation:
```bash
cd tokenizer
python evaluate_tokenizer.py
```

Results saved to `results/` as CSV and JSON.

### Compare Multiple Models
After running evaluation for multiple models, compare them:
```bash
cd tokenizer
python compare_tokenizer.py --results_dir results/ --output_dir plots/
```

Generates comparison tables and 8 publication-quality plots:
- Fertility bars per language
- Fertility heatmap
- Radar chart of average metrics
- Fertility distribution boxplot
- Fertility vs. vocabulary coverage scatter
- Grade distribution stacked bars
- NSL lines per language
- Normalized summary bars

## Requirements
- Python 3.8+
- Dependencies: transformers, datasets, pandas, tabulate, tqdm, tiktoken, sentencepiece, protobuf, matplotlib, seaborn
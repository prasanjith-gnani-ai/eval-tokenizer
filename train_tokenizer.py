"""
Indic Tokenizer Training Pipeline
==================================
- Train BPE on Sangraha UNVERIFIED (large, noisy → diverse coverage)
- Evaluate fertility on Sangraha VERIFIED (clean → generalization test)
- Script-aware pre-tokenization following MUTANT paper (arxiv 2511.03237)

Usage:
    pip install datasets tokenizers regex transformers huggingface_hub
    python indic_tokenizer_pipeline.py
"""

import os
import re
import json
import regex
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset, get_dataset_config_names
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import PreTokenizer, Split
import tokenizers
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATA_DIR       = Path("sangraha_data")
TRAIN_FILE     = DATA_DIR / "sangraha_unverified_train.txt"
TOKENIZER_DIR  = Path("indic_tokenizer_output")
MAX_TRAIN_DOCS = 200_000   # per language; reduce if RAM is tight
MAX_TEST_DOCS  = 5_000     # per language for fertility eval

# All 22 official Indic languages in Sangraha
INDIC_LANGUAGES = [
    "asm",  # Assamese
    "ben",  # Bengali
    "brx",  # Bodo
    "doi",  # Dogri
    "guj",  # Gujarati
    "hin",  # Hindi
    "kan",  # Kannada
    "kas",  # Kashmiri
    "kok",  # Konkani
    "mai",  # Maithili
    "mal",  # Malayalam
    "mar",  # Marathi
    "mni",  # Manipuri
    "nep",  # Nepali
    "ory",  # Odia
    "pan",  # Punjabi
    "san",  # Sanskrit
    "sat",  # Santali  ← worst fertility in your chart, needs most help
    "snd",  # Sindhi
    "tam",  # Tamil
    "tel",  # Telugu
    "urd",  # Urdu
]

# ─────────────────────────────────────────────
# STEP 1: DOWNLOAD SANGRAHA
# ─────────────────────────────────────────────

def download_sangraha(
    languages=INDIC_LANGUAGES,
    max_train_docs=MAX_TRAIN_DOCS,
):
    DATA_DIR.mkdir(exist_ok=True)

    train_counts = defaultdict(int)

    print("=" * 60)
    print("Downloading Sangraha UNVERIFIED (train) ...")
    print("=" * 60)

    with open(TRAIN_FILE, "w", encoding="utf-8") as f_train:
        for lang in languages:
            print(f"  [{lang}] unverified ...")
            try:
                # Sangraha unverified has language-specific splits
                ds = load_dataset(
                    "ai4bharat/sangraha",
                    name="unverified",
                    split=lang,
                    streaming=True,         # stream to avoid OOM
                )
                for i, row in enumerate(ds):
                    if i >= max_train_docs:
                        break
                    text = row.get("text", "").strip()
                    if len(text) > 20:      # skip near-empty docs
                        f_train.write(text + "\n")
                        train_counts[lang] += 1
                print(f"      → {train_counts[lang]:,} docs")
            except Exception as e:
                print(f"      ✗ Skipping {lang}: {e}")

    print(f"\n✓ Train file: {TRAIN_FILE} ({TRAIN_FILE.stat().st_size / 1e6:.1f} MB)")
    return dict(train_counts)


# ─────────────────────────────────────────────
# STEP 2: MUTANT-STYLE PRE-TOKENIZER
# ─────────────────────────────────────────────
#
# The key insight from MUTANT (arxiv 2511.03237):
# Use script-specific regex so BPE merges happen WITHIN a script,
# never ACROSS scripts. This prevents garbage cross-script tokens
# and massively reduces fertility for rare scripts like Santali (Ol Chiki).
#
# Pattern priority order matters:
#   1. Each Indic script block  → script-pure tokens
#   2. Latin words              → English preserved
#   3. Digits                   → numbers as units
#   4. Punctuation/symbols      → isolated
# ─────────────────────────────────────────────

MUTANT_PATTERN = regex.compile(
    r"""(?x)
    # ── Indic scripts (one block per script; order = Unicode block order) ──
    [\u0900-\u097F\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF]+  # Devanagari (Hindi, Marathi, Sanskrit, Maithili, Dogri, Nepali, Konkani, Bodo, Kashmiri/Devanagari)
    |[\u0980-\u09FF]+  # Bengali (also Assamese)
    |[\u0A00-\u0A7F]+  # Gurmukhi (Punjabi)
    |[\u0A80-\u0AFF]+  # Gujarati
    |[\u0B00-\u0B7F]+  # Odia
    |[\u0B80-\u0BFF]+  # Tamil
    |[\u0C00-\u0C7F]+  # Telugu
    |[\u0C80-\u0CFF]+  # Kannada
    |[\u0D00-\u0D7F]+  # Malayalam
    |[\u1C50-\u1C7F]+  # Ol Chiki (Santali) ← critical for your worst performer
    |[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+  # Arabic script (Urdu, Kashmiri/Nastaliq, Sindhi)
    |[\u0900-\u097F]+  # Devanagari fallback
    # ── Latin / English ──
    |[a-zA-Z]+
    # ── Digits (keep as units) ──
    |[0-9]+(?:[.,][0-9]+)*
    # ── Whitespace (will be filtered by BPE trainer) ──
    |\s+
    # ── Everything else: punctuation, symbols ──
    |.
    """,
    regex.UNICODE,
)


class MUTANTPreTokenizer:
    """
    Custom pre-tokenizer that applies the MUTANT script-aware regex.
    Returns (token_string, (start, end)) tuples as required by HF tokenizers.
    """
    def pre_tokenize_str(self, sequence: str):
        tokens = []
        for match in MUTANT_PATTERN.finditer(sequence):
            span = match.span()
            token = match.group()
            if token.strip():          # drop pure whitespace spans
                tokens.append((token, span))
        return tokens


# ─────────────────────────────────────────────
# STEP 3: TRAIN BPE TOKENIZER
# ─────────────────────────────────────────────

def train_tokenizer(train_file=TRAIN_FILE):
    print("\n" + "=" * 60)
    print("Training BPE tokenizer (MUTANT-style) ...")
    print("=" * 60)

    # Pre-process training data with MUTANT regex
    print("  Pre-processing training data with MUTANT regex...")
    processed_file = train_file.parent / (train_file.name + ".processed")

    # Count total lines for progress bar
    with open(train_file, "r", encoding="utf-8") as f_in:
        total_lines = sum(1 for _ in f_in)

    with open(train_file, "r", encoding="utf-8") as f_in, \
         open(processed_file, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in, total=total_lines, desc="Processing MUTANT", unit="line"):
            line = line.strip()
            if not line:
                continue
            # Apply MUTANT pre-tokenization
            pre_tokenized = []
            for match in MUTANT_PATTERN.finditer(line):
                token = match.group()
                if token.strip():  # Skip pure whitespace
                    pre_tokenized.append(token)
            # Join with spaces (BPE will learn from this)
            f_out.write(" ".join(pre_tokenized) + "\n")

    print(f"  Processed data saved to: {processed_file}")

    # Initialise BPE model with standard pre-tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = BpeTrainer(
        vocab_size=32_000,       # will find ~30k novel tokens after overlap removal
        min_frequency=5,         # ignore hapax legomena (noise in unverified data)
        special_tokens=[
            "[UNK]",
            "[PAD]",
            "[BOS]",
            "[EOS]",
        ],
        show_progress=True,
    )

    print(f"  Training on: {processed_file}")
    tokenizer.train(files=[str(processed_file)], trainer=trainer)

    # Save
    TOKENIZER_DIR.mkdir(exist_ok=True)
    save_path = str(TOKENIZER_DIR / "indic_bpe_tokenizer.json")
    tokenizer.save(save_path)
    print(f"\n✓ Tokenizer saved to: {save_path}")
    print(f"  Final vocab size: {tokenizer.get_vocab_size():,}")

    return tokenizer


# ─────────────────────────────────────────────
# STEP 4: EVALUATE FERTILITY ON VERIFIED SET
# ─────────────────────────────────────────────
#
# Fertility = tokens / words  (lower is better; 1.0 = perfect)
# We evaluate per-language by loading the verified docs and
# matching against the language codes in the filename.
# Since we merged all languages into one file, we re-download
# verified per-language for accurate per-lang fertility.
# ─────────────────────────────────────────────

def simple_word_count(text: str) -> int:
    """Count whitespace-delimited words (language-agnostic)."""
    return len(text.split())


def evaluate_fertility(
    tokenizer: Tokenizer,
    languages=INDIC_LANGUAGES,
    max_docs=MAX_TEST_DOCS,
):
    print("\n" + "=" * 60)
    print("Evaluating fertility on Sangraha UNVERIFIED ...")
    print("=" * 60)

    results = {}

    for lang in languages:
        total_tokens = 0
        total_words  = 0
        docs_seen    = 0

        try:
            ds = load_dataset(
                "ai4bharat/sangraha",
                name="unverified",
                split=lang,
                streaming=True,
            )
            for row in ds:
                if docs_seen >= max_docs:
                    break
                text = row.get("text", "").strip()
                if len(text) < 20:
                    continue

                words  = simple_word_count(text)
                # Apply MUTANT pre-tokenization before encoding
                pre_tokenized = []
                for match in MUTANT_PATTERN.finditer(text):
                    token = match.group()
                    if token.strip():
                        pre_tokenized.append(token)
                pre_tokenized_text = " ".join(pre_tokenized)
                enc    = tokenizer.encode(pre_tokenized_text)
                tokens = len(enc.ids)

                total_words  += words
                total_tokens += tokens
                docs_seen    += 1

        except Exception as e:
            print(f"  [{lang}] ✗ {e}")
            continue

        if total_words > 0:
            fertility = total_tokens / total_words
            results[lang] = {
                "fertility":   round(fertility, 3),
                "total_words": total_words,
                "total_tokens": total_tokens,
                "docs":        docs_seen,
            }

    # ── Print table ──
    print(f"\n{'Lang':<6} {'Fertility':>10} {'Docs':>8} {'Rating'}")
    print("-" * 45)

    THRESHOLDS = {
        "Excellent": 1.5,
        "Good":      2.5,
        "Marginal":  4.0,
    }

    for lang, r in sorted(results.items(), key=lambda x: x[1]["fertility"]):
        f = r["fertility"]
        if f < 1.5:
            rating = "✅ Excellent"
        elif f < 2.5:
            rating = "🟢 Good"
        elif f < 4.0:
            rating = "🟡 Marginal"
        else:
            rating = "🔴 Poor"
        print(f"{lang:<6} {f:>10.3f} {r['docs']:>8,}   {rating}")

    avg = sum(r["fertility"] for r in results.values()) / len(results) if results else 0
    print("-" * 45)
    print(f"{'AVG':<6} {avg:>10.3f}              ← compare vs Nemotron baseline")

    # Save results
    out_path = TOKENIZER_DIR / "fertility_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {out_path}")

    return results


# ─────────────────────────────────────────────
# STEP 5: FIND NOVEL TOKENS (for Nemotron extension)
# ─────────────────────────────────────────────

def find_novel_tokens(indic_tokenizer: Tokenizer, base_model_name: str):
    """
    Compare new Indic vocab against Nemotron base vocab.
    Returns the list of tokens to append.
    """
    print("\n" + "=" * 60)
    print("Finding novel tokens for Nemotron extension ...")
    print("=" * 60)

    from transformers import AutoTokenizer
    base_tok = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )
    base_vocab = set(base_tok.get_vocab().keys())

    indic_vocab = set(indic_tokenizer.get_vocab().keys())
    special = {"[UNK]", "[PAD]", "[BOS]", "[EOS]"}
    indic_vocab -= special

    novel = sorted(indic_vocab - base_vocab)

    print(f"  Nemotron vocab size : {len(base_vocab):,}")
    print(f"  Indic BPE vocab size: {len(indic_vocab):,}")
    print(f"  Overlapping tokens  : {len(indic_vocab & base_vocab):,}")
    print(f"  Novel tokens to add : {len(novel):,}")

    # Debug: Show first 10 novel tokens
    print(f"  First 10 novel tokens: {novel[:10]}")

    # Check if special tokens are in base vocab
    base_special = special & base_vocab
    print(f"  Special tokens in base: {base_special}")
    print(f"  Special tokens in Indic: {special & indic_vocab}")

    # Save the list
    out_path = TOKENIZER_DIR / "novel_tokens.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(novel, f, ensure_ascii=False, indent=2)
    print(f"✓ Novel token list saved to: {out_path}")

    return novel


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. Download data ──────────────────────
    #download_sangraha()

    # ── 2. Train tokenizer ────────────────────
    #tokenizer = train_tokenizer()

    # ── 3. Evaluate fertility ─────────────────
    #results = evaluate_fertility(tokenizer)


    tokenizer_path = "indic_tokenizer_output/indic_bpe_tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # ── 4. Find novel tokens (optional) ───────
    # Uncomment when you have Nemotron downloaded locally or enough disk:
    #
    novel = find_novel_tokens(
        tokenizer,
        base_model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    )

    # print("\n✅ Pipeline complete.")
    # print(f"   Tokenizer : {TOKENIZER_DIR}/indic_bpe_tokenizer.json")
    # print(f"   Fertility : {TOKENIZER_DIR}/fertility_results.json")
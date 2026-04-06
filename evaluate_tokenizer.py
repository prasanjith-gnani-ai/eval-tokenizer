"""
=============================================================================
Indic Language Tokenizer Evaluation Suite
=============================================================================
Evaluates any HuggingFace tokenizer on the Sangraha dataset across all 23
Indic languages in the verified split.

Metrics computed per language:
  1. Fertility          — avg tokens per whitespace word (lower = better)
  2. NSL                — Normalized Sequence Length vs. a reference tokenizer
  3. Bytes Per Token    — compression efficiency (higher = better)
  4. Continuation Rate  — % words split into ≥2 sub-tokens (lower = better)
  5. OOV / UNK Rate     — fraction of tokens that are <unk> (lower = better)
  6. Vocab Coverage     — % of script's Unicode codepoints in vocabulary
  7. Avg Token Length   — mean character length of produced tokens

Usage:
    pip install transformers datasets pandas tabulate tqdm
    python evaluate_tokenizer.py

Configuration is at the top of the file — change MODEL_ID, REF_MODEL_ID,
SAMPLES_PER_LANG, and SUBSET to fit your needs.
=============================================================================
"""

import re
import unicodedata
import json
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset, logging as ds_logging
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
ds_logging.set_verbosity_error()


# =============================================================================
# CONFIGURATION  — edit these
# =============================================================================

MODEL_ID = "google/gemma-4-31B-it"

# Reference tokenizer to compute NSL against. Use a strong multilingual one.
REF_MODEL_ID = "google/mt5-base"   # or "ai4bharat/indic-bert"

# Sangraha subset: "verified" (highest quality, 23 langs) or "unverified"
SUBSET = "verified"

# How many documents to sample per language.
# Recommendation:
#   - 500  → fast smoke test (~5 min)
#   - 1000 → robust evaluation (~15 min)  ← DEFAULT
#   - 5000 → publication-quality (~1 hr)
SAMPLES_PER_LANG = 1000

# Max characters per document to avoid outlier very-long docs dominating stats
MAX_CHARS_PER_DOC = 2000

# Output directory for results
OUTPUT_DIR = "results"

# Whether to also print per-language token visualisation examples
PRINT_EXAMPLES = True
EXAMPLES_PER_LANG = 2       # number of sample sentences to show
EXAMPLE_MAX_CHARS = 120     # truncate long examples


# =============================================================================
# LANGUAGE METADATA
# =============================================================================

# All languages in the Sangraha *verified* split
SANGRAHA_VERIFIED_LANGS = {
    "asm": ("Assamese",  "Assamese",       (0x0980, 0x09FF)),  # shares Bengali script
    "ben": ("Bengali",   "Bengali",         (0x0980, 0x09FF)),
    "brx": ("Bodo",      "Devanagari",      (0x0900, 0x097F)),
    "doi": ("Dogri",     "Devanagari",      (0x0900, 0x097F)),
    "eng": ("English",   "Latin",           (0x0041, 0x007A)),
    "gom": ("Konkani",   "Devanagari",      (0x0900, 0x097F)),
    "guj": ("Gujarati",  "Gujarati",        (0x0A80, 0x0AFF)),
    "hin": ("Hindi",     "Devanagari",      (0x0900, 0x097F)),
    "kan": ("Kannada",   "Kannada",         (0x0C80, 0x0CFF)),
    "kas": ("Kashmiri",  "Perso-Arabic",    (0x0600, 0x06FF)),
    "mai": ("Maithili",  "Devanagari",      (0x0900, 0x097F)),
    "mal": ("Malayalam", "Malayalam",       (0x0D00, 0x0D7F)),
    "mar": ("Marathi",   "Devanagari",      (0x0900, 0x097F)),
    "mni": ("Manipuri",  "Bengali",         (0x0980, 0x09FF)),
    "nep": ("Nepali",    "Devanagari",      (0x0900, 0x097F)),
    "ori": ("Odia",      "Odia",            (0x0B00, 0x0B7F)),
    "pan": ("Punjabi",   "Gurmukhi",        (0x0A00, 0x0A7F)),
    "san": ("Sanskrit",  "Devanagari",      (0x0900, 0x097F)),
    "sat": ("Santali",   "Ol Chiki",        (0x1C50, 0x1C7F)),
    "snd": ("Sindhi",    "Perso-Arabic",    (0x0600, 0x06FF)),
    "tam": ("Tamil",     "Tamil",           (0x0B80, 0x0BFF)),
    "tel": ("Telugu",    "Telugu",          (0x0C00, 0x0C7F)),
    "urd": ("Urdu",      "Perso-Arabic",    (0x0600, 0x06FF)),
}

# Languages in unverified split (subset of verified)
SANGRAHA_UNVERIFIED_LANGS = {
    k: v for k, v in SANGRAHA_VERIFIED_LANGS.items()
    if k in {"asm","ben","guj","hin","kan","mal","mar","nep",
              "ori","pan","san","tam","tel","urd"}
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_sangraha_language(lang_code: str, subset: str, n_samples: int) -> List[str]:
    """
    Load n_samples text documents from Sangraha for a given language.
    Uses streaming to avoid downloading the full 705 GB dataset.
    Truncates each doc to MAX_CHARS_PER_DOC for fair comparison.
    """
    data_dir = f"{subset}/{lang_code}"
    try:
        ds = load_dataset(
            "ai4bharat/sangraha",
            data_dir=data_dir,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        texts = []
        for example in ds:
            text = example.get("text", "").strip()
            if len(text) < 10:          # skip near-empty docs
                continue
            text = text[:MAX_CHARS_PER_DOC]
            texts.append(text)
            if len(texts) >= n_samples:
                break
        return texts
    except Exception as e:
        print(f"  [WARN] Could not load {lang_code}/{subset}: {e}")
        return []


# =============================================================================
# METRIC HELPERS
# =============================================================================

def tokenize_batch(tokenizer, texts: List[str]) -> List[List[str]]:
    """Return list of token lists for each text."""
    results = []
    for text in texts:
        enc = tokenizer(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
        results.append(tokens)
    return results


def get_words(text: str) -> List[str]:
    """Whitespace-split words, filtering empty strings."""
    return [w for w in text.split() if w]


def fertility_score(texts: List[str], token_lists: List[List[str]]) -> float:
    """
    Fertility = total_tokens / total_words.
    Measures how many sub-tokens the tokenizer creates per word.
    Ideal = 1.0; English BPE ≈ 1.1–1.3; unseen scripts can reach 8+.
    """
    total_tokens = sum(len(toks) for toks in token_lists)
    total_words  = sum(len(get_words(t)) for t in texts)
    return total_tokens / total_words if total_words else 0.0


def continuation_rate(texts: List[str], token_lists: List[List[str]],
                      tokenizer) -> float:
    """
    Proportion of words that are split into ≥2 tokens.
    A word is 'continued' if any of its tokens is a continuation piece
    (starts with ##, Ġ, ▁ continuation, or byte-level Ċ patterns).
    We detect this by re-tokenizing each word individually.
    """
    split_words = 0
    total_words = 0
    for text in texts:
        for word in get_words(text):
            ids = tokenizer(word, add_special_tokens=False)["input_ids"]
            total_words += 1
            if len(ids) > 1:
                split_words += 1
    return split_words / total_words if total_words else 0.0


def bytes_per_token(texts: List[str], token_lists: List[List[str]]) -> float:
    """
    Average UTF-8 bytes per token.
    Higher = more information-dense tokens (better compression).
    English GPT-4 tokenizer: ~3.7 bytes/token.
    Indic scripts are 3 bytes/char; good Indic tokenizers: 6–12 bytes/token.
    """
    total_bytes  = sum(len(t.encode("utf-8")) for t in texts)
    total_tokens = sum(len(toks) for toks in token_lists)
    return total_bytes / total_tokens if total_tokens else 0.0


def unk_rate(token_lists: List[List[str]], tokenizer) -> float:
    """
    Fraction of tokens that are the <unk> token.
    BPE tokenizers with byte-fallback never produce UNK, so this is 0.
    WordPiece tokenizers can produce UNK for unseen characters.
    """
    unk = tokenizer.unk_token
    if unk is None:
        return 0.0  # byte-fallback BPE: no UNK possible
    total_tokens = sum(len(toks) for toks in token_lists)
    unk_tokens   = sum(toks.count(unk) for toks in token_lists)
    return unk_tokens / total_tokens if total_tokens else 0.0


def vocab_coverage(tokenizer, script_range: Tuple[int, int]) -> float:
    """
    Fraction of Unicode codepoints in the script's range that appear
    as a single-character token in the vocabulary.
    Low coverage → characters are always byte-split → high fertility.
    """
    start, end = script_range
    vocab = tokenizer.get_vocab()
    # Normalise vocab keys: some tokenizers prepend spaces/underscores
    vocab_chars = set()
    for tok in vocab:
        # Strip common BPE prefixes
        cleaned = tok.lstrip("▁Ġ##Ċ")
        if cleaned:
            vocab_chars.add(cleaned[0])
    chars_in_range = [chr(c) for c in range(start, end + 1)
                      if unicodedata.category(chr(c)) not in ("Cn",)]
    if not chars_in_range:
        return 0.0
    covered = sum(1 for c in chars_in_range if c in vocab_chars)
    return covered / len(chars_in_range)


def nsl_score(texts: List[str],
              model_token_lists: List[List[str]],
              ref_tokenizer) -> float:
    """
    Normalized Sequence Length (NSL) = avg_len(model) / avg_len(reference).
    NSL < 1 → model tokenizes more efficiently than reference.
    NSL > 1 → model uses more tokens than reference (worse for this language).
    Reference is typically a strong multilingual model (mT5, IndicBERT).
    """
    model_lens = [len(toks) for toks in model_token_lists]
    ref_lens = []
    for text in texts:
        ref_enc = ref_tokenizer(text, add_special_tokens=False)
        ref_lens.append(len(ref_enc["input_ids"]))
    avg_model = sum(model_lens) / len(model_lens) if model_lens else 0
    avg_ref   = sum(ref_lens)   / len(ref_lens)   if ref_lens   else 0
    return avg_model / avg_ref if avg_ref else 0.0


def avg_token_length(token_lists: List[List[str]], tokenizer) -> float:
    """
    Average character length of tokens (excluding special tokens).
    Longer tokens = richer vocabulary representation.
    """
    unk = tokenizer.unk_token or ""
    all_lens = []
    for toks in token_lists:
        for tok in toks:
            if tok == unk:
                continue
            # Remove BPE space markers for length calculation
            clean = tok.lstrip("▁Ġ##Ċ<>[]")
            all_lens.append(len(clean))
    return sum(all_lens) / len(all_lens) if all_lens else 0.0


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class LangResult:
    lang_code:         str
    lang_name:         str
    script:            str
    n_docs:            int
    n_words:           int
    n_tokens:          int
    fertility:         float
    continuation_rate: float
    bytes_per_token:   float
    unk_rate:          float
    vocab_coverage:    float
    nsl:               float
    avg_token_len:     float
    examples:          List[Tuple[str, List[str]]] = field(default_factory=list)


# =============================================================================
# MAIN EVALUATION LOOP
# =============================================================================

def evaluate_language(
    lang_code:     str,
    lang_meta:     tuple,
    model_tok,
    ref_tok,
    subset:        str,
    n_samples:     int,
    n_examples:    int,
) -> Optional[LangResult]:
    """Run all metrics for one language. Returns None if no data loaded."""
    lang_name, script, script_range = lang_meta

    print(f"\n  [{lang_code}] {lang_name} ({script})")

    # 1. Load data
    texts = load_sangraha_language(lang_code, subset, n_samples)
    if not texts:
        print(f"       ✗ No data available, skipping.")
        return None
    print(f"       ✓ Loaded {len(texts)} documents")

    # 2. Tokenize with the model tokenizer
    token_lists = tokenize_batch(model_tok, texts)

    # 3. Compute all metrics
    fert   = fertility_score(texts, token_lists)
    bpt    = bytes_per_token(texts, token_lists)
    unk    = unk_rate(token_lists, model_tok)
    cov    = vocab_coverage(model_tok, script_range)
    nsl    = nsl_score(texts, token_lists, ref_tok)
    avg_tl = avg_token_length(token_lists, model_tok)

    # Continuation rate is expensive (re-tokenizes word by word) — use a sample
    cont_texts = texts[:min(200, len(texts))]
    cont = continuation_rate(cont_texts, token_lists[:len(cont_texts)], model_tok)

    # Aggregate word/token counts
    total_words  = sum(len(get_words(t)) for t in texts)
    total_tokens = sum(len(toks) for toks in token_lists)

    # 4. Collect examples
    examples = []
    for text in texts[:n_examples]:
        sentence = text[:EXAMPLE_MAX_CHARS].replace("\n", " ")
        enc = model_tok(sentence, add_special_tokens=False)
        toks = model_tok.convert_ids_to_tokens(enc["input_ids"])
        examples.append((sentence, toks))

    result = LangResult(
        lang_code         = lang_code,
        lang_name         = lang_name,
        script            = script,
        n_docs            = len(texts),
        n_words           = total_words,
        n_tokens          = total_tokens,
        fertility         = fert,
        continuation_rate = cont,
        bytes_per_token   = bpt,
        unk_rate          = unk,
        vocab_coverage    = cov,
        nsl               = nsl,
        avg_token_len     = avg_tl,
        examples          = examples,
    )

    # Quick inline summary
    print(f"       Fertility={fert:.2f}  BPT={bpt:.2f}  "
          f"ContRate={cont:.1%}  VocabCov={cov:.1%}  NSL={nsl:.2f}")
    return result


def run_evaluation(
    model_id:    str = MODEL_ID,
    ref_id:      str = REF_MODEL_ID,
    subset:      str = SUBSET,
    n_samples:   int = SAMPLES_PER_LANG,
):
    # ── Load tokenizers ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Tokenizer Evaluation — Indic Languages (Sangraha/{subset})")
    print(f"{'='*70}")
    print(f"\n  Model      : {model_id}")
    print(f"  Reference  : {ref_id}")
    print(f"  Subset     : {subset}")
    print(f"  Samples    : {n_samples} docs/language")

    print("\n  Loading tokenizers...")
    model_tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    ref_tok   = AutoTokenizer.from_pretrained(ref_id,   trust_remote_code=True)

    vocab_size = model_tok.vocab_size
    print(f"  Model vocab size : {vocab_size:,}")
    print(f"  Ref   vocab size : {ref_tok.vocab_size:,}")

    # ── Select languages based on subset ─────────────────────────────────────
    lang_map = (SANGRAHA_VERIFIED_LANGS
                if subset == "verified"
                else SANGRAHA_UNVERIFIED_LANGS)

    # ── Run per-language evaluation ───────────────────────────────────────────
    print(f"\n  Evaluating {len(lang_map)} languages...\n")
    results: List[LangResult] = []

    for lang_code, lang_meta in lang_map.items():
        result = evaluate_language(
            lang_code  = lang_code,
            lang_meta  = lang_meta,
            model_tok  = model_tok,
            ref_tok    = ref_tok,
            subset     = subset,
            n_samples  = n_samples,
            n_examples = EXAMPLES_PER_LANG,
        )
        if result:
            results.append(result)

    return results, model_tok


# =============================================================================
# REPORTING
# =============================================================================

FERTILITY_BANDS = [
    (1.0, 1.5, "🟢 Excellent"),
    (1.5, 2.5, "🟡 Good"),
    (2.5, 4.0, "🟠 Marginal"),
    (4.0, 999, "🔴 Poor"),
]

def grade_fertility(f: float) -> str:
    for lo, hi, label in FERTILITY_BANDS:
        if lo <= f < hi:
            return label
    return "🔴 Poor"


def print_results_table(results: List[LangResult]):
    rows = []
    for r in sorted(results, key=lambda x: x.fertility):
        rows.append([
            r.lang_code,
            r.lang_name,
            r.script,
            f"{r.fertility:.2f}",
            f"{r.continuation_rate:.1%}",
            f"{r.bytes_per_token:.2f}",
            f"{r.unk_rate:.3%}",
            f"{r.vocab_coverage:.1%}",
            f"{r.nsl:.2f}",
            f"{r.avg_token_len:.2f}",
            grade_fertility(r.fertility),
        ])

    headers = [
        "Code", "Language", "Script",
        "Fertility↓", "ContRate↓", "BPT↑",
        "UNK↓", "VocabCov↑", "NSL↓",
        "AvgTokLen↑", "Grade"
    ]

    print(f"\n{'='*70}")
    print("  RESULTS (sorted by Fertility, lower = better)")
    print(f"{'='*70}\n")
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    # Summary stats
    ferties = [r.fertility for r in results]
    print(f"\n  ── Fertility Summary ──")
    print(f"  Best   : {min(ferties):.2f} ({results[ferties.index(min(ferties))].lang_name})")
    print(f"  Worst  : {max(ferties):.2f} ({results[ferties.index(max(ferties))].lang_name})")
    print(f"  Mean   : {sum(ferties)/len(ferties):.2f}")
    print(f"  Median : {sorted(ferties)[len(ferties)//2]:.2f}")

    counts = defaultdict(list)
    for r in results:
        counts[grade_fertility(r.fertility)].append(r.lang_name)
    print(f"\n  ── Grade Distribution ──")
    for lo, hi, label in FERTILITY_BANDS:
        langs = counts[label]
        if langs:
            print(f"  {label}: {', '.join(langs)}")


def print_examples(results: List[LangResult]):
    print(f"\n{'='*70}")
    print("  TOKENIZATION EXAMPLES")
    print(f"{'='*70}")
    for r in results:
        if not r.examples:
            continue
        print(f"\n  [{r.lang_code}] {r.lang_name}")
        print(f"  {'─'*60}")
        for i, (sentence, tokens) in enumerate(r.examples, 1):
            print(f"  Example {i}: {sentence}")
            # Display tokens with | separators, colour-coded by length
            display = " | ".join(tokens[:30])
            if len(tokens) > 30:
                display += f" ... (+{len(tokens)-30} more)"
            print(f"  Tokens   : {display}")
            print(f"  Count    : {len(tokens)} tokens for "
                  f"{len(get_words(sentence))} words "
                  f"(local fertility={len(tokens)/max(1,len(get_words(sentence))):.1f})")
            print()


def save_results(results: List[LangResult], model_id: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # CSV
    records = []
    for r in results:
        records.append({
            "lang_code":         r.lang_code,
            "lang_name":         r.lang_name,
            "script":            r.script,
            "n_docs":            r.n_docs,
            "n_words":           r.n_words,
            "n_tokens":          r.n_tokens,
            "fertility":         round(r.fertility, 4),
            "continuation_rate": round(r.continuation_rate, 4),
            "bytes_per_token":   round(r.bytes_per_token, 4),
            "unk_rate":          round(r.unk_rate, 6),
            "vocab_coverage":    round(r.vocab_coverage, 4),
            "nsl":               round(r.nsl, 4),
            "avg_token_len":     round(r.avg_token_len, 4),
            "grade":             grade_fertility(r.fertility),
            "model":             model_id,
        })
    df = pd.DataFrame(records)
    # Derive a safe short name from model_id for the filename
    model_short = model_id.replace("/", "_").replace("-", "_").lower()[:40]
    csv_path = os.path.join(output_dir, f"results_{model_short}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  ✓ CSV saved  → {csv_path}")

    # JSON (includes examples)
    json_records = []
    for r in results:
        rec = {k: v for k, v in records[results.index(r)].items()}
        rec["examples"] = [
            {"text": s, "tokens": toks[:20]}
            for s, toks in r.examples
        ]
        json_records.append(rec)
    json_path = os.path.join(output_dir, f"results_{model_short}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_records, f, ensure_ascii=False, indent=2)
    print(f"  ✓ JSON saved → {json_path}")

    return df


def print_metric_explanations():
    print("""
  ── Metric Guide ────────────────────────────────────────────────────────
  Fertility     Avg tokens per word. Lower = better. 1.0 is perfect.
                English GPT-4: ~1.2  |  Unseen scripts: 5–10+
  ContRate      % words split into ≥2 sub-tokens. Lower = better.
  BPT           Bytes per token. Higher = denser tokens, better compression.
                Indic chars = 3 UTF-8 bytes. Good: 6–12 bpt.
  UNK           % tokens that are <unk>. Should be 0 for BPE+byte-fallback.
  VocabCov      % of script's Unicode chars that appear in vocabulary.
                Low coverage → byte-level fallback → high fertility.
  NSL           Normalized Sequence Length vs. reference tokenizer (mT5).
                NSL < 1 = more efficient than reference. NSL > 1 = worse.
  AvgTokLen     Mean chars per token. Longer = richer vocabulary pieces.
  ─────────────────────────────────────────────────────────────────────────
""")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results, model_tok = run_evaluation(
        model_id  = MODEL_ID,
        ref_id    = REF_MODEL_ID,
        subset    = SUBSET,
        n_samples = SAMPLES_PER_LANG,
    )

    print_metric_explanations()
    print_results_table(results)

    if PRINT_EXAMPLES:
        print_examples(results)

    df = save_results(results, MODEL_ID, OUTPUT_DIR)

    print(f"\n{'='*70}")
    print("  EVALUATION COMPLETE")
    print(f"{'='*70}\n")
#!/usr/bin/env python3
"""
Download English data from FineWeb-edu and add to Sangraha training data.
This maintains the ratio by adding the same number of documents as each Indic language.
"""

from datasets import load_dataset

# Same limit as used for Indic languages
MAX_ENGLISH_DOCS = 200_000

def add_english_to_sangraha():
    print("Downloading English data from FineWeb-edu...")

    # Load FineWeb-edu dataset (educational English web pages)
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "CC-MAIN-2024-10",
        split="train",
        streaming=True
    )

    train_file = "sangraha_data/sangraha_unverified_train.txt"

    added_count = 0
    with open(train_file, "a", encoding="utf-8") as f:
        for row in ds:
            if added_count >= MAX_ENGLISH_DOCS:
                break

            text = row.get("text", "").strip()
            if len(text) > 20:  # Skip near-empty documents
                f.write(text + "\n")
                added_count += 1

            if added_count % 10000 == 0:
                print(f"  Added {added_count:,} English documents...")

    print(f"✓ Added {added_count:,} English documents to {train_file}")
    print("English data successfully added to Sangraha training set!")

if __name__ == "__main__":
    add_english_to_sangraha()
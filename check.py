import json

with open("indic_tokenizer_output/novel_tokens.json") as f:
    novel = json.load(f)

# Spot check: sample tokens by length
short  = [t for t in novel if len(t) <= 2]
medium = [t for t in novel if 3 <= len(t) <= 6]
long_  = [t for t in novel if len(t) > 6]

print(f"Short  (≤2 chars): {len(short):,}  → e.g. {short[:10]}")
print(f"Medium (3-6 chars): {len(medium):,} → e.g. {medium[:10]}")
print(f"Long   (>6 chars):  {len(long_):,}  → e.g. {long_[:10]}")

# Flag suspicious tokens (mixed script = bad)
import regex
mixed = [t for t in novel if 
         regex.search(r'\p{Latin}', t) and 
         regex.search(r'[\u0900-\u0D7F]', t)]
print(f"\nMixed-script tokens (should be ~0): {len(mixed)}")
print(mixed[:20])
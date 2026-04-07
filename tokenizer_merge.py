from transformers import AutoTokenizer
from tokenizers import Tokenizer

# Load Nemotron tokenizer
base_tok = AutoTokenizer.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    trust_remote_code=True
)

base_vocab = set(base_tok.get_vocab().keys())

# Load your tokenizer correctly
indic_tokenizer = Tokenizer.from_file(
    "indic_tokenizer_output/indic_bpe_tokenizer.json"
)

new_vocab = set(indic_tokenizer.get_vocab().keys())

# Compute novel tokens
novel_tokens = list(new_vocab - base_vocab)

print(f"New tokens to add: {len(novel_tokens)}")
print("First 20 tokens:", novel_tokens[:20])
# Add the novel tokens
base_tok.add_tokens(novel_tokens)

# Save the extended tokenizer
base_tok.save_pretrained("nemotron_indic_tokenizer")

print(f"Extended vocab size: {len(base_tok)}")
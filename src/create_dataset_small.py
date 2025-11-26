import argparse
import random
from transformers import AutoTokenizer
import numpy as np

# =======================
# Command-line arguments
# =======================
parser = argparse.ArgumentParser(description="Sample and filter prompts from a text file")
parser.add_argument("--prompts-file", type=str, default="data/test_dataset.txt")
parser.add_argument("--output-file", type=str, default="data/test_dataset_small.txt")
parser.add_argument("--num-requests", type=int, default=50000)
parser.add_argument("--model-name", type=str, default="microsoft/Phi-4-mini-instruct")
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--min-tokens", type=int, default=256)
parser.add_argument("--random-seed", type=int, default=42)
args = parser.parse_args()

# =======================
# Load prompts from text file
# =======================
print(f"Loading prompts from {args.prompts_file}...")
with open(args.prompts_file, "r", encoding="utf-8") as f:
    all_prompts = [line.strip() for line in f if line.strip()]

print(f"Total prompts loaded: {len(all_prompts)}")

# =======================
# Initialize tokenizer
# =======================
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

# =======================
# Filter prompts by token count
# =======================
filtered_prompts = [
    prompt for prompt in all_prompts
    if args.min_tokens <= len(tokenizer.encode(prompt)) <= args.max_tokens
]

print(
    f"Prompts after token filtering ({args.min_tokens}-{args.max_tokens}): {len(filtered_prompts)}"
)

# =======================
# Sample prompts
# =======================
random.seed(args.random_seed)
if len(filtered_prompts) > args.num_requests:
    prompts = random.sample(filtered_prompts, args.num_requests)
else:
    prompts = filtered_prompts  # take all if less than requested

print(f"Sampled {len(prompts)} prompts.")

# =======================
# Compute statistics
# =======================
char_counts = np.array([len(p) for p in prompts])
token_counts = np.array([len(tokenizer.encode(p)) for p in prompts])


def print_stats(name, arr):
    print(f"\n{name} statistics:")
    print(f"Min: {arr.min()}")
    print(f"Max: {arr.max()}")
    print(f"Avg: {arr.mean():.2f}")
    print(f"P90: {np.percentile(arr, 90)}")
    print(f"P95: {np.percentile(arr, 95)}")
    print(f"P98: {np.percentile(arr, 98)}")
    print(f"P99: {np.percentile(arr, 99)}")


print_stats("Character count", char_counts)
print_stats("Token count", token_counts)

# =======================
# Save prompts as plain text
# =======================
with open(args.output_file, "w", encoding="utf-8") as f:
    for prompt in prompts:
        f.write(prompt.replace("\n", " ") + "\n")

print(f"\nSaved sampled prompts to {args.output_file}")

"""
Build a Parquet dataset for GRPO RL fine-tuning with verl.

Reads train_sequences.txt, groups subdomains by root domain, splits each
domain's subdomains 80/20 into prompt seeds and held-out targets, then
writes a Parquet file with columns: prompt, root_domain, held_out.
"""

import argparse
import json
import os
import random
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import PreTrainedTokenizerFast


def parse_sequences(input_path: str) -> dict[str, list[str]]:
    """Parse train_sequences.txt and group subdomain prefixes by root domain.

    Each line has format: <bos> rootdomain.com <sep> prefix1 <sep> prefix2 ... <eos>
    Returns dict mapping root_domain -> list of all prefixes seen.
    """
    domains: dict[str, set[str]] = defaultdict(set)

    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Remove <bos> and <eos>
            line = line.replace("<bos>", "").replace("<eos>", "").strip()
            parts = [p.strip() for p in line.split("<sep>") if p.strip()]
            if len(parts) < 2:
                continue
            root_domain = parts[0]
            prefixes = parts[1:]
            for p in prefixes:
                domains[root_domain].add(p)

    return {k: sorted(v) for k, v in domains.items()}


def build_rl_dataset(
    input_path: str,
    output_path: str,
    tokenizer_dir: str,
    min_subs: int = 5,
    prompt_ratio: float = 0.8,
    seed: int = 42,
):
    """Build Parquet dataset for verl GRPO training.

    Args:
        input_path: Path to train_sequences.txt.
        output_path: Output Parquet file path.
        tokenizer_dir: Path to tokenizer (for registering chat template).
        min_subs: Minimum subdomains per domain to include.
        prompt_ratio: Fraction of subdomains used as prompt seeds (rest held out).
        seed: Random seed for reproducibility.
    """
    random.seed(seed)

    domains = parse_sequences(input_path)
    print(f"Parsed {len(domains)} unique root domains")

    # Filter to domains with enough subdomains
    eligible = {k: v for k, v in domains.items() if len(v) >= min_subs}
    print(f"Eligible domains (>= {min_subs} subs): {len(eligible)}")

    prompts = []
    root_domains = []
    held_outs = []

    for root_domain, prefixes in eligible.items():
        random.shuffle(prefixes)
        split_idx = max(1, int(len(prefixes) * prompt_ratio))
        seeds = prefixes[:split_idx]
        held_out = prefixes[split_idx:]

        if not held_out:
            # Move at least one to held-out
            held_out = [seeds.pop()]

        # Build prompt in raw format: <bos> domain.com <sep> seed1 <sep> seed2 <sep>
        prompt_parts = [f"<bos> {root_domain}"]
        for s in seeds:
            prompt_parts.append(f"<sep> {s}")
        prompt_parts.append("<sep>")
        prompt = " ".join(prompt_parts)

        prompts.append(prompt)
        root_domains.append(root_domain)
        held_outs.append(json.dumps(held_out))

    # Write Parquet
    table = pa.table({
        "prompt": prompts,
        "root_domain": root_domains,
        "held_out": held_outs,
    })
    pq.write_table(table, output_path)
    print(f"Written {len(prompts)} examples to {output_path}")

    # Register no-op chat template on tokenizer so verl passes raw prompts through
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    if not tokenizer.chat_template:
        tokenizer.chat_template = "{{ messages[0]['content'] }}"
        tokenizer.save_pretrained(tokenizer_dir)
        print(f"Registered no-op chat template on {tokenizer_dir}")

    return len(prompts)


def main():
    parser = argparse.ArgumentParser(description="Build RL dataset for GRPO fine-tuning")
    parser.add_argument("--input", type=str, default="data/train_sequences.txt",
                        help="Path to train_sequences.txt")
    parser.add_argument("--output", type=str, default="data/rl_train.parquet",
                        help="Output Parquet file")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer",
                        help="Path to tokenizer directory")
    parser.add_argument("--min-subs", type=int, default=5,
                        help="Minimum subdomains per domain")
    parser.add_argument("--prompt-ratio", type=float, default=0.8,
                        help="Fraction of subs used as prompt seeds")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_rl_dataset(
        args.input, args.output, args.tokenizer_dir,
        args.min_subs, args.prompt_ratio, args.seed,
    )


if __name__ == "__main__":
    main()

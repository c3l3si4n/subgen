"""
Raw CSV -> grouped sequences for language model training.

Pipeline:
1. Read FQDNs, strip trailing dots
2. Extract root domain via tldextract
3. Group subdomains by root domain
4. Build sequences: <bos> root <sep> prefix1 <sep> prefix2 ... <eos>
5. Split large groups (~40 subs per sequence)
6. Train/val split by deterministic hash on root domain (98/2)
"""

import argparse
import hashlib
import os
import random
from collections import defaultdict
from pathlib import Path

import tldextract
from tqdm import tqdm

BOS = "<bos>"
EOS = "<eos>"
SEP = "<sep>"
SUBS_PER_SEQ = 40
VAL_RATIO = 0.02


def is_val_domain(root_domain: str) -> bool:
    """Deterministic hash-based train/val split on root domain."""
    h = hashlib.md5(root_domain.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF < VAL_RATIO


def extract_root_and_sub(fqdn: str) -> tuple[str, str, str] | None:
    """Extract root domain and subdomain prefix.

    Returns (root_domain, prefix, fqdn) or None if no subdomain.
    e.g. "www.example.com" -> ("example.com", "www", "www.example.com")
         "a.b.example.com" -> ("example.com", "a.b", "a.b.example.com")
    """
    ext = tldextract.extract(fqdn)
    if not ext.domain or not ext.suffix:
        return None
    root = f"{ext.domain}.{ext.suffix}"
    if not ext.subdomain:
        return None
    return root, ext.subdomain, fqdn


def build_sequence(root_domain: str, subdomains: list[str]) -> str:
    """Build a training sequence from root domain and its subdomains."""
    parts = [BOS, root_domain]
    for sub in subdomains:
        parts.append(SEP)
        parts.append(sub)
    parts.append(EOS)
    return " ".join(parts)


def chunk_subdomains(subdomains: list[str], chunk_size: int = SUBS_PER_SEQ) -> list[list[str]]:
    """Split subdomains into chunks, shuffling for training diversity."""
    subs = list(subdomains)
    random.shuffle(subs)
    return [subs[i:i + chunk_size] for i in range(0, len(subs), chunk_size)]


def preprocess(
    input_path: str,
    output_dir: str,
    seed: int = 42,
    max_lines: int | None = None,
    max_subs_per_domain: int | None = None,
):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train_sequences.txt")
    val_path = os.path.join(output_dir, "val_sequences.txt")

    # Phase 1: Group subdomains by root domain
    print("Phase 1: Grouping subdomains by root domain...")
    groups: dict[str, set[str]] = defaultdict(set)

    with open(input_path, "r") as f:
        for i, line in enumerate(tqdm(f, desc="Reading domains")):
            if max_lines and i >= max_lines:
                break
            fqdn = line.strip().rstrip(".")
            if not fqdn:
                continue
            result = extract_root_and_sub(fqdn)
            if result is None:
                continue
            root, prefix, _ = result
            groups[root].add(prefix)

    print(f"Found {len(groups)} unique root domains")
    total_subs = sum(len(v) for v in groups.values())
    print(f"Total subdomains: {total_subs}")

    # Apply per-domain frequency cap
    if max_subs_per_domain is not None:
        capped = 0
        for root in groups:
            subs = groups[root]
            if len(subs) > max_subs_per_domain:
                groups[root] = set(random.sample(sorted(subs), max_subs_per_domain))
                capped += 1
        capped_total = sum(len(v) for v in groups.values())
        print(f"Capped {capped} domains to {max_subs_per_domain} subs each")
        print(f"Total subdomains after cap: {capped_total}")

    # Phase 2: Build sequences and write to train/val files
    print("Phase 2: Building sequences...")
    train_count = 0
    val_count = 0

    with open(train_path, "w") as f_train, open(val_path, "w") as f_val:
        for root, subs in tqdm(groups.items(), desc="Building sequences"):
            chunks = chunk_subdomains(list(subs))
            is_val = is_val_domain(root)
            f_out = f_val if is_val else f_train

            for chunk in chunks:
                seq = build_sequence(root, chunk)
                f_out.write(seq + "\n")
                if is_val:
                    val_count += 1
                else:
                    train_count += 1

    print(f"Train sequences: {train_count}")
    print(f"Val sequences: {val_count}")
    print(f"Output written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess domain data for LM training")
    parser.add_argument("--input", type=str, default="dataset/domains_export.csv",
                        help="Path to raw domain CSV")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for sequences")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-lines", type=int, default=None,
                        help="Max lines to process (for debugging)")
    parser.add_argument("--max-subs-per-domain", type=int, default=None,
                        help="Cap subdomains per root domain (randomly samples if exceeded)")
    args = parser.parse_args()

    preprocess(args.input, args.output_dir, args.seed, args.max_lines, args.max_subs_per_domain)


if __name__ == "__main__":
    main()

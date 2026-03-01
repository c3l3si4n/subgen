"""
Raw CSV -> grouped sequences for language model training.

Pipeline:
1. Read FQDNs, extract root domain + subdomain prefix to temp file
2. External sort by root domain (handles files larger than RAM)
3. Stream sorted pairs, group by root domain, build sequences
4. Train/val split by deterministic hash on root domain (98/2)

Uses constant memory regardless of input size.
"""

import argparse
import hashlib
import os
import random
import subprocess
import tempfile

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


def extract_root_and_sub(fqdn: str) -> tuple[str, str] | None:
    """Extract root domain and subdomain prefix.

    Returns (root_domain, prefix) or None if no subdomain.
    """
    ext = tldextract.extract(fqdn)
    if not ext.domain or not ext.suffix or not ext.subdomain:
        return None
    root = f"{ext.domain}.{ext.suffix}"
    return root, ext.subdomain


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
    sort_tmp_dir: str | None = None,
    sort_buffer_size: str = "4G",
    sort_parallel: int = 4,
):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train_sequences.txt")
    val_path = os.path.join(output_dir, "val_sequences.txt")

    # Use output_dir for sort temp files if not specified
    tmp_dir = sort_tmp_dir or output_dir

    # Phase 1: Extract root/prefix pairs to temp file (streaming, constant memory)
    print("Phase 1: Extracting root/prefix pairs...")
    pairs_path = os.path.join(tmp_dir, "_pairs.tsv")

    with open(input_path, "r") as f_in, open(pairs_path, "w") as f_out:
        for i, line in enumerate(tqdm(f_in, desc="Extracting")):
            if max_lines and i >= max_lines:
                break
            fqdn = line.strip().rstrip(".")
            if not fqdn:
                continue
            result = extract_root_and_sub(fqdn)
            if result is None:
                continue
            root, prefix = result
            f_out.write(f"{root}\t{prefix}\n")

    # Phase 2: Sort by root domain and deduplicate (external sort, handles any file size)
    print("Phase 2: Sorting and deduplicating...")
    sorted_path = os.path.join(tmp_dir, "_sorted.tsv")

    sort_cmd = [
        "sort",
        "-t", "\t",          # tab delimiter
        "-k1,1",             # sort by root domain (for grouping)
        f"--buffer-size={sort_buffer_size}",
        f"--parallel={sort_parallel}",
        f"--temporary-directory={tmp_dir}",
        "-o", sorted_path,
        pairs_path,
    ]
    subprocess.run(sort_cmd, check=True)

    # Remove unsorted pairs file
    os.unlink(pairs_path)

    # Phase 3: Stream sorted file, group by root domain, write sequences
    print("Phase 3: Building sequences...")
    train_count = 0
    val_count = 0
    total_subs = 0
    total_roots = 0

    with (
        open(sorted_path, "r") as f_in,
        open(train_path, "w") as f_train,
        open(val_path, "w") as f_val,
    ):
        current_root = None
        current_subs: set[str] = set()

        def flush_group():
            nonlocal train_count, val_count, total_subs, total_roots
            if current_root is None:
                return

            total_roots += 1
            subs = list(current_subs)

            # Skip domains with fewer than 2 subdomains
            if len(subs) < 2:
                return

            # Apply per-domain cap
            if max_subs_per_domain and len(subs) > max_subs_per_domain:
                subs = random.sample(subs, max_subs_per_domain)

            total_subs += len(subs)
            is_val = is_val_domain(current_root)
            f_out = f_val if is_val else f_train

            for chunk in chunk_subdomains(subs):
                seq = build_sequence(current_root, chunk)
                f_out.write(seq + "\n")
                if is_val:
                    val_count += 1
                else:
                    train_count += 1

        for line in tqdm(f_in, desc="Building sequences"):
            root, prefix = line.rstrip("\n").split("\t", 1)
            if root != current_root:
                flush_group()
                current_root = root
                current_subs = set()
            current_subs.add(prefix)

        flush_group()  # flush last group

    # Clean up sorted file
    os.unlink(sorted_path)

    print(f"Found {total_roots} unique root domains")
    print(f"Total subdomains: {total_subs}")
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
    parser.add_argument("--sort-tmp-dir", type=str, default=None,
                        help="Temp directory for external sort (default: output-dir)")
    parser.add_argument("--sort-buffer-size", type=str, default="4G",
                        help="Memory buffer for external sort (default: 4G)")
    parser.add_argument("--sort-parallel", type=int, default=4,
                        help="Parallel sort threads (default: 4)")
    args = parser.parse_args()

    preprocess(
        args.input, args.output_dir, args.seed, args.max_lines,
        args.max_subs_per_domain, args.sort_tmp_dir,
        args.sort_buffer_size, args.sort_parallel,
    )


if __name__ == "__main__":
    main()

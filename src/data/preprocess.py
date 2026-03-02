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
import multiprocessing
import os
import random
import subprocess

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


class FastTLDExtractor:
    """Fast domain splitter. Loads tldextract's suffix list once, then uses
    pure string ops for O(1) lookups per FQDN — ~10-50x faster than
    calling tldextract.extract() per line.
    """

    def __init__(self):
        # Build suffix set from tldextract's bundled snapshot
        self._suffixes: set[str] = set()
        snapshot = tldextract.suffix_list.get_suffix_lists(
            cache=tldextract.cache.DiskCache(None),
            urls=(),
            cache_fetch_timeout=None,
            fallback_to_snapshot=True,
        )
        for suffix_list in snapshot:
            for suffix in suffix_list:
                self._suffixes.add(suffix)
        print(f"Loaded {len(self._suffixes)} public suffixes")

    def extract(self, fqdn: str) -> tuple[str, str] | None:
        """Extract (root_domain, subdomain_prefix) from FQDN. Returns None if no subdomain."""
        labels = fqdn.lower().split(".")
        n = len(labels)
        if n < 3:
            return None

        # Find longest matching suffix (try from longer to shorter)
        for i in range(max(0, n - 4), n - 1):
            candidate = ".".join(labels[i:])
            if candidate in self._suffixes:
                domain_idx = i - 1
                if domain_idx < 0:
                    return None
                root = f"{labels[domain_idx]}.{candidate}"
                subdomain = ".".join(labels[:domain_idx])
                if not subdomain:
                    return None
                return root, subdomain

        # Fallback: assume last label is TLD
        suffix = labels[-1]
        domain = labels[-2]
        subdomain = ".".join(labels[:-2])
        if not subdomain or not domain or not suffix:
            return None
        return f"{domain}.{suffix}", subdomain


_worker_suffixes: set[str] | None = None


def _init_worker(suffixes: set[str]):
    """Initialize per-worker suffix set (avoids pickling the extractor)."""
    global _worker_suffixes
    _worker_suffixes = suffixes


def _extract_batch(lines: list[str]) -> list[str]:
    """Process a batch of FQDNs, return TSV lines 'root\\tprefix'."""
    suffixes = _worker_suffixes
    out = []
    for line in lines:
        fqdn = line.strip().rstrip(".")
        if not fqdn:
            continue
        labels = fqdn.lower().split(".")
        n = len(labels)
        if n < 3:
            continue

        # Find longest matching suffix
        root = subdomain = None
        for i in range(max(0, n - 4), n - 1):
            candidate = ".".join(labels[i:])
            if candidate in suffixes:
                domain_idx = i - 1
                if domain_idx < 0:
                    break
                root = f"{labels[domain_idx]}.{candidate}"
                subdomain = ".".join(labels[:domain_idx])
                break

        if root is None:
            # Fallback: assume last label is TLD
            root = f"{labels[-2]}.{labels[-1]}"
            subdomain = ".".join(labels[:-2])

        if subdomain:
            out.append(f"{root}\t{subdomain}\n")
    return out


def build_sequence(root_domain: str, subdomains: list[str]) -> str:
    """Build a training sequence from root domain and its subdomains."""
    parts = [BOS, root_domain]
    for sub in subdomains:
        parts.append(SEP)
        parts.append(sub)
    parts.append(EOS)
    return " ".join(parts)


def chunk_subdomains(
    subdomains: list[str],
    chunk_size: int = SUBS_PER_SEQ,
    orderings: list[str] | None = None,
) -> list[list[str]]:
    """Split subdomains into chunks with multiple orderings for diversity.

    Each ordering strategy produces its own set of chunks. This teaches the
    model that subdomain order is arbitrary and exposes it to more contexts.

    Args:
        orderings: List of ordering strategies to apply. Options:
            "random" — random shuffle (original behavior)
            "alpha"  — alphabetical order
            "reverse" — reverse alphabetical
            Default: ["random", "alpha", "reverse"]
    """
    if orderings is None:
        orderings = ["random", "alpha", "reverse"]

    all_chunks = []
    for ordering in orderings:
        subs = list(subdomains)
        if ordering == "random":
            random.shuffle(subs)
        elif ordering == "alpha":
            subs.sort()
        elif ordering == "reverse":
            subs.sort(reverse=True)
        all_chunks.extend(subs[i:i + chunk_size] for i in range(0, len(subs), chunk_size))

    return all_chunks


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

    # Phase 1: Extract root/prefix pairs to temp file (parallel across CPU cores)
    print("Phase 1: Extracting root/prefix pairs...")
    extractor = FastTLDExtractor()
    pairs_path = os.path.join(tmp_dir, "_pairs.tsv")

    num_workers = min(multiprocessing.cpu_count(), 16)
    batch_size = 10_000
    print(f"Using {num_workers} workers, batch size {batch_size}")

    def read_batches(path, max_lines=None):
        batch = []
        count = 0
        with open(path, "r", buffering=8 * 1024 * 1024) as f:
            for line in f:
                if max_lines and count >= max_lines:
                    break
                batch.append(line)
                count += 1
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    with open(pairs_path, "w", buffering=8 * 1024 * 1024) as f_out, \
         multiprocessing.Pool(num_workers, initializer=_init_worker,
                              initargs=(extractor._suffixes,)) as pool:
        write = f_out.write
        total = 0
        for result_batch in pool.imap(
            _extract_batch,
            tqdm(read_batches(input_path, max_lines), desc="Extracting",
                 unit="batch"),
            chunksize=4,
        ):
            for tsv_line in result_batch:
                write(tsv_line)
            total += len(result_batch)
    print(f"Extracted {total} pairs")

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

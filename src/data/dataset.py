"""
Pre-tokenize sequences into numpy memmap files and provide a PyTorch Dataset.

Supports two modes:
- Standard: one sequence per row, padded to max_seq_len (original behavior)
- Packed: greedy bin-packing of multiple sequences into rows separated by <eos><bos>
  boundaries, eliminating ~70% padding waste.

Produces data/train.bin and data/val.bin (uint16 memmap, 512 tokens per row).
Zero-copy reads from disk — no RAM pressure at scale.
"""

import argparse
import multiprocessing
import os
import random
import shutil
import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

MAX_SEQ_LEN = 512

_worker_tokenizer: PreTrainedTokenizerFast | None = None
_worker_max_seq_len: int = MAX_SEQ_LEN


def _init_tok_worker(tokenizer_dir: str, max_seq_len: int):
    global _worker_tokenizer, _worker_max_seq_len
    _worker_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    _worker_max_seq_len = max_seq_len


def _tokenize_batch(lines: list[str]) -> list[list[int]]:
    """Tokenize a batch of sequence lines."""
    results = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        token_ids = _worker_tokenizer.encode(line)
        if len(token_ids) > _worker_max_seq_len:
            token_ids = token_ids[:_worker_max_seq_len]
        results.append(token_ids)
    return results


def pretokenize(
    input_path: str,
    output_path: str,
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int = MAX_SEQ_LEN,
):
    """Tokenize all sequences and write to a flat memmap file (one per row, padded)."""
    print(f"Counting sequences in {input_path}...")
    num_seqs = 0
    with open(input_path, "r") as f:
        for _ in f:
            num_seqs += 1
    print(f"Found {num_seqs} sequences")

    mmap = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=(num_seqs, max_seq_len))
    pad_id = tokenizer.pad_token_id

    print(f"Tokenizing to {output_path}...")
    with open(input_path, "r") as f:
        for i, line in enumerate(tqdm(f, total=num_seqs, desc="Tokenizing")):
            line = line.strip()
            if not line:
                mmap[i] = pad_id
                continue
            token_ids = tokenizer.encode(line)
            if len(token_ids) > max_seq_len:
                token_ids = token_ids[:max_seq_len]
            padded = token_ids + [pad_id] * (max_seq_len - len(token_ids))
            mmap[i] = np.array(padded, dtype=np.uint16)

    mmap.flush()
    print(f"Written {num_seqs} sequences to {output_path}")
    return num_seqs


def pretokenize_packed(
    input_path: str,
    output_path: str,
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int = MAX_SEQ_LEN,
    seed: int = 42,
):
    """Tokenize sequences in parallel, then greedily bin-pack into fixed-width rows."""
    pad_id = tokenizer.pad_token_id

    # Save tokenizer to temp dir for worker processes
    tmp_tok_dir = tempfile.mkdtemp(prefix="tok_")
    tokenizer.save_pretrained(tmp_tok_dir)

    num_workers = min(multiprocessing.cpu_count(), 16)
    batch_size = 5_000
    print(f"Tokenizing with {num_workers} workers...")

    def read_batches(path):
        batch = []
        with open(path, "r", buffering=8 * 1024 * 1024) as f:
            for line in f:
                batch.append(line)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    # Parallel tokenization into memory
    all_token_ids = []
    with multiprocessing.Pool(num_workers, initializer=_init_tok_worker,
                              initargs=(tmp_tok_dir, max_seq_len)) as pool:
        for result_batch in pool.imap(
            _tokenize_batch,
            tqdm(read_batches(input_path), desc="Tokenizing", unit="batch"),
            chunksize=4,
        ):
            all_token_ids.extend(result_batch)

    shutil.rmtree(tmp_tok_dir, ignore_errors=True)
    print(f"Tokenized {len(all_token_ids)} sequences")

    # Shuffle for diversity before packing
    random.seed(seed)
    random.shuffle(all_token_ids)

    # Greedy bin-packing
    rows = []
    current_row = []
    current_len = 0

    for token_ids in tqdm(all_token_ids, desc="Packing"):
        seq_len = len(token_ids)

        if current_len + seq_len <= max_seq_len:
            current_row.extend(token_ids)
            current_len += seq_len
        else:
            if current_row:
                padded = current_row + [pad_id] * (max_seq_len - current_len)
                rows.append(padded)

            if seq_len <= max_seq_len:
                current_row = list(token_ids)
                current_len = seq_len
            else:
                current_row = list(token_ids[:max_seq_len])
                current_len = max_seq_len

    if current_row:
        padded = current_row + [pad_id] * (max_seq_len - current_len)
        rows.append(padded)

    num_rows = len(rows)
    non_pad = sum(max_seq_len - row.count(pad_id) if isinstance(row, list) else max_seq_len for row in rows)

    print(f"Packed {len(all_token_ids)} sequences into {num_rows} rows")
    print(f"Avg sequences per row: {len(all_token_ids) / num_rows:.1f}")
    print(f"Packing efficiency: {non_pad / (num_rows * max_seq_len) * 100:.1f}%")

    # Write to memmap
    mmap = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=(num_rows, max_seq_len))
    for i, row in enumerate(tqdm(rows, desc="Writing")):
        mmap[i] = np.array(row, dtype=np.uint16)
    mmap.flush()

    print(f"Written {num_rows} packed rows to {output_path}")
    return num_rows


class DomainDataset(Dataset):
    """PyTorch Dataset backed by a numpy memmap file."""

    def __init__(self, bin_path: str, max_seq_len: int = MAX_SEQ_LEN, pad_token_id: int = 0):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.num_seqs = len(self.data) // max_seq_len
        self.data = self.data.reshape(self.num_seqs, max_seq_len)
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return self.num_seqs

    def __getitem__(self, idx):
        tokens = torch.from_numpy(self.data[idx].astype(np.int64))
        attention_mask = (tokens != self.pad_token_id).long()
        labels = tokens.clone()
        labels[labels == self.pad_token_id] = -100
        return {
            "input_ids": tokens,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize sequences into binary format")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer",
                        help="Path to trained tokenizer")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing sequence files and for output bins")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--packed", action="store_true", default=False,
                        help="Use greedy bin-packing to pack multiple sequences per row")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffle before packing")
    args = parser.parse_args()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    tokenize_fn = pretokenize_packed if args.packed else pretokenize

    for split in ["train", "val"]:
        input_path = os.path.join(args.data_dir, f"{split}_sequences.txt")
        output_path = os.path.join(args.data_dir, f"{split}.bin")
        if os.path.exists(input_path):
            if args.packed:
                tokenize_fn(input_path, output_path, tokenizer, args.max_seq_len, args.seed)
            else:
                tokenize_fn(input_path, output_path, tokenizer, args.max_seq_len)
        else:
            print(f"Skipping {split}: {input_path} not found")


if __name__ == "__main__":
    main()

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
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

MAX_SEQ_LEN = 512


def pretokenize(
    input_path: str,
    output_path: str,
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int = MAX_SEQ_LEN,
):
    """Tokenize all sequences and write to a flat memmap file (one per row, padded)."""
    # First pass: count sequences
    print(f"Counting sequences in {input_path}...")
    num_seqs = 0
    with open(input_path, "r") as f:
        for _ in f:
            num_seqs += 1
    print(f"Found {num_seqs} sequences")

    # Create memmap
    mmap = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=(num_seqs, max_seq_len))

    pad_id = tokenizer.pad_token_id

    # Second pass: tokenize and write
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

            # Pad to max_seq_len
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
    """Tokenize sequences and greedily bin-pack into fixed-width rows.

    Multiple sequences are concatenated into each row separated by natural
    <eos><bos> boundaries (already present in the sequence text). Remainder
    tokens in each row are padded.
    """
    pad_id = tokenizer.pad_token_id

    # Tokenize all sequences into memory
    print(f"Tokenizing sequences from {input_path}...")
    all_token_ids = []
    with open(input_path, "r") as f:
        for line in tqdm(f, desc="Tokenizing"):
            line = line.strip()
            if not line:
                continue
            token_ids = tokenizer.encode(line)
            if len(token_ids) > max_seq_len:
                token_ids = token_ids[:max_seq_len]
            all_token_ids.append(token_ids)

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
            # Fits in current row
            current_row.extend(token_ids)
            current_len += seq_len
        else:
            # Finalize current row
            if current_row:
                padded = current_row + [pad_id] * (max_seq_len - current_len)
                rows.append(padded)

            # Start new row with this sequence
            if seq_len <= max_seq_len:
                current_row = list(token_ids)
                current_len = seq_len
            else:
                # Sequence too long even alone — truncate
                current_row = list(token_ids[:max_seq_len])
                current_len = max_seq_len

    # Don't forget the last row
    if current_row:
        padded = current_row + [pad_id] * (max_seq_len - current_len)
        rows.append(padded)

    num_rows = len(rows)
    total_tokens = sum(len(ids) for ids in all_token_ids)
    useful_tokens = num_rows * max_seq_len
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

        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = (tokens != self.pad_token_id).long()

        # Labels: mask padding positions with -100 so they're ignored in loss
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

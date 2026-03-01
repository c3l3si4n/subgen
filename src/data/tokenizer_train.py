"""
Train a Byte-Level BPE tokenizer on preprocessed domain sequences.

Produces a HuggingFace-compatible tokenizer saved to tokenizer/ directory.
Special tokens: <pad>=0, <bos>=1, <eos>=2, <sep>=3
"""

import argparse
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.normalizers import Lowercase
from transformers import PreTrainedTokenizerFast


def _line_iterator(input_files: list[str]):
    """Yield lines from input files without loading them into memory."""
    for path in input_files:
        with open(path, "r") as f:
            for line in f:
                yield line.rstrip("\n")


def train_tokenizer(
    input_files: list[str],
    output_dir: str,
    vocab_size: int = 8192,
    min_frequency: int = 2,
):
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<pad>", "<bos>", "<eos>", "<sep>"]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=special_tokens,
    )

    print(f"Training tokenizer on {len(input_files)} file(s)...")
    tokenizer.train_from_iterator(_line_iterator(input_files), trainer=trainer)
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    # Wrap as PreTrainedTokenizerFast for HuggingFace Trainer compatibility
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        sep_token="<sep>",
    )

    wrapped.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir}")

    # Verification
    test_seq = "<bos> example.com <sep> www.example.com <sep> mail.example.com <eos>"
    encoded = wrapped.encode(test_seq)
    decoded = wrapped.decode(encoded)
    print(f"\nVerification:")
    print(f"  Input:   {test_seq}")
    print(f"  Tokens:  {encoded}")
    print(f"  Decoded: {decoded}")
    print(f"  Special token IDs: pad={wrapped.pad_token_id}, bos={wrapped.bos_token_id}, "
          f"eos={wrapped.eos_token_id}, sep={wrapped.sep_token_id}")

    return wrapped


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on domain sequences")
    parser.add_argument("--input", type=str, nargs="+",
                        default=["data/train_sequences.txt"],
                        help="Input sequence files")
    parser.add_argument("--output-dir", type=str, default="tokenizer",
                        help="Output directory for tokenizer")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    train_tokenizer(args.input, args.output_dir, args.vocab_size, args.min_frequency)


if __name__ == "__main__":
    main()

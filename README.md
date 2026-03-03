# subgen

A language model that generates subdomain candidates for a given root domain. Built on a ~152M parameter Llama architecture trained on real-world subdomain data.

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

### 1. Preprocess raw domains

Takes a CSV/text file of FQDNs, groups subdomains by root domain, and builds training sequences.

```bash
PYTHONPATH=src python -m data.preprocess --input dataset/domains_export.csv --output-dir data
```

Sequence format: `<bos>example.com<sep>www<sep>mail<sep>cdn.assets<eos>`

Subdomains are stored as prefixes only (no base domain), which keeps sequences short and prevents the model from hallucinating base domains during generation.

Large domain groups are chunked (~40 subdomains per sequence). Train/val split is 98/2 by deterministic hash on root domain.

### 2. Train tokenizer

Trains a character-level BPE tokenizer (vocab size 4096, DNS alphabet) on the preprocessed sequences.

```bash
PYTHONPATH=src python -m data.tokenizer_train --input data/train_sequences.txt --output-dir tokenizer
```

### 3. Pre-tokenize to binary

Converts text sequences into memory-mapped numpy arrays for zero-copy training.

```bash
PYTHONPATH=src python -m data.dataset --tokenizer-dir tokenizer --data-dir data
```

Produces `data/train.bin` and `data/val.bin` (uint16, 512 tokens per sequence).

### 4. Train

```bash
PYTHONPATH=src python -m train
```

Key arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--per-device-batch-size` | 32 | Batch size per GPU |
| `--gradient-accumulation-steps` | 4 | Gradient accumulation steps |
| `--lr` | 6e-4 | Peak learning rate |
| `--num-epochs` | 1 | Number of training epochs |
| `--save-steps` | 5000 | Checkpoint save interval |
| `--eval-steps` | 1000 | Evaluation interval |
| `--no-wandb` | off | Disable W&B logging |

Training auto-resumes from the latest checkpoint in `checkpoints/` if one exists.

For 16GB VRAM GPUs, reduce batch size:

```bash
PYTHONPATH=src python -m train --per-device-batch-size 16 --gradient-accumulation-steps 8
```

### 5. Generate subdomains

```bash
PYTHONPATH=src python -m generate --model-path checkpoints/final --domain example.com
```

Feed in a wordlist from subfinder (or any tool that outputs one FQDN per line):

```bash
subfinder -d example.com -o subs.txt
PYTHONPATH=src python -m generate --model-path checkpoints/final --wordlist subs.txt
```

Or condition on known prefixes manually:

```bash
PYTHONPATH=src python -m generate \
    --model-path checkpoints/final \
    --domain example.com \
    --known www mail cdn \
    --num-samples 10 \
    --temperature 0.8
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | required | Path to trained model |
| `--domain` | — | Target root domain (inferred from wordlist if not set) |
| `--wordlist` | — | Path to subdomain wordlist (e.g. subfinder output) |
| `--known` | — | Known subdomain prefixes to condition on |
| `--num-samples` | 5 | Number of generation passes |
| `--batch-size` | 16 | Sequences per generation sample |
| `--temperature` | 0.9 | Sampling temperature |
| `--min-p` | 0.05 | Min-p sampling threshold |
| `--repetition-penalty` | 1.3 | Repetition penalty for diverse output |
| `--no-temperature-sweep` | off | Use fixed temperature instead of sweeping |
| `--max-new-tokens` | 512 | Max tokens to generate per sample |

## Model

- **Architecture**: Llama (RoPE, RMSNorm, SwiGLU, GQA)
- **Parameters**: ~152M
- **Hidden size**: 1024
- **Layers**: 12
- **Attention heads**: 16 (4 KV heads)
- **Context length**: 1024 tokens
- **Vocab size**: 4096

## Project structure

```
subgen/
├── dataset/
│   └── domains_export.csv      # Raw domain data
├── data/
│   ├── train_sequences.txt     # Preprocessed training sequences
│   ├── val_sequences.txt       # Preprocessed validation sequences
│   ├── train.bin               # Tokenized training data (memmap)
│   └── val.bin                 # Tokenized validation data (memmap)
├── tokenizer/                  # Trained character-level BPE tokenizer
├── checkpoints/                # Training checkpoints
├── src/
│   ├── data/
│   │   ├── preprocess.py       # Raw CSV -> text sequences
│   │   ├── dataset.py          # Text -> binary + PyTorch Dataset
│   │   └── tokenizer_train.py  # BPE tokenizer training
│   ├── model/
│   │   └── config.py           # Llama model configuration
│   ├── train.py                # Training entrypoint
│   └── generate.py             # Inference / generation
└── requirements.txt
```

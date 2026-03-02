"""
Inference / generation for subdomain candidates.

Prompt format: <bos> targetdomain.com <sep> prefix1 <sep> prefix2 <sep>
Model generates subdomain prefixes, base domain is appended at output.
"""

import argparse
import random
import re
import sys
from collections import Counter

import torch
import tldextract
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast


def load_model(
    model_path: str,
    device: str = "auto",
) -> tuple[LlamaForCausalLM, PreTrainedTokenizerFast]:
    """Load model and tokenizer from a checkpoint directory."""
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


def build_prompt(
    root_domain: str,
    known_prefixes: list[str] | None = None,
) -> str:
    """Build a generation prompt from root domain and known subdomain prefixes."""
    parts = ["<bos>", root_domain]
    if known_prefixes:
        for prefix in known_prefixes:
            parts.append("<sep>")
            parts.append(prefix)
    parts.append("<sep>")
    return " ".join(parts)


def parse_prefixes(text: str, root_domain: str) -> list[str]:
    """Extract subdomain prefixes from generated text."""
    parts = text.split("<sep>")
    candidates = []
    for part in parts:
        part = part.strip()
        part = part.replace("<bos>", "").replace("<eos>", "").strip()
        if not part:
            continue
        # Skip the root domain itself
        if part == root_domain:
            continue
        # Validate prefix format: alphanumeric, hyphens, dots (for multi-level like a.b)
        if re.match(r'^[a-z0-9]([a-z0-9\-\.]*[a-z0-9])?$', part):
            candidates.append(part)
    return candidates


def _fit_prefixes_to_context(
    root_domain: str,
    known_prefixes: list[str],
    tokenizer: PreTrainedTokenizerFast,
    max_context: int = 512,
    reserve_for_generation: int = 384,
) -> list[str]:
    """Sample a random subset of known prefixes that fits within context."""
    max_prompt_tokens = max_context - reserve_for_generation

    subs = list(known_prefixes)
    random.shuffle(subs)

    base = f"<bos> {root_domain} <sep>"
    base_len = len(tokenizer.encode(base))

    selected = []
    current_len = base_len
    for prefix in subs:
        addition = f" <sep> {prefix}"
        add_len = len(tokenizer.encode(addition))
        if current_len + add_len > max_prompt_tokens:
            break
        selected.append(prefix)
        current_len += add_len

    return selected


@torch.no_grad()
def generate_subdomains(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    root_domain: str,
    known_prefixes: list[str] | None = None,
    num_samples: int = 5,
    batch_size: int = 16,
    temperature: float = 0.9,
    min_p: float = 0.05,
    repetition_penalty: float = 1.1,
    max_new_tokens: int = 384,
    temperature_sweep: bool = True,
) -> list[str]:
    """Generate subdomain candidates for a target domain.

    Returns full FQDNs (prefix + root_domain). Each sample uses a different
    random subset of known prefixes that fits within context. Within each
    sample, batch_size sequences are generated in parallel.

    Uses min-p sampling (context-adaptive candidate filtering) instead of
    top-p/top-k. When temperature_sweep is True, each sample uses a different
    temperature from conservative (0.6) to exploratory (1.0).
    """
    all_prefixes = set()
    known = known_prefixes or []

    if temperature_sweep and num_samples > 1:
        # Sweep from conservative to exploratory across samples
        temps = [
            0.6 + (1.0 - 0.6) * i / (num_samples - 1)
            for i in range(num_samples)
        ]
    else:
        temps = [temperature] * num_samples

    for i in range(num_samples):
        if known:
            sample = _fit_prefixes_to_context(root_domain, known, tokenizer)
        else:
            sample = None

        prompt = build_prompt(root_domain, sample)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=batch_size,
            temperature=temps[i],
            min_p=min_p,
            top_p=1.0,
            top_k=0,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        for seq in outputs:
            generated_text = tokenizer.decode(seq, skip_special_tokens=False)
            prefixes = parse_prefixes(generated_text, root_domain)
            all_prefixes.update(prefixes)

    # Remove known prefixes from results
    if known:
        all_prefixes -= set(known)

    # Convert prefixes to full FQDNs
    return sorted(f"{p}.{root_domain}" for p in all_prefixes)


def load_wordlist(path: str) -> tuple[str, list[str]]:
    """Load a subdomain wordlist and infer the root domain.

    Reads one FQDN per line (e.g. subfinder output), extracts the most common
    root domain, and returns (root_domain, list_of_prefixes).
    """
    prefixes = []
    roots = Counter()

    with open(path, "r") as f:
        for line in f:
            fqdn = line.strip().rstrip(".")
            if not fqdn:
                continue
            ext = tldextract.extract(fqdn)
            if ext.domain and ext.suffix and ext.subdomain:
                root = f"{ext.domain}.{ext.suffix}"
                roots[root] += 1
                prefixes.append((root, ext.subdomain))

    if not roots:
        print(f"Error: no valid domains found in {path}", file=sys.stderr)
        sys.exit(1)

    root_domain = roots.most_common(1)[0][0]
    # Filter to only prefixes matching the inferred root
    known = list({p for r, p in prefixes if r == root_domain})
    return root_domain, known


def main():
    parser = argparse.ArgumentParser(description="Generate subdomain candidates")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--domain", type=str, default=None,
                        help="Target root domain (inferred from wordlist if not set)")
    parser.add_argument("--known", type=str, nargs="*", default=None,
                        help="Known subdomain prefixes to condition on (e.g. www mail cdn)")
    parser.add_argument("--wordlist", type=str, default=None,
                        help="Path to subdomain wordlist (e.g. subfinder output)")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of generation runs (each with different context)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Sequences to generate in parallel per sample")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--min-p", type=float, default=0.05)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-temperature-sweep", action="store_true",
                        help="Use fixed temperature instead of sweeping across samples")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file (default: stdout)")
    args = parser.parse_args()

    if args.wordlist:
        root_domain, known = load_wordlist(args.wordlist)
        if args.domain:
            root_domain = args.domain
        if args.known:
            known.extend(args.known)
        print(f"Inferred root domain: {root_domain}", file=sys.stderr)
        print(f"Loaded {len(known)} known subdomains from wordlist", file=sys.stderr)
    elif args.domain:
        root_domain = args.domain
        known = args.known
    else:
        parser.error("either --domain or --wordlist is required")

    model, tokenizer = load_model(args.model_path, args.device)

    candidates = generate_subdomains(
        model,
        tokenizer,
        root_domain,
        known,
        args.num_samples,
        args.batch_size,
        args.temperature,
        args.min_p,
        args.repetition_penalty,
        args.max_new_tokens,
        temperature_sweep=not args.no_temperature_sweep,
    )

    print(f"\nGenerated {len(candidates)} unique subdomain candidates for {root_domain}:",
          file=sys.stderr)

    if args.output:
        with open(args.output, "w") as f:
            for c in candidates:
                f.write(c + "\n")
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        for c in candidates:
            print(c)


if __name__ == "__main__":
    main()

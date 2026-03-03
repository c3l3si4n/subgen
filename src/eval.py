"""
Held-out recall evaluation for subdomain generation.

For each domain in the validation set:
1. Hold out 20% of subdomains
2. Generate candidates conditioned on the remaining 80%
3. Measure recall@k: fraction of held-out subdomains that were generated

Usage:
    python -m eval --model-path checkpoints/final --val-sequences data/val_sequences.txt
"""

import argparse
import random
import sys

import torch
from tqdm import tqdm

from generate import generate_subdomains, load_model


def parse_sequence(line: str) -> tuple[str, list[str]] | None:
    """Parse a training sequence back into (root_domain, subdomain_prefixes)."""
    line = line.strip()
    if not line:
        return None
    # Format: <bos>domain.com<sep>sub1<sep>sub2...<eos>
    line = line.replace("<bos>", "").replace("<eos>", "").strip()
    parts = [p.strip() for p in line.split("<sep>") if p.strip()]
    if len(parts) < 2:
        return None
    root_domain = parts[0]
    prefixes = parts[1:]
    return root_domain, prefixes


def load_val_domains(
    val_path: str,
    min_subs: int = 5,
) -> dict[str, list[str]]:
    """Load validation sequences and group subdomains by root domain.

    Only keeps domains with at least min_subs subdomains (need enough to
    split into prompt and held-out sets).
    """
    domains: dict[str, set[str]] = {}
    with open(val_path) as f:
        for line in f:
            parsed = parse_sequence(line)
            if parsed is None:
                continue
            root, prefixes = parsed
            if root not in domains:
                domains[root] = set()
            domains[root].update(prefixes)

    # Filter to domains with enough subdomains for meaningful eval
    return {
        root: sorted(subs)
        for root, subs in domains.items()
        if len(subs) >= min_subs
    }


def evaluate_recall(
    model,
    tokenizer,
    val_domains: dict[str, list[str]],
    holdout_fraction: float = 0.2,
    num_samples: int = 5,
    batch_size: int = 16,
    max_domains: int | None = None,
    seed: int = 42,
) -> dict:
    """Compute held-out recall@k across validation domains.

    Returns dict with per-domain and aggregate metrics.
    """
    random.seed(seed)
    domains = list(val_domains.items())
    if max_domains:
        domains = domains[:max_domains]

    results = []
    total_held_out = 0
    total_recalled = 0

    for root_domain, all_prefixes in tqdm(domains, desc="Evaluating"):
        # Split into prompt (80%) and held-out (20%)
        shuffled = list(all_prefixes)
        random.shuffle(shuffled)
        n_holdout = max(1, int(len(shuffled) * holdout_fraction))
        held_out = set(shuffled[:n_holdout])
        prompt_prefixes = shuffled[n_holdout:]

        # Generate candidates conditioned on prompt prefixes
        candidates = generate_subdomains(
            model,
            tokenizer,
            root_domain,
            known_prefixes=prompt_prefixes,
            num_samples=num_samples,
            batch_size=batch_size,
            temperature_sweep=True,
        )

        # Extract generated prefixes from FQDNs
        generated_prefixes = set()
        suffix = f".{root_domain}"
        for fqdn in candidates:
            if fqdn.endswith(suffix):
                generated_prefixes.add(fqdn[: -len(suffix)])

        # Compute recall
        recalled = held_out & generated_prefixes
        recall = len(recalled) / len(held_out) if held_out else 0.0

        total_held_out += len(held_out)
        total_recalled += len(recalled)

        results.append({
            "domain": root_domain,
            "num_prompt": len(prompt_prefixes),
            "num_held_out": len(held_out),
            "num_generated": len(candidates),
            "num_recalled": len(recalled),
            "recall": recall,
        })

    # Aggregate metrics
    macro_recall = (
        sum(r["recall"] for r in results) / len(results) if results else 0.0
    )
    micro_recall = total_recalled / total_held_out if total_held_out else 0.0

    return {
        "num_domains": len(results),
        "macro_recall": macro_recall,
        "micro_recall": micro_recall,
        "total_held_out": total_held_out,
        "total_recalled": total_recalled,
        "per_domain": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate held-out recall")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--val-sequences", type=str, required=True,
                        help="Path to val_sequences.txt")
    parser.add_argument("--min-subs", type=int, default=5,
                        help="Minimum subdomains per domain to include in eval")
    parser.add_argument("--holdout-fraction", type=float, default=0.2,
                        help="Fraction of subdomains to hold out (default: 0.2)")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Generation samples per domain")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Sequences per generation sample")
    parser.add_argument("--max-domains", type=int, default=None,
                        help="Max domains to evaluate (for quick testing)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    print("Loading model...", file=sys.stderr)
    model, tokenizer = load_model(args.model_path, args.device)

    print("Loading validation domains...", file=sys.stderr)
    val_domains = load_val_domains(args.val_sequences, args.min_subs)
    print(f"Found {len(val_domains)} domains with >= {args.min_subs} subdomains",
          file=sys.stderr)

    if not val_domains:
        print("No qualifying domains found. Try lowering --min-subs.",
              file=sys.stderr)
        sys.exit(1)

    results = evaluate_recall(
        model,
        tokenizer,
        val_domains,
        holdout_fraction=args.holdout_fraction,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_domains=args.max_domains,
        seed=args.seed,
    )

    print(f"\n{'=' * 50}")
    print(f"Held-out Recall Evaluation")
    print(f"{'=' * 50}")
    print(f"Domains evaluated: {results['num_domains']}")
    print(f"Total held-out subdomains: {results['total_held_out']}")
    print(f"Total recalled: {results['total_recalled']}")
    print(f"Macro recall: {results['macro_recall']:.4f}")
    print(f"Micro recall: {results['micro_recall']:.4f}")

    # Show per-domain breakdown (top 10 by recall)
    print(f"\n{'=' * 50}")
    print("Top 10 domains by recall:")
    sorted_domains = sorted(results["per_domain"], key=lambda x: x["recall"],
                            reverse=True)
    for r in sorted_domains[:10]:
        print(f"  {r['domain']}: recall={r['recall']:.3f} "
              f"({r['num_recalled']}/{r['num_held_out']} held-out, "
              f"{r['num_generated']} generated)")

    # Show worst 10
    print(f"\nBottom 10 domains by recall:")
    for r in sorted_domains[-10:]:
        print(f"  {r['domain']}: recall={r['recall']:.3f} "
              f"({r['num_recalled']}/{r['num_held_out']} held-out, "
              f"{r['num_generated']} generated)")


if __name__ == "__main__":
    main()

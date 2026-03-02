"""
GRPO reward function for subdomain generation RL fine-tuning.

Composite reward combining coverage, diversity, and DNS validity.
Interface compatible with verl's rule-based reward system.
"""

import re

from generate import parse_prefixes


# Regex for valid DNS subdomain labels
_DNS_LABEL_RE = re.compile(r"^[a-z0-9]([a-z0-9\-.]*[a-z0-9])?$")


def compute_score(
    solution_str: str,
    ground_truth: list[str],
    extra_info: dict | None = None,
) -> float:
    """Compute a composite reward for a generated subdomain sequence.

    Args:
        solution_str: Raw model output text (decoded tokens, with special tokens).
        ground_truth: List of held-out subdomain prefixes to measure recall against.
        extra_info: Optional dict with 'root_domain' key for prefix parsing.

    Returns:
        Float reward in [0, 1]. Weighted combination of:
          - coverage (0.6): fraction of ground-truth prefixes recalled
          - diversity (0.2): unique / total generated prefixes
          - dns_validity (0.2): fraction of generated prefixes matching DNS label regex
    """
    root_domain = (extra_info or {}).get("root_domain", "")
    prefixes = parse_prefixes(solution_str, root_domain)

    # Coverage: recall over held-out ground truth
    if ground_truth:
        gt_set = set(ground_truth)
        recalled = sum(1 for p in prefixes if p in gt_set)
        coverage = recalled / len(gt_set)
    else:
        coverage = 0.0

    # Diversity: penalize degenerate repetition
    if prefixes:
        diversity = len(set(prefixes)) / len(prefixes)
    else:
        diversity = 0.0

    # DNS validity: fraction of prefixes matching valid label format
    if prefixes:
        valid = sum(1 for p in prefixes if _DNS_LABEL_RE.match(p))
        dns_validity = valid / len(prefixes)
    else:
        dns_validity = 0.0

    return 0.6 * coverage + 0.2 * diversity + 0.2 * dns_validity

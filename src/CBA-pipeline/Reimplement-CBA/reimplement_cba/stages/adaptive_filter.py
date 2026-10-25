from __future__ import annotations

from collections.abc import Iterable

from reimplement_cba.contracts import CurriculumDecision, SampleScore


def decide_sample(
    score: SampleScore,
    *,
    domain: str,
    threshold: float = 50.0,
) -> CurriculumDecision:
    """Apply CBA's domain-specific previous/current SemScore thresholds."""
    if not 0 <= score.previous_score <= 100 or not 0 <= score.current_score <= 100:
        raise ValueError(f"Scores for {score.sample_id} must be between 0 and 100")
    if not 0 <= threshold <= 100:
        raise ValueError("threshold must be between 0 and 100")
    if domain not in {"in_domain", "out_of_domain"}:
        raise ValueError("domain must be 'in_domain' or 'out_of_domain'")

    both_below = score.previous_score < threshold and score.current_score < threshold
    both_above = score.previous_score > threshold and score.current_score > threshold
    improved = score.current_score > score.previous_score

    if both_below:
        category = "noisy"
    elif score.previous_score >= threshold and score.current_score >= threshold:
        category = "easy"
    else:
        category = "hard"

    if domain == "in_domain":
        keep = not both_below
        reason = "retain in-domain sample unless both scores are below threshold"
    else:
        keep = both_above and improved
        reason = "retain out-of-domain sample only when both scores exceed threshold and improve"

    return CurriculumDecision(
        sample_id=score.sample_id,
        domain=domain,
        category=category,
        keep=keep,
        previous_score=score.previous_score,
        current_score=score.current_score,
        reason=reason,
    )


def apply_adaptive_rule(
    scores: Iterable[SampleScore],
    *,
    domain: str,
    threshold: float = 50.0,
) -> tuple[list[CurriculumDecision], list[CurriculumDecision]]:
    decisions = [decide_sample(score, domain=domain, threshold=threshold) for score in scores]
    return (
        [decision for decision in decisions if decision.keep],
        [decision for decision in decisions if not decision.keep],
    )

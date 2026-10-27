"""Align ASR transcripts and fit every NLI text pair into the model window."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass(frozen=True)
class NLIChunk:
    reference: str
    hypothesis: str
    weight: int
    unmatched_weight: int = 0


def _content_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(
        text, add_special_tokens=False, truncation=False, verbose=False
    ))


def _alignment_key(word: str) -> str:
    folded = unicodedata.normalize("NFC", word).casefold()
    key = "".join(char for char in folded
                  if not unicodedata.category(char).startswith("P"))
    return key or folded


def _spread(words: list[str], slots: int) -> list[str]:
    result = [""] * slots
    for index, word in enumerate(words):
        position = (2 * index + 1) * slots // (2 * len(words))
        result[position] = word
    return result


def _aligned_units(reference: str, hypothesis: str) -> list[tuple[str, str]]:
    refs, hyps = re.findall(r"\S+", reference), re.findall(r"\S+", hypothesis)
    matcher = SequenceMatcher(
        None, [_alignment_key(word) for word in refs],
        [_alignment_key(word) for word in hyps], autojunk=False,
    )
    units: list[tuple[str, str]] = []
    for operation, ref_start, ref_end, hyp_start, hyp_end in matcher.get_opcodes():
        ref_words, hyp_words = refs[ref_start:ref_end], hyps[hyp_start:hyp_end]
        if operation == "equal":
            units.extend(zip(ref_words, hyp_words))
        elif operation == "replace":
            slots = max(len(ref_words), len(hyp_words))
            units.extend(zip(_spread(ref_words, slots), _spread(hyp_words, slots)))
        elif operation == "delete":
            units.extend((word, "") for word in ref_words)
        else:
            units.extend(("", word) for word in hyp_words)
    return units


def _chunk(tokenizer, reference: str, hypothesis: str,
           units: list[tuple[str, str]] | None = None) -> NLIChunk:
    if units is None:
        return NLIChunk(reference, hypothesis, max(
            _content_tokens(tokenizer, reference), _content_tokens(tokenizer, hypothesis)
        ))
    matched_weight = 0
    unmatched_weight = 0
    for ref, hyp in units:
        ref_tokens = _content_tokens(tokenizer, ref)
        hyp_tokens = _content_tokens(tokenizer, hyp)
        if ref and hyp:
            matched_weight += max(ref_tokens, hyp_tokens)
        else:
            unmatched_weight += ref_tokens + hyp_tokens
    return NLIChunk(reference, hypothesis, matched_weight, unmatched_weight)


def _split_oversized_word(tokenizer, reference: str, hypothesis: str,
                          capacity: int) -> list[NLIChunk]:
    if not reference and not hypothesis:
        return []
    if _content_tokens(tokenizer, reference) + _content_tokens(tokenizer, hypothesis) <= capacity:
        return [_chunk(tokenizer, reference, hypothesis, [(reference, hypothesis)])]
    if len(reference) + len(hypothesis) <= 1:
        raise ValueError("One transcript character exceeds the NLI token budget")
    if len(reference) == len(hypothesis) == 1:
        return (_split_oversized_word(tokenizer, reference, "", capacity)
                + _split_oversized_word(tokenizer, "", hypothesis, capacity))
    ref_mid = (len(reference) + 1) // 2
    hyp_mid = (len(hypothesis) + 1) // 2
    return (
        _split_oversized_word(tokenizer, reference[:ref_mid], hypothesis[:hyp_mid], capacity)
        + _split_oversized_word(tokenizer, reference[ref_mid:], hypothesis[hyp_mid:], capacity)
    )


def _fit_units(tokenizer, units: list[tuple[str, str]], capacity: int) -> list[NLIChunk]:
    reference = " ".join(word for word, _ in units if word)
    hypothesis = " ".join(word for _, word in units if word)
    if _content_tokens(tokenizer, reference) + _content_tokens(tokenizer, hypothesis) <= capacity:
        return [_chunk(tokenizer, reference, hypothesis, units)]
    if len(units) == 1:
        return _split_oversized_word(tokenizer, reference, hypothesis, capacity)

    weights = [max(1, _content_tokens(tokenizer, ref) + _content_tokens(tokenizer, hyp))
               for ref, hyp in units]
    half = sum(weights) / 2
    cumulative = 0
    split_at = 1
    smallest_gap = float("inf")
    for index, weight in enumerate(weights[:-1], start=1):
        cumulative += weight
        gap = abs(cumulative - half)
        if gap < smallest_gap:
            smallest_gap, split_at = gap, index
    return (_fit_units(tokenizer, units[:split_at], capacity)
            + _fit_units(tokenizer, units[split_at:], capacity))


def build_nli_chunks(reference: str, hypothesis: str, tokenizer,
                     *, max_length: int) -> list[NLIChunk]:
    """Return ordered, non-overlapping pairs with no lost transcript content."""
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and tokenizer_limit < 1_000_000:
        if max_length > tokenizer_limit:
            raise ValueError(
                f"NLI max_length={max_length} exceeds tokenizer/model maximum {tokenizer_limit}"
            )
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=True))
    capacity = max_length - special_tokens
    if capacity < 1:
        raise ValueError("NLI max_length leaves no room for transcript tokens")
    reference = unicodedata.normalize("NFC", reference)
    hypothesis = unicodedata.normalize("NFC", hypothesis)
    ref_tokens = _content_tokens(tokenizer, reference)
    hyp_tokens = _content_tokens(tokenizer, hypothesis)
    if not ref_tokens or not hyp_tokens:
        return [NLIChunk(reference, hypothesis, 0, max(1, ref_tokens + hyp_tokens))]
    if ref_tokens + hyp_tokens <= capacity:
        return [_chunk(tokenizer, reference, hypothesis)]
    # SequenceMatcher's tie-breaking depends on argument order for repeated
    # words. Canonicalize that order so swapping ref/hyp cannot change the
    # alignment or the final symmetric NLI score.
    swapped = reference > hypothesis
    if swapped:
        reference, hypothesis = hypothesis, reference
    chunks = _fit_units(tokenizer, _aligned_units(reference, hypothesis), capacity)
    if swapped:
        return [NLIChunk(chunk.hypothesis, chunk.reference, chunk.weight,
                         chunk.unmatched_weight) for chunk in chunks]
    return chunks

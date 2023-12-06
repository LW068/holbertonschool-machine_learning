#!/usr/bin/env python3
"""
1-ngram_bleu.py - module that calculates the
n-gram BLEU score for a sentence
"""

import math
from collections import Counter
from typing import List, Tuple


def get_ngrams(sequence: List[str], n: int) -> List[Tuple[str, ...]]:
    """extracts n-grams from a sequence of items."""

    return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]


def ngram_bleu(references: List[List[str]], sentence: List[str], n: int) -> float:
    """calculates the n-gram BLEU score for a sentence."""

    sentence_ngrams = get_ngrams(sentence, n)
    ref_ngrams = [get_ngrams(ref, n) for ref in references]

    sentence_counts = Counter(sentence_ngrams)
    max_ref_counts = Counter()

    for ref in ref_ngrams:
        max_ref_counts |= Counter(ref)

    clipped_counts = {word: min(count, max_ref_counts[word]) for word, count in sentence_counts.items()}

    clipped_total = sum(clipped_counts.values())
    total = len(sentence_ngrams)

    precision = clipped_total / total if total > 0 else 0

    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda ref_len: abs(ref_len - len(sentence)))
    brevity_pen = math.exp(1 - closest_ref_len / len(sentence)) if len(sentence) < closest_ref_len else 1

    bleu_score = brevity_pen * precision

    return bleu_score

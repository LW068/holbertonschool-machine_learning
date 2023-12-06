#!/usr/bin/env python3
"""
1-ngram_bleu.py - module that calculates the n-gram BLEU score for a sentence
"""

import math
from collections import Counter
from typing import List, Tuple


def get_ngrams(sequence: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    This method extracts n-grams from a sequence of items.
    """

    return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]


def ngram_bleu(references, sentence, n):
    """
    This emthod calculates the n-gram BLEU score for a sentence.
    """
    sentence_ngrams = get_ngrams(sentence, n)
    ref_ngrams = [get_ngrams(ref, n) for ref in references]

    sentence_counts = Counter(sentence_ngrams)
    max_ref_counts = Counter()

    for ref in ref_ngrams:
        max_ref_counts |= Counter(ref)

    clipped_counts = {}
    for word, count in sentence_counts.items():
        clipped_counts[word] = min(count, max_ref_counts[word])

    clipped_total = sum(clipped_counts.values())
    total = len(sentence_ngrams)

    precision = clipped_total / total if total > 0 else 0

    def length_difference(ref_len):
        return abs(ref_len - len(sentence))

    closest_ref_len = min(ref_lens, key=length_difference)

    is_brevity = len(sentence) < closest_ref_len
    brevity_factor = 1 - closest_ref_len / len(sentence)
    brevity_pen = math.exp(brevity_factor) if is_brevity else 1
    bleu_score = brevity_pen * precision

    return bleu_score


def get_closest_ref_length(ref_lens, sentence_length):
    """
    Finds the length of the closest reference translation.
    """
    return min(ref_lens, key=lambda ref_len: abs(ref_len - sentence_length))

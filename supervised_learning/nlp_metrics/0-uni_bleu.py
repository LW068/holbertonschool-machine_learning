#!/usr/bin/env python3
"""
0-uni_bleu.py - module that calculates the unigram BLEU score for a sentence
"""

import math
from collections import Counter

def uni_bleu(references, sentence):
    """
    calculates the unigram BLEU score for a sentence
    """
    word_counts = Counter(sentence)
    max_ref_word_counts = {}

    for ref in references:
        for word in ref:
            max_ref_word_counts[word] = max(max_ref_word_counts.get(word, 0), ref.count(word))

    clipped_counts = sum(min(word_counts[word], max_ref_word_counts.get(word, 0)) for word in word_counts)

    sentence_len = len(sentence)
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda ref_len: abs(ref_len - sentence_len))
    brevity_pen = math.exp(1 - closest_ref_len / sentence_len) if sentence_len < closest_ref_len else 1

    bleu_score = brevity_pen * (clipped_counts / sentence_len)

    return bleu_score

# testing the function
if __name__ == '__main__':
    references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
    sentence = ["there", "is", "a", "cat", "here"]

    print(uni_bleu(references, sentence))
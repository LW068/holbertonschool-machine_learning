#!/usr/bin/env python3
"""calculates the culumative n-gram BLEU score of a sentnece"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """calculates the culumative n-gram BLEU score of a sentnece"""
    
    individual_scores = []
    
    for i in range(1, n + 1):
        individual_scores.append(ngram_bleu(references, sentence, i))
    
    cumulative_bleu_score = np.exp(np.mean(np.log(individual_scores)))
    
    return cumulative_bleu_score

import numpy as np
import re

def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.
    """
    # preprocess sentences: lowercase and remove punctuation
    sentences = [re.sub(r'[^\w\s]', '', sentence.lower()) for sentence in sentences]

    if vocab is None:
        vocab = sorted(set(word for sentence in sentences for word in sentence.split()))

    embeddings = np.zeros((len(sentences), len(vocab)))
    vocab_dict = {word: idx for idx, word in enumerate(vocab)}

    for idx, sentence in enumerate(sentences):
        words = sentence.split()
        for word in words:
            if word in vocab_dict:
                embeddings[idx, vocab_dict[word]] += 1

    return embeddings, vocab

#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_datasets as tfds

class Dataset:
    """Dataset class for loading and preparing a machine translation dataset."""

    def __init__(self):
        """initializes the Dataset instance."""
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for the dataset."""
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15)
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15)

        return tokenizer_pt, tokenizer_en

#!/usr/bin/env python3
"""
QA Bot using BERT: A simple script to find answers frorm a text passage using
BERT model from TensorFlow Hub and Transformers library
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(query, passage):
    """Finds an answer to a question from a given passage using BERT"""
    bert_tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')
    bert_model = hub.load(
        "https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    tokens = bert_tokenizer.encode_plus(
        query, passage, return_tensors='tf', truncation=True, 
        padding=True, max_length=512)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    token_type_ids = tokens["token_type_ids"]

    outputs = bert_model([input_ids, attention_mask, token_type_ids])
    start_scores, end_scores = outputs[0], outputs[1]

    start_idx = tf.argmax(start_scores, axis=1).numpy()[0]
    end_idx = tf.argmax(end_scores, axis=1).numpy()[0]

    answer_tokens = bert_tokenizer.convert_ids_to_tokens(
        input_ids.numpy()[0][start_idx:end_idx + 1])
    answer = bert_tokenizer.convert_tokens_to_string(answer_tokens)
    # checking if answer is valuid
    if answer and not answer.strip("[CLS] [SEP]").strip():
        return None  # No valid answer found
    return answer

#!/usr/bin/env python3
"""Calculate Accuracy"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculate accuracy of prediction"""
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

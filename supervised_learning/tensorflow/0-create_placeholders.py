#!/usr/bin/env python3
"""Create Placeholders"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """returns two placeholders, x and y, for the neural network"""
    x = tf.placeholder(dtype=tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, classes], name='y')
    return x, y

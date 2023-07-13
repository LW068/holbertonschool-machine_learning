#!/usr/bin/env python3
"""builds a modified version of LeNet-5"""

import tensorflow as tf


def lenet5(x, y):
    """Builds a modified version of the LeNet-5...
    ...architecture using tensorflow"""
    # Initialize weights using He et al.
    # method (also known as 'he_normal')
    init = tf.contrib.layers.variance_scaling_initializer()

    # Layer 1: Convolutional
    conv1 = tf.layers.conv2d(x, filters=6,
                             kernel_size=5, padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=init)

    # Layer 2: Max pooling
    pool1 = tf.layers.max_pooling2d(conv1,
                                    pool_size=[2, 2], strides=2)

    # Layer 3: Convolutional
    conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=5,
                             activation=tf.nn.relu,
                             kernel_initializer=init)

    # Layer 4: Max pooling
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)

    # Flatten the network
    fc1 = tf.layers.flatten(pool2)

    # Layer 5: Fully connected layer
    fc1 = tf.layers.dense(fc1, units=120, activation=tf.nn.relu,
                          kernel_initializer=init)

    # Layer 6: Fully connected layer
    fc2 = tf.layers.dense(fc1, units=84, activation=tf.nn.relu,
                          kernel_initializer=init)

    # Output layer: Fully connected layer
    output = tf.layers.dense(fc2, units=10, kernel_initializer=init)

    # Apply softmax to the output
    softmax = tf.nn.softmax(output)

    # Define loss
    loss = tf.losses.softmax_cross_entropy(y, output)

    # Define optimizer (Adam)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # Define accuracy
    correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return softmax, optimizer, loss, accuracy

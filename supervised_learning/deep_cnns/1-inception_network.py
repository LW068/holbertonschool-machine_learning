#!/usr/bin/env python3
"""Inception Network

This module contains a function that builds the inception network using Keras
as described in "Going Deeper with Convolutions (2014)".
"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """ Builds the inception network

    All convolutions inside and outside the inception block should use a rectified linear activation (ReLU)

    Returns:
        keras model: the keras model
    """
    input_layer = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(input_layer)
    pool1 = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(conv1)
    conv2 = K.layers.Conv2D(64, 1, activation='relu')(pool1)
    conv3 = K.layers.Conv2D(192, 3, padding='same', activation='relu')(conv2)
    pool3 = K.layers.MaxPooling2D(pool_size=3, strides=2)(conv3)

    inception_3a = inception_block(pool3, [64, 96, 128, 16, 32, 32])
    inception_3b = inception_block(inception_3a, [128, 128, 192, 32, 96, 64])

    pool5 = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(inception_3b)
    
    inception_4a = inception_block(pool5, [192, 96, 208, 16, 48, 64])
    inception_4b = inception_block(inception_4a, [160, 112, 224, 24, 64, 64])
    inception_4c = inception_block(inception_4b, [128, 128, 256, 24, 64, 64])
    inception_4d = inception_block(inception_4c, [112, 144, 288, 32, 64, 64])
    inception_4e = inception_block(inception_4d, [256, 160, 320, 32, 128, 128])

    pool10 = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(inception_4e)

    inception_5a = inception_block(pool10, [256, 160, 320, 32, 128, 128])
    inception_5b = inception_block(inception_5a, [384, 192, 384, 48, 128, 128])
    
    pool13 = K.layers.AveragePooling2D(pool_size=7, strides=1, padding='valid')(inception_5b)
    drop = K.layers.Dropout(0.4)(pool13)

    linear = K.layers.Dense(1000, activation='softmax')(drop)

    model = K.models.Model(inputs=input_layer, outputs=linear)

    return model

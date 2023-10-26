#!/usr/bin/env python3
""" variational autoencoder """

import tensorflow as tf


def sampling(args):
    """ sampling """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ variational autoencoder """

    # encoder
    inputs = tf.keras.layers.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = tf.keras.layers.Dense(units, activation='relu')(x)
    z_mean = tf.keras.layers.Dense(latent_dims)(x)
    z_log_var = tf.keras.layers.Dense(latent_dims)(x)
    z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = tf.keras.Model(inputs, [z, z_mean, z_log_var])

    # decoder
    decoder_h = tf.keras.layers.Input(shape=(latent_dims,))
    x = decoder_h
    for units in reversed(hidden_layers):
        x = tf.keras.layers.Dense(units, activation='relu')(x)
    x = tf.keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = tf.keras.Model(decoder_h, x)

    # autoencoder
    outputs = decoder(encoder(inputs)[0])
    auto = tf.keras.Model(inputs, outputs)

    # loss
    reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= input_dims
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)

    # compile
    auto.compile(optimizer='adam')

    return encoder, decoder, auto

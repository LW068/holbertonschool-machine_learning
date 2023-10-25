#!/usr/bin/env python3
""" creates vanilla autoencoder """

import tensorflow.keras as keras

def autoencoder(input_dims, hidden_layers, latent_dims):
    """ creates vanilla autoencoder """

    # encoder
    encoder_input = keras.layers.Input(shape=(input_dims,))
    x = encoder_input
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    latent = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(encoder_input, latent)
    
    # decoder
    decoder_input = keras.layers.Input(shape=(latent_dims,))
    x = decoder_input
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, decoder_output)
    
    # autoencoder
    auto_input = keras.layers.Input(shape=(input_dims,))
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(auto_input, decoded)

    # compile model
    auto.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')
    
    return encoder, decoder, auto

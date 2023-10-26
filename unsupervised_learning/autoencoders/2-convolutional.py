#!/usr/bin/env python3
""" convolutional autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """ convolutional autoencoder """
    # encder
    enc_in = keras.layers.Input(shape=input_dims)
    x = enc_in
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.models.Model(enc_in, x)
    
    # decoder
    dec_in = keras.layers.Input(shape=latent_dims)
    x = dec_in
    for f in reversed(filters[:-1]):
        x = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(filters[-1], (3, 3), activation='relu')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    dec_out = keras.layers.Conv2D(input_dims[-1], (3, 3),
                                  activation='sigmoid', padding='same')(x)
    decoder = keras.models.Model(dec_in, dec_out)
    
    # autoencoder
    auto_in = keras.layers.Input(shape=input_dims)
    enc = encoder(auto_in)
    dec = decoder(enc)
    auto = keras.models.Model(auto_in, dec)
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto

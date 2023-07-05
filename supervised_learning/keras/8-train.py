#!/usr/bin/env python3
"""
Module to train model with...
...learning rate decay and save best iteration
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """train_model"""
    def learning_rate(epoch):
        """ updates the learning rate using inverse time decay"""
        return alpha / (1 + decay_rate * epoch)

    callbacks = []

    if validation_data:
        if early_stopping:
            callbacks.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=patience))
        if learning_rate_decay:
            callbacks.append(K.callbacks.LearningRateScheduler(learning_rate,
                                                               verbose=1))
        if save_best:
            callbacks.append(K.callbacks.ModelCheckpoint(filepath,
                                                         save_best_only=True,
                                                         monitor='val_loss',
                                                         mode='min'))

    history = network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                          validation_data=validation_data, callbacks=callbacks,
                          verbose=verbose, shuffle=shuffle)

    return history

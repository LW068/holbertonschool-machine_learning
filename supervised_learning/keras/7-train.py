#!/usr/bin/env python3
"""
Module to train model with learning rate decay
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """train_model"""
    def learning_rate(epoch):
        """ updates the learning rate using inverse time decay"""
        return alpha / (1 + decay_rate * epoch)
    """
    Function that trains a model using mini-batch gradient descent

    Arguments:
     - network is the model to train
     - data is a numpy.ndarray of shape (m, nx) containing the input data
     - labels is a one-hot numpy.ndarray of shape (m, classes)
     - batch_size is the size of the batch used for mini-batch gradient descent
     - epochs is the number of passes through data for mini-batch
        gradient descent
     - verbose is a boolean that determines if output should be printed
     - shuffle is a boolean that determines whether to shuffle the batches
        every epoch.
     - early_stopping is a boolean that indicates whether early stopping
        should be used
     - patience is the patience used for early stopping
     - learning_rate_decay is a boolean that indicates whether learning
        rate decay should be used
     - alpha is the initial learning rate
     - decay_rate is the decay rate

    Returns:
     - History object generated after training the model
    """

    callbacks = []

    if validation_data:
        if early_stopping:
            callbacks.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=patience))
        if learning_rate_decay:
            callbacks.append(K.callbacks.LearningRateScheduler(learning_rate,
                                                               verbose=1))

    history = network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                          validation_data=validation_data, callbacks=callbacks,
                          verbose=verbose, shuffle=shuffle)

    return history

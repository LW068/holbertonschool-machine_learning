#!/usr/bin/env python3
"""
Module to create, train, and save a neural network model
in tensorflow using Adam optimization, mini-batch gradient descent,
learning rate decay, and batch normalization.
"""

import tensorflow as tf
import numpy as np


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=init)
    mean, var = tf.nn.moments(layer(prev), axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), name="gamma")
    beta = tf.Variable(tf.constant(0.0, shape=[n]), name="beta")
    BN = tf.nn.batch_normalization(layer(prev), mean, var, offset=beta,
                                   scale=gamma, variance_epsilon=1e-8)
    return activation(BN)


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in tensorflow using Adam
    optimization, mini-batch gradient descent, learning rate decay, and batch
    normalization.
    """
    # Unpack the data
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    # Placeholder for the input data
    x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]])
    y = tf.placeholder(tf.float32, shape=[None, Y_train.shape[1]])

    # Build the network
    for i in range(len(layers)):
        if i == 0:
            layer = create_batch_norm_layer(x, layers[i], activations[i])
        else:
            layer = create_batch_norm_layer(layer, layers[i], activations[i])

    # Define the cost function
    cost = tf.losses.softmax_cross_entropy(y, layer)

    # Define the learning rate decay
    global_step = tf.Variable(0, trainable=False)
    alpha = tf.train.inverse_time_decay(alpha, global_step, 1, decay_rate)

    # Define the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                       beta2=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(cost, global_step=global_step)

    # Add an operation to initialize the variables
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()

    # Start a session to run the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(epochs):
            cost_train = 0.
            batches = int(X_train.shape[0] / batch_size)

            # Shuffle the data
            s = np.arange(X_train.shape[0])
            np.random.shuffle(s)
            X_train = X_train[s]
            Y_train = Y_train[s]

            # Minibatch training
            for i in range(0, X_train.shape[0], batch_size):
                X_train_mini = X_train[i:i+batch_size]
                Y_train_mini = Y_train[i:i+batch_size]
                _, cost_mini = sess.run([train_op, cost],
                                        feed_dict={x: X_train_mini,
                                                   y: Y_train_mini})
                cost_train += cost_mini

                if (i/batch_size) % 100 == 0 and i/batch_size > 0:
                    cost_mini = sess.run(cost, feed_dict={x: X_train_mini,
                                                          y: Y_train_mini})
                    accuracy_mini = np.mean(
                        np.argmax(Y_train_mini, axis=1) == sess.run(
                            tf.argmax(layer, 1),
                            feed_dict={x: X_train_mini, y: Y_train_mini}))
                    print("\tStep {}:".format(int(i/batch_size)))
                    print("\t\tCost: {}".format(cost_mini))
                    print("\t\tAccuracy: {}".format(accuracy_mini))

            cost_train = sess.run(cost, feed_dict={x: X_train, y: Y_train})
            accuracy_train = np.mean(
                np.argmax(Y_train, axis=1) == sess.run(
                    tf.argmax(layer, 1), feed_dict={x: X_train, y: Y_train}))
            cost_valid = sess.run(cost, feed_dict={x: X_valid, y: Y_valid})
            accuracy_valid = np.mean(
                np.argmax(Y_valid, axis=1) == sess.run(
                    tf.argmax(layer, 1), feed_dict={x: X_valid, y: Y_valid}))
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(cost_train))
            print("\tTraining Accuracy: {}".format(accuracy_train))
            print("\tValidation Cost: {}".format(cost_valid))
            print("\tValidation Accuracy: {}".format(accuracy_valid))

        # Save the model
        save_path = saver.save(sess, save_path)

    return 

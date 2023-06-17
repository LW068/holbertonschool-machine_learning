#!/usr/bin/env python3
"""
Trains, and saves a neural network classifier
"""
# Import required modules
import tensorflow as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Function that builds, trains, and saves a neural network classifier.
    """

    # Placeholders for input data and target values
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # Compute the predicted output (y_pred) using forward propagation
    y_pred = forward_prop(x, layer_sizes, activations)

    # Calculate the accuracy of the prediction
    accuracy = calculate_accuracy(y, y_pred)

    # Calculate the cost (loss) of the prediction
    loss = calculate_loss(y, y_pred)

    # Define the training operation using Gradient Descent Optimization
    train_op = create_train_op(loss, alpha)

    # Add all the created tensors and operations to collections for future retrieval
    params = {'x': x, 'y': y, 'y_pred': y_pred, 'accuracy': accuracy, 'loss': loss, 'train_op': train_op}
    for k, v in params.items():
        tf.add_to_collection(k, v)

    # Initialize a Saver object to save the model after training
    saver = tf.train.Saver()

    # Initialize a global variables initializer
    init = tf.global_variables_initializer()

    # Start a new Tensorflow session
    with tf.Session() as sess:
        # Initialize all variables
        sess.run(init)

        # Iterate over the defined number of iterations
        for i in range(iterations + 1):
            sess.run(y_pred, feed_dict={x: X_train, y: Y_train})
            if i % 100 == 0 or i == iterations:
                # Calculate loss and accuracy metrics for both training and validation data
                loss_t = sess.run(loss, feed_dict={x: X_train, y: Y_train})
                acc_t = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
                loss_v = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
                acc_v = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

                # Print the calculated metrics
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(loss_t))
                print("\tTraining Accuracy: {}".format(acc_t))
                print("\tValidation Cost: {}".format(loss_v))
                print("\tValidation Accuracy: {}".format(acc_v))

            if i < iterations:
                # Perform training on the training data
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        # Save the model at the defined save_path
        save_path = saver.save(sess, save_path)

    return save_path

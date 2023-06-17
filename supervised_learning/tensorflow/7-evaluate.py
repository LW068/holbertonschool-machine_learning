#!/usr/bin/env python3
"""
Evaluate the performance of a neural network
"""
# Import the necessary module
import tensorflow as tf


def evaluate_nn(X, Y, save_path):
    """
    Function to evaluate the output of a neural network.
    It loads a pre-trained model from a file and assesses its performance.
    """

    # Start a TensorFlow session
    with tf.Session() as sess:
        # Load the meta graph from the save path
        loader = tf.train.import_meta_graph(f'{save_path}.meta')
        # Restore the session
        loader.restore(sess, save_path)

        # List of tensor names we want to retrieve
        tensor_names = ['x', 'y', 'y_pred', 'accuracy', 'loss']

        # Retrieve the required tensors from the graph's collection
        for tensor_name in tensor_names:
            # Get the tensor by its name and assign it to a global variable
            globals()[tensor_name] = tf.get_collection(tensor_name)[0]

        # Evaluate the network's output, accuracy, and loss
        # by running the corresponding operations in the session
        y_pred_evaluated = sess.run(globals()['y_pred'], feed_dict={x: X, y: Y})
        loss_evaluated = sess.run(globals()['loss'], feed_dict={x: X, y: Y})
        accuracy_evaluated = sess.run(globals()['accuracy'], feed_dict={x: X, y: Y})

    return y_pred_evaluated, accuracy_evaluated, loss_evaluated

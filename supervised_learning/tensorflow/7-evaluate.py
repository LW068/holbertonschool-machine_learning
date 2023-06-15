#!/usr/bin/env python3
"""Evaluate the output of a neural network"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluate the output of a neural network"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sess, save_path)
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        y_pred = graph.get_tensor_by_name("y_pred:0")
        loss = graph.get_tensor_by_name("loss:0")
        accuracy = graph.get_tensor_by_name("accuracy:0")
        y_pred = sess.run(y_pred, feed_dict={x: X, y: Y})
        loss = sess.run(loss, feed_dict={x: X, y: Y})
        accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
    return y_pred, accuracy, loss

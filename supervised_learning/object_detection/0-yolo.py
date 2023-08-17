#!/usr/bin/env python3
"""Yolo Algorithm for Object Detection"""

import tensorflow.keras as K


class Yolo:
    """ Yolo v3 algorithm to perform object detection """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ class constructor
        model_path: path to Darknet Keras model
        classes_path: path to list of class names for Darknet model
        class_t: float representing the box score threshold
        nms_t: float representing the IOU threshold for non-max suppression
        anchors: numpy.ndarray shape (outputs, anchor_boxes, 2)
        containing all of the anchor boxes
        """

        # Load Darknet Keras model
        self.model = K.models.load_model(model_path)

        # Load class names
        with open(classes_path, 'r') as file:
            self.class_names = [line.strip() for line in file.readlines()]

        # Set class threshold, IOU threshold, and anchor boxes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

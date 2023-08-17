#!/usr/bin/env python3
"""Yolo Object Detection - Filtering Boxes"""

import tensorflow as tf
import numpy as np


class Yolo:
    """
    Implementation of Yolo v3 for object detection.

    Attributes:
        model (Keras model): The model built on Darknet architecture.
        class_names (list): Names of classes recognized by the model.
        class_t (float): Threshold for filtering initial box scores.
        nms_t (float): Threshold for non-maximum suppression.
        anchors (numpy.ndarray): Anchor boxes dimensions.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.rstrip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Handles the processing of the model outputs.

        Args:
            outputs (list): Raw outputs from the model.
            image_size (tuple): Original size of the image.

        Returns:
            tuple: Processed boundary boxes, confidences, and class probabilities.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        # Implementation details...

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filters and refines the boxes based on confidence and class probabilities.

        Args:
            boxes (list): Processed boundary boxes.
            box_confidences (list): Processed box confidences.
            box_class_probs (list): Processed box class probabilities.

        Returns:
            tuple: Filtered bounding boxes, class numbers, and box scores.
        """
        box_scores = [conf * prob for conf, prob in zip(box_confidences, box_class_probs)]
        # Further implementation details...

        return filtered_boxes, box_classes, box_scores

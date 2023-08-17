#!/usr/bin/env python3
"""Object Detection - 2. Yolo - Filter Boxes"""

import tensorflow as tf
import numpy as np


class Yolo:
    """
    Yolo v3 algorithm to perform object detection.

    Attributes:
        model (Keras model): The Darknet Keras model.
        class_names (list): A list of the class names for the model.
        class_t (float): The box score threshold for the initial filtering.
        nms_t (float): The IOU threshold for non-max suppression.
        anchors (numpy.ndarray): The anchor boxes.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process model outputs."""
        # Same implementation as 1-yolo.py

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes.

        Parameters:
        boxes: List of ndarrays containing boundary boxes.
        box_confidences: List of ndarrays containing box confidences.
        box_class_probs: List of ndarrays containing box class probabilities.

        Returns:
        Tuple of (filtered_boxes, box_classes, box_scores).
        """
        box_scores_full = []
        box_classes_full = []
        box_class_scores = []

        # Calculate scores for each class in each bounding box
        for box_conf, box_class_prob in zip(box_confidences, box_class_probs):
            box_scores = box_conf * box_class_prob
            box_classes = np.argmax(box_scores, axis=-1)
            box_class_scores = np.max(box_scores, axis=-1)
            box_scores_full.append(box_scores)
            box_classes_full.append(box_classes)
            box_class_scores.append(box_class_scores)

        # Concatenate results from different anchor boxes
        filtered_boxes = np.concatenate([box.reshape(-1, 4) for box in boxes])
        box_classes = np.concatenate([class_box.reshape(-1) for class_box in box_classes_full])
        box_scores = np.concatenate([score.reshape(-1) for score in box_class_scores])

        # Apply threshold to scores
        filtering_mask = box_scores >= self.class_t

        # Apply filtering mask to obtain final filtered boxes, classes, and scores
        filtered_boxes = filtered_boxes[filtering_mask]
        box_classes = box_classes[filtering_mask]
        box_scores = box_scores[filtering_mask]

        return filtered_boxes, box_classes, box_scores


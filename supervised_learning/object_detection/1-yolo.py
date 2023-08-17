#!/usr/bin/env python3
"""Yolo class"""

import numpy as np
import tensorflow as tf


class Yolo:
    """Yolo class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor"""
        # Load the model
        self.model = tf.keras.models.load_model(model_path)

        # Load the classes
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        # Set the other attributes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Process the model outputs

        outputs: a list of numpy.ndarrays containing the predictions
                 from the Darknet model for a single image
        image_size: numpy.ndarray containing the image’s original size

        Returns: tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape
            box = output[..., :4]

            # Process the boxes
            for h in range(grid_height):
                for w in range(grid_width):
                    for anchor in range(anchor_boxes):
                        pw, ph = self.anchors[i][anchor]
                        tx, ty, tw, th = box[h, w, anchor, :]
                        bx = (tx + w) / grid_width
                        by = (ty + h) / grid_height
                        bw = pw * np.exp(tw) / self.model.input.shape[1]
                        bh = ph * np.exp(th) / self.model.input.shape[2]
                        box[h, w, anchor, 0] = (
                            bx - bw / 2) * image_size[1]
                        box[h, w, anchor, 1] = (
                            by - bh / 2) * image_size[0]
                        box[h, w, anchor, 2] = (
                            bx + bw / 2) * image_size[1]
                        box[h, w, anchor, 3] = (
                            by + bh / 2) * image_size[0]

            boxes.append(box)
            box_confidences.append(output[..., 4:5])
            box_class_probs.append(output[..., 5:])

        return boxes, box_confidences, box_class_probs

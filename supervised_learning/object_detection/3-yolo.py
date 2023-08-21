#!/usr/bin/env python3
"""Yolo Version 3 - Object Detection System"""

import tensorflow as tf
import numpy as np


class Yolo:
    """
    Implements Yolo v3 for object recognition.

    Attributes:
        model (Keras model): The pre-trained Darknet model.
        class_names (list): Names of classes that can be detected.
        class_t (float): Minimum confidence threshold for filtering.
        nms_t (float): Intersection over Union threshold for
                       non-max suppression.
        anchors (numpy.ndarray): Pre-defined anchor boxes.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize the Yolo object detector."""
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as file:
            self.class_names = [line.rstrip() for line in file]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Convert model's output into human-readable format."""
        boxes, confidences, class_probs = [], [], []

        for idx, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape
            tx, ty, tw, th = (
                output[..., 0:1], output[..., 1:2],
                output[..., 2:3], output[..., 3:4]
            )
            confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            class_prob = 1 / (1 + np.exp(-output[..., 5:]))

            confidences.append(confidence)
            class_probs.append(class_prob)

            for cy in range(grid_h):
                for cx in range(grid_w):
                    for b in range(anchor_boxes):
                        pw, ph = self.anchors[idx][b]
                        bx, by = (
                            (1 / (1 + np.exp(-tx[cy, cx, b]))) + cx,
                            (1 / (1 + np.exp(-ty[cy, cx, b]))) + cy
                        )
                        bw, bh = (
                            pw * np.exp(tw[cy, cx, b]),
                            ph * np.exp(th[cy, cx, b])
                        )

                        bx, by = bx / grid_w, by / grid_h
                        bw, bh = (
                            bw / int(self.model.input.shape[1]),
                            bh / int(self.model.input.shape[2])
                        )

                        x1, y1, x2, y2 = (
                            (bx - bw / 2) * image_size[1],
                            (by - bh / 2) * image_size[0],
                            (bx + bw / 2) * image_size[1],
                            (by + bh / 2) * image_size[0]
                        )

                        tx[cy, cx, b], ty[cy, cx, b], \
                            tw[cy, cx, b], th[cy, cx, b] = (
                                x1, y1, x2, y2
                        )

            boxes.append(np.concatenate((tx, ty, tw, th), axis=-1))

        return boxes, confidences, class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Refine and filter boxes based on
        confidence and class probability."""
        box_scores = [
            conf * prob for conf, prob in zip(box_confidences, box_class_probs)
        ]
        box_classes = [np.argmax(score, axis=-1) for score in box_scores]
        box_class_scores = [np.max(score, axis=-1) for score in box_scores]

        prediction_mask = [score >= self.class_t for score in box_class_scores]

        filtered_boxes = [box[mask]
                          for box, mask in zip(boxes, prediction_mask)]
        box_classes = [cls[mask]
                       for cls, mask in zip(box_classes, prediction_mask)]
        box_scores = [
            score[mask] for score,
            mask in zip(
                box_class_scores,
                prediction_mask)]

        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Apply non-max suppression to filter boxes."""
        selected_boxes = []
        selected_classes = []
        selected_scores = []

        for unique_class in np.unique(box_classes):
            indices = np.where(box_classes == unique_class)
            boxes_of_class = filtered_boxes[indices]
            scores_of_class = box_scores[indices]

            while boxes_of_class.shape[0] > 0:
                # Select the box with the highest score
                top_index = np.argmax(scores_of_class)
                top_box = boxes_of_class[top_index]
                top_score = scores_of_class[top_index]
                top_class = unique_class

                # Add to the selected list
                selected_boxes.append(top_box)
                selected_classes.append(top_class)
                selected_scores.append(top_score)

                # Remove the top box
                boxes_of_class = np.delete(boxes_of_class, top_index, axis=0)
                scores_of_class = np.delete(scores_of_class, top_index, axis=0)

                # Compute IOU with other boxes
                iou_values = self.calc_intersection_over_union(
                    top_box, boxes_of_class)

                # Filter boxes based on IOU threshold
                valid_indices = np.where(iou_values < self.nms_t)
                boxes_of_class = boxes_of_class[valid_indices]
                scores_of_class = scores_of_class[valid_indices]

        return (np.array(selected_boxes), np.array(selected_classes),
                np.array(selected_scores))

    def calc_intersection_over_union(self, main_box, other_boxes):
        """Calculate the intersection over union!"""
        x1 = np.maximum(main_box[0], other_boxes[:, 0])
        y1 = np.maximum(main_box[1], other_boxes[:, 1])
        x2 = np.minimum(main_box[2], other_boxes[:, 2])
        y2 = np.minimum(main_box[3], other_boxes[:, 3])

        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

        main_area = (main_box[2] - main_box[0]) * (main_box[3] - main_box[1])
        other_areas = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:,
                                                                 3] - other_boxes[:, 1])

        union = main_area + other_areas - intersection

        return intersection / union

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
        nms_t (float): Intersection over Union threshold for non-max suppression.
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

                        tx[cy, cx, b], ty[cy, cx, b], tw[cy, cx, b], th[cy, cx, b] = x1, y1, x2, y2

            boxes.append(np.concatenate((tx, ty, tw, th), axis=-1))

        return boxes, confidences, class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Refine and filter boxes based on confidence and class probability."""
        box_scores = [conf * prob for conf, prob in zip(box_confidences, box_class_probs)]
        box_classes = [np.argmax(score, axis=-1) for score in box_scores]
        box_class_scores = [np.max(score, axis=-1) for score in box_scores]

        prediction_mask = [score >= self.class_t for score in box_class_scores]

        filtered_boxes = [box[mask] for box, mask in zip(boxes, prediction_mask)]
        box_classes = [cls[mask] for cls, mask in zip(box_classes, prediction_mask)]
        box_scores = [score[mask] for score, mask in zip(box_class_scores, prediction_mask)]

        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Apply non-max suppression to filter boxes."""
        final_boxes, final_classes, final_scores = [], [], []

        unique_classes = set(box_classes)

        for cls in unique_classes:
            indices = np.where(box_classes == cls)
            class_specific_boxes = filtered_boxes[indices]
            class_specific_scores = box_scores[indices]

            sorted_indices = np.argsort(class_specific_scores)[::-1]
            class_specific_boxes = class_specific_boxes[sorted_indices]
            class_specific_scores = class_specific_scores[sorted_indices]

            while len(class_specific_boxes) > 0:
                final_boxes.append(class_specific_boxes[0])
                final_classes.append(cls)
                final_scores.append(class_specific_scores[0])

                if len(class_specific_boxes) == 1:
                    break

                iou_values = self.calculate_iou(final_specific_boxes[0], class_specific_boxes[1:])
                iou_mask = iou_values < self.nms_t

                class_specific_boxes = class_specific_boxes[1:][iou_mask]
                class_specific_scores = class_specific_scores[1:][iou_mask]

        return (np.array(final_boxes), np.array(final_classes), np.array(final_scores))

    def calculate_iou(self, box1, boxes):
        """Calculate Intersection over Union (IoU) between boxes."""
        x1, y1, x2, y2 = (
            np.maximum(box1[0], boxes[:, 0]),
            np.maximum(box1[1], boxes[:, 1]),
            np.minimum(box1[2], boxes[:, 2]),
            np.minimum(box1[3], boxes[:, 3])
        )

        intersection_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union_area = box1_area + boxes_area - intersection_area

        return intersection_area / union_area

#!/usr/bin/env python3
"""
This module contains functions that calculate the minor matrix and determinant of a matrix.
"""

def determinant(matrix):
    """
    Calculates the determinant of a matrix.
    """
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for i in range(len(matrix)):
        minor = [row[:i] + row[i+1:] for row in matrix[1:]]
        cofactor = (-1) ** i * matrix[0][i]
        det += cofactor * determinant(minor)
    return det


def minor(matrix):
    """
    Calculates the minor matrix of a matrix.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0 or len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    minor_matrix = []
    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix[i])):
            sub_matrix = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            minor_value = determinant(sub_matrix)
            minor_row.append(minor_value)
        minor_matrix.append(minor_row)

    return minor_matrix

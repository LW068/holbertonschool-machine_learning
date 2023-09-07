#!/usr/bin/env python3
"""
Calculates the cofactor matrix of a given matrix.
"""


def cofactor(matrix):
    """
    Calculates the cofactor matrix.
    """
    if not isinstance(matrix, list) or \
       not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0 or \
       any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    cofactor_matrix = []
    for i in range(len(matrix)):
        cofactor_row = []
        for j in range(len(matrix[i])):
            sub_matrix = [row[:j] + row[j + 1:] for row in
                          (matrix[:i] + matrix[i + 1:])]
            minor_value = determinant(sub_matrix)
            cofactor_value = minor_value if (i + j) % 2 == 0 else -minor_value
            cofactor_row.append(cofactor_value)
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix

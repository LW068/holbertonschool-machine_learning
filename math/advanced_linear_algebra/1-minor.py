#!/usr/bin/env python3
"""
This module contains a function that calculates the minor matrix of a matrix.
"""

def determinant(matrix):
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
    """Calculates the minor matrix of a matrix."""
    if not isinstance(matrix, list) or \
            not all(isinstance(row, list) for row in matrix) or \
            len(matrix) == 0 or len(matrix) != len(matrix[0]):
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


if __name__ == '__main__':
    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(minor(mat1))
    print(minor(mat2))
    print(minor(mat3))
    print(minor(mat4))
    try:
        minor(mat5)
    except Exception as e:
        print(e)
    try:
        minor(mat6)
    except Exception as e:
        print(e)

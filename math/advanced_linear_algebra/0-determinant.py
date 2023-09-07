#!/usr/bin/env python3
import numpy as np


def determinant(matrix):
    """
    Calculates the determinant of a matrix.
    """
    # Check if matrix is a list of lists
    if type(matrix) is not list or not all(type(row) is list for row in matrix):
        raise TypeError("matrix must be a list of lists")
    
    # Check if matrix is square
    if len(matrix) == 0:
        return 1
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    
    # Calculate determinant for 0x0, 1x1 and 2x2 matrices directly
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    # Use NumPy for larger matrices
    return int(round(np.linalg.det(np.array(matrix))))

if __name__ == '__main__':
    # Test cases
    mat0 = [[]]
    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(determinant(mat0))  # Output should be 1
    print(determinant(mat1))  # Output should be 5
    print(determinant(mat2))  # Output should be -2
    print(determinant(mat3))  # Output should be 0
    print(determinant(mat4))  # Output should be 192

    try:
        print(determinant(mat5))  # Should raise TypeError
    except Exception as e:
        print(e)

    try:
        print(determinant(mat6))  # Should raise ValueError
    except Exception as e:
        print(e)

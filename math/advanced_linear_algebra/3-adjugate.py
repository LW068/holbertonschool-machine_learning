#!/usr/bin/env python3
"""
Calculates the adjugate, cofactor, and determinant of a matrix.
"""


def determinant(matrix):
    """Calculates the determinant of a matrix."""
    if not isinstance(matrix, list) or \
            not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0 or len(matrix[0]) == 0:
        return 1

    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

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


if __name__ == '__main__':
    mat0 = [[]]
    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(determinant(mat0))
    print(determinant(mat1))
    print(determinant(mat2))
    print(determinant(mat3))
    print(determinant(mat4))
    try:
        determinant(mat5)
    except Exception as e:
        print(e)
    try:
        determinant(mat6)
    except Exception as e:
        print(e)


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


def adjugate(matrix):
    """
    Calculates the adjugate matrix.
    """
    if not isinstance(matrix, list) or \
       not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0 or \
       any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    cofactor_matrix = cofactor(matrix)
    adjugate_matrix = list(map(list, zip(*cofactor_matrix)))  # Transpose

    return adjugate_matrix

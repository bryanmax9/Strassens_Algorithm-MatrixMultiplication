import numpy as np
import json

def read_matrix_from_file(filename):
    with open(filename, 'r') as file:
        content = file.read().strip()
        matrix = json.loads(content.replace('{', '[').replace('}', ']'))
    return np.array(matrix)

def split_matrix(A):
    half = len(A) // 2
    return A[:half, :half], A[:half, half:], A[half:, :half], A[half:, half:]

def strassen(A, B):
    n = len(A)

    if n <= 256:  # Increase the threshold for direct multiplication
        return A @ B  # Use native NumPy multiplication

    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)

    M1 = strassen(A11, B12 - B22)
    M2 = strassen(A11 + A12, B22)
    M3 = strassen(A21 + A22, B11)
    M4 = strassen(A22, B21 - B11)
    M5 = strassen(A11 + A22, B11 + B22)
    M6 = strassen(A12 - A22, B21 + B22)
    M7 = strassen(A11 - A21, B11 + B12)

    C11 = M5 + M4 - M2 + M6
    C12 = M1 + M2
    C21 = M3 + M4
    C22 = M5 + M1 - M3 - M7

    # Efficiently combine the matrices using numpy functions
    top = np.hstack((C11, C12))
    bottom = np.hstack((C21, C22))
    return np.vstack((top, bottom))

def next_power_of_2(n):
    return 1 << (n - 1).bit_length()

def pad_matrix(A):
    n, m = A.shape
    new_size = max(next_power_of_2(n), next_power_of_2(m))
    padded = np.zeros((new_size, new_size))
    padded[:n, :m] = A
    return padded

if __name__ == "__main__":
    matrix_A = pad_matrix(read_matrix_from_file('10a.txt'))
    matrix_B = pad_matrix(read_matrix_from_file('10b.txt'))

    result = strassen(matrix_A, matrix_B)

    sumEntries = np.sum(result)  # Use direct numpy function

    for row in result:
        print(row)
    
    print("Total sum of the resulting:", sumEntries)

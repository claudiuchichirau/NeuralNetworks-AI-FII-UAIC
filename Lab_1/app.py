import re
import math

#1. Parsing the System of Equations
def parse_equation(equation):
    # 2x + 3y - z = 5     =>     2x +3y -z + 0 = 5
    equation = equation.replace("+ ", "+").replace("- ", "-")  
    
    lhs, rhs = equation.split('=') 
    
    terms = lhs.split() # split the left-hand side into terms

    coefficients = [0] * len(terms) 

    for term in terms:
        match = re.match(r'([+-]?\d*)([a-zA-Z]+)', term)
        if match:
            coeff = match.group(1)
            var = match.group(2)
            
            # cases like 'x' which is equivalent to '1x' and '-x' to '-1x'
            if coeff == '' or coeff == '+':
                coeff = 1
            elif coeff == '-':
                coeff = -1
            else:
                coeff = int(coeff)
            
            coefficients[terms.index(term)] = coeff

    constant = int(rhs.strip())
    
    return coefficients, constant

def read_system_from_file(file_path):
    A = []
    B = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    for line in lines:
        coefficients, constant = parse_equation(line.strip())
        A.append(coefficients)
        B.append(constant)
    
    return A, B

file_path = 'equations.txt'  # The file should contain the system of equations in the described format
A, B = read_system_from_file(file_path)

print("\n1. Parsing the System of Equations")
print("Matrix A:")
for row in A:
    print(row)

print("\nVector B:")
print(B)


#2. Matrix and Vector Operations

#2.1 Determinant
def get_submatrix(A, row, col):
    return [ [A[i][j] for j in range(len(A)) if j != col] for i in range(len(A)) if i != row ]

def determinant(A):
    n = len(A)
    
    # 1x1 matrix
    if n == 1:
        return A[0][0]
    
    # 2x2 matrix
    if n == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    
    det = 0
    for col in range(n):
        # submatrix by eliminating the first row and the current column
        submatrix = get_submatrix(A, 0, col)
        
        det += ((-1) ** col) * A[0][col] * determinant(submatrix)
    
    return det

print("\n\n2. Matrix and Vector Operations")
print("\n2.1 Determinant")
print("Determinant of matrix A:", determinant(A))

#2.2 Trace
def trace(A):
    return sum([A[i][i] for i in range(len(A))])

print("\n2.2 Trace")
print("Trace of matrix A:", trace(A))

#2.3 Vector Norm of Vector B
def vector_norm(B):
    return sum([abs(B[i]) for i in range(len(B))])

def euclidean_norm(B):
    sum_of_squares = sum(x**2 for x in B)
    
    norm = math.sqrt(sum_of_squares)
    
    return norm

print("\n2.3 Vector Norm of Vector B")
norm_B = euclidean_norm(B)
print("Euclidean norm of B:", norm_B)

#2.4 Transpose of Matrix A
def transpose(A):
    return [ [A[j][i] for j in range(len(A))] for i in range(len(A[0])) ]

print("\n2.4 Transpose of Matrix A")
A_transpose = transpose(A)
for row in A_transpose:
    print(row)

#2.5 Matrix-vector multiplication: Write a function that multiplies matrix A with vector B.
def matrix_vector_multiplication(A, B):
    result = [0] * len(A)
    
    for i in range(len(A)):
        for j in range(len(B)):
            result[i] += A[i][j] * B[j]
    
    return result

print("\n2.5 Matrix-vector multiplication")
result = matrix_vector_multiplication(A, B)
print(result)

#3 Solving using Cramer’s Rule
def cramer_rule(A, B):
    n = len(A)
    
    det_A = determinant(A)
    
    X = []
    for i in range(n):
        Ai = [row[:] for row in A]
        for j in range(n):
            Ai[j][i] = B[j]
        
        det_Ai = determinant(Ai)
        X.append(det_Ai / det_A)
    
    return X

print("\n\n3. Solving using Cramer’s Rule")
X = cramer_rule(A, B)
print("Solution:")
print(X)

#4. Solving using Inversion

#4.1 Cofactor Matrix
def cofactor_matrix(A):
    n = len(A)
    
    cofactor_matrix = []
    for i in range(n):
        cofactor_matrix.append([])
        for j in range(n):
            submatrix = get_submatrix(A, i, j)
            cofactor_matrix[i].append(((-1) ** (i + j)) * determinant(submatrix))
    
    return cofactor_matrix

A_cofactor = cofactor_matrix(A)
A_adj = transpose(A_cofactor)
A_inv = [[A_adj[i][j] / determinant(A) for j in range(len(A))] for i in range(len(A))]

print("\n\n4. Solving using Inversion")
print("\n4.1 Cofactor Matrix")
for row in A_cofactor:
    print(row)

#4.2 Inverse of Matrix A
print("\n4.2 Inverse of Matrix A")
for row in A_inv:
    print(row)

#4.3 Solution
X = matrix_vector_multiplication(A_inv, B)
print("\n4.3 Solution:")
print(X)

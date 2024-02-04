import numpy as np

class Classical_own:
    def __init__(self, size:int, matrix = None, vector = None):
        self.size = size

    def lu_decomposition(self,A):
        n = len(A)
        L = np.zeros((n, n))
        U = np.zeros((n, n))
        steps = 0  # Inicializace počítadla kroků

        for i in range(n):
            L[i][i] = 1
            for j in range(i, n):
                sum_U = sum(L[i][k] * U[k][j] for k in range(i))
                U[i][j] = A[i][j] - sum_U
                steps += 2 + i  # Počítáme operace pro sčítání a násobení
            for j in range(i + 1, n):
                sum_L = sum(L[j][k] * U[k][i] for k in range(i))
                L[j][i] = (A[j][i] - sum_L) / U[i][i]
                steps += 2 + i  # Počítáme operace pro sčítání a násobení

        return L, U, steps  # Vracíme i celkový počet kroků

    def forward_substitution(self,L, b):
        n = len(L)
        y = np.zeros(n)
        steps = 0  # Inicializace počítadla kroků

        for i in range(n):
            sum_ly = sum(L[i][j] * y[j] for j in range(i))
            y[i] = b[i] - sum_ly
            steps += i  # Počítáme operace pro sčítání a násobení

        return y, steps

    def backward_substitution(self,U, y):
        n = len(U)
        x = np.zeros(n)
        steps = 0  # Inicializace počítadla kroků

        for i in range(n - 1, -1, -1):
            sum_ux = sum(U[i][j] * x[j] for j in range(i + 1, n))
            x[i] = (y[i] - sum_ux) / U[i][i]
            steps += i + 1  # Počítáme operace pro sčítání a násobení

        return x, steps

    def solve_linear_system(self,A, b):
        L, U, steps_decomp = self.lu_decomposition(A)
        y, steps_forward = self.forward_substitution(L, b)
        x, steps_backward = self.backward_substitution(U, y)
        
        total_steps = steps_decomp + steps_forward + steps_backward  # Celkový počet kroků
        
        return x, total_steps
    def thomas_algorithm_count_operations(self, matrix, vector):  #a, b, c, d
        """
        Solve a tridiagonal linear system Ax = d using the Thomas Algorithm.
        
        Parameters:
        - a, b, c: coefficients of the tridiagonal matrix A
        - d: right-hand side vector
        
        Returns:
        - x: solution vector
        - operations: dictionary containing the count of operations (additions, multiplications, divisions)
        """
        c = self.extract_diagonal(matrix, shift=1)
        c.append(0)
        b = self.extract_diagonal(matrix, diagonal = [])
        a = self.extract_diagonal(matrix, shift=-1, diagonal=[0])
        d = vector.copy()

        n = len(d)
        operations = {'additions': 0, 'multiplications': 0, 'divisions': 0}
        
        # Forward elimination
        for i in range(1, n):
            factor = a[i] / b[i-1]
            b[i] -= factor * c[i-1]
            d[i] -= factor * d[i-1]
            operations['additions'] += 2  # subtraction in b[i] and d[i]
            operations['multiplications'] += 2  # factor * c[i-1] and factor * d[i-1]
            operations['divisions'] += 1  # division in factor
        
        # Backward substitution
        x = [0] * n
        x[-1] = d[-1] / b[-1]
        operations['divisions'] += 1  # division in x[-1]
        
        for i in range(n-2, -1, -1):
            x[i] = (d[i] - c[i] * x[i+1]) / b[i]
            operations['additions'] += 2  # subtraction in x[i]
            operations['multiplications'] += 2  # c[i] * x[i+1] and d[i] - c[i] * x[i+1]
            operations['divisions'] += 1  # division in (d[i] - c[i] * x[i+1]) / b[i]
        
        return x, operations
    
    def extract_diagonal(self, matrix, shift=0, diagonal=[]):
        if shift == 1:
            for i in range(len(matrix)-1):
                diagonal.append(matrix[i][i+shift])
        elif shift == -1:
            for i in range(1, len(matrix)):
                diagonal.append(matrix[i][i+shift])
        else:
            for i in range(len(matrix)):
                diagonal.append(matrix[i][i])
        return diagonal
""" # Example usage:
a = [0, 7, 1, 10]  # lower diagonal
b = [2, 5, 2, 8]  # main diagonal
c = [3, 3, 12, 0]  # upper diagonal
d = [6, 12, 18, 20]  # right-hand side
matrix = np.array([[2, 3, 0, 0],
                   [7, 5, 3, 0],
                   [0, 1, 2, 12],
                   [0, 0, 10, 8]])
vector = [6, 12, 18, 20]

solution, operation_counts = Classical_own(3).thomas_algorithm_count_operations(matrix, vector)
print("Solution:", solution)
print("Operation Counts:", operation_counts)
solution, operation_counts = Classical_own(3).thomas_algorithm_count_operations_old(a, b, c, d)
print("Solution:", solution)
print("Operation Counts:", operation_counts) """

""" classical_solver = Classical_own(8)
print(classical_solver.matrix)
print(classical_solver.vector)
solution, total_steps = classical_solver.solve_linear_system(classical_solver.matrix, classical_solver.vector)
print(solution)
print(total_steps) """

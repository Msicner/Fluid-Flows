import numpy as np
from linear_solvers import NumPyLinearSolver, HHL
from linear_solvers.matrices.tridiagonal_toeplitz import TridiagonalToeplitz
from qiskit.quantum_info import Statevector
from qiskit import transpile, QuantumCircuit
from qiskit import Aer
import matplotlib.pyplot as plt
from math import e, sqrt
import time


class HHL_own:
    def __init__(self, n_qubits:int, matrix = [0], vector = [0], hermitian:bool=True):
        """Initializes the hhl solver with backend and number of qubits"""
        self.backend = Aer.get_backend('aer_simulator')
        self.hhl = HHL(1e-7, quantum_instance=self.backend)

        self.n_qubits = n_qubits

        if len(matrix) == 1:
            self.matrix, self.vector = self.get_matrix()
        else:
            self.matrix = matrix
            self.vector = vector
        
        if hermitian == False:
            #self.matrix = np.multiply(np.transpose(self.matrix), self.matrix)
            #self.vector = np.multiply(np.transpose(self.vector), self.vector)
            self.matrix = self.create_hermitian_matrix()
            self.vector = np.hstack((vector, [0]*(len(self.matrix)//2)))
            self.n_qubits += 1  # Since the matrix has increased in size
        
        assert len(self.matrix) == len(self.vector), (len(self.matrix), len(self.vector))

        self.depths = []

    def get_matrix(self, dimension:int = None):
        """Returns the hermitian matrix and right-hand-side vector with the corresponding dimensions
        
        Parameters
        -------
        dimension: (max 16)
            dimension of the matrix, when None → the dimension corresponds to the number of qubits"""
        matrix = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 3, 7, 1, 3, 5, 7],
                        [0, 0, 0, 0, 0, 0, 0, 0, 6, 3, 8, 9, 2, 4, 6, 9],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 3, 3, 7, 2, 5],
                        [0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 15, 9, 7, 8, 9, 4],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 7, 4, 3, 0, 8],
                        [0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 7, 8, 12, 2, 8, 9],
                        [0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 2, 9, 8, 9, 7, 5],
                        [0, 0, 0, 0, 0, 0, 0, 0, 7, 9, 5, 4, 7, 11, 13, 4],
                        [2, 6, 0, 7, 1, 3, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0],
                        [5, 3, 2, 8, 2, 4, 6, 9, 0, 0, 0, 0, 0, 0, 0, 0],
                        [3, 8, 1, 15, 3, 7, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                        [7, 9, 3, 9, 7, 8, 9, 4, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 2, 3, 7, 4, 12, 8, 7, 0, 0, 0, 0, 0, 0, 0, 0],
                        [3, 4, 7, 8, 3, 2, 9, 11, 0, 0, 0, 0, 0, 0, 0, 0],
                        [5, 6, 2, 9, 0, 8, 7, 13, 0, 0, 0, 0, 0, 0, 0, 0],
                        [7, 9, 5, 4, 8, 9, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0]])
        vector = np.array([1]*16)
        if dimension == None:
            dimension = int(pow(2, self.n_qubits))
        
        matrix = matrix[(16 - dimension)//2:(16 - dimension)//2 + dimension, (16 - dimension)//2:(16 - dimension)//2 + dimension]
        vector = vector[(16 - dimension)//2:(16 - dimension)//2 + dimension]
        
        return matrix, vector
    
    def create_hermitian_matrix(self):
        """Creates the hermitian matrix by creating a diagonal from matrices"""
        dimension = pow(2, self.n_qubits)
        zero = np.zeros((dimension, dimension))
        self.matrix = np.block([[zero, self.matrix], [np.transpose(self.matrix), zero]])
        return self.matrix
    
    def solve_hhl(self):
        """Solves the system using HHL and returns the solution vector"""
        self.naive_hhl_solution = self.hhl.solve(self.matrix, self.vector)
        self.naive_qc = transpile(self.naive_hhl_solution.state,
                                basis_gates=['id', 'rz', 'sx', 'x', 'cx'])
        self.naive_depth = self.naive_qc.depth()
        self.depths.append({len(self.matrix):self.naive_depth})
        if self.n_qubits == 1:
            self.position = [2, 4]
        elif self.n_qubits == 2:
            self.position = [64, 68]
        elif self.n_qubits == 3:
            self.position = [256, 264]
        elif self.n_qubits == 4:
            self.position = [1024, 1040]
        self.solution_vector = self.get_solution_vector(self.naive_hhl_solution)

        self.naive_sv = Statevector(self.naive_hhl_solution.state).data.real
        self.naive_full_vector = np.array(self.naive_sv[self.position[0]:self.position[1]])  # 2**(1+self.n_qubits*3) + 2**(self.n_qubits-1): 2**(1+self.n_qubits*3) + 2**self.n_qubits
        return self.solution_vector
        
    def get_solution_vector(self, solution):
        """Extracts and normalizes simulated state vector from LinearSolverResult."""
        # NEXT ROW IS PROBLEMATIC
        self.solution_vector = Statevector(solution.state).data[self.position[0]:self.position[1]].real
        self.norm = solution.euclidean_norm
        return self.norm * self.solution_vector / np.linalg.norm(self.solution_vector)

    def solve_classically(self):
        """Returns the classical solution of the system"""
        self.classical_solution = NumPyLinearSolver().solve(self.matrix,self.vector/np.linalg.norm(self.vector))
        return self.classical_solution

    def print_results(self):
        print("-------------RESULTS-------------")

        print('classical state:', self.classical_solution.state)
        print('naive state:')
        self.naive_hhl_solution.state.draw(output="mpl")
        plt.show()

        print('classical Euclidean norm:', self.classical_solution.euclidean_norm)
        print('naive Euclidean norm:', self.naive_hhl_solution.euclidean_norm)

        #print('naive depth:', naive_depth)
        print('depths:', self.depths)

        #print('naive raw solution vector:', naive_full_vector)

        print('full naive solution vector:', self.solution_vector)
        print('classical state:', self.classical_solution.state)
        self.graph()

    def graph(self):
        plt.scatter([x for x in range(len(self.naive_sv))], self.naive_sv)
        plt.show()

""" hermitian_sparse_matrix = np.array([
    [0.45, 0, 0, 0, 0, 0.23, 0, 0],
    [0, 12, 0, 0, 0.37, 0, 0, 0],
    [0, 0, 0, 0.35, 0, 0, 0, 5],
    [0, 0, 0.35, 0, 0, 0.4, 0, 0],
    [0, 0.37, 0, 0, 0, 0, 10.12, 0],
    [0.23, 0, 0, 0.4, 0, 0, 0, 0],
    [0, 0, 0, 0, 10.12, 0, 3, 0],
    [0, 0, 5, 0, 0, 0, 0, 30]
], dtype=np.complex128)

hermitian_tridiagonal_matrix = np.array([
    [0.45, 0.78, 0, 0, 0, 0, 0, 0],
    [0.78, 0.55, 0.68, 0, 0, 0, 0, 0],
    [0, 0.68, 0.87, 0.8, 0, 0, 0, 0],
    [0, 0, 0.8, 0.5, 0.42, 0, 0, 0],
    [0, 0, 0, 0.42, 0.36, 0.56, 0, 0],
    [0, 0, 0, 0, 0.56, 0.62, 0.7, 0],
    [0, 0, 0, 0, 0, 0.7, 0.71, 0.91],
    [0, 0, 0, 0, 0, 0, 0.91, 0.76]
], dtype=np.complex128)

full_hermitian_matrix = np.array([
    [0.67 + 0.23j, 0.88 - 0.71j, 0.45 + 0.98j, 0.12 + 0.76j, 0.32 - 0.51j, 0.19 - 0.66j, 0.93 + 0.37j, 0.51 - 0.84j],
    [0.88 + 0.71j, 0.91 - 0.52j, 0.34 + 0.18j, 0.84 - 0.67j, 0.27 + 0.42j, 0.56 - 0.24j, 0.73 - 0.91j, 0.61 + 0.45j],
    [0.45 - 0.98j, 0.34 - 0.18j, 0.67 + 0.05j, 0.22 - 0.34j, 0.48 - 0.11j, 0.92 - 0.77j, 0.64 + 0.26j, 0.78 + 0.93j],
    [0.12 - 0.76j, 0.84 + 0.67j, 0.22 + 0.34j, 0.77 - 0.82j, 0.15 + 0.29j, 0.35 - 0.62j, 0.21 + 0.78j, 0.44 - 0.16j],
    [0.32 + 0.51j, 0.27 - 0.42j, 0.48 + 0.11j, 0.15 - 0.29j, 0.48 + 0.15j, 0.64 - 0.81j, 0.39 + 0.95j, 0.11 - 0.43j],
    [0.19 + 0.66j, 0.56 + 0.24j, 0.92 + 0.77j, 0.35 + 0.62j, 0.64 + 0.81j, 0.29 - 0.47j, 0.55 + 0.69j, 0.74 - 0.06j],
    [0.93 - 0.37j, 0.73 + 0.91j, 0.64 - 0.26j, 0.21 - 0.78j, 0.39 - 0.95j, 0.55 - 0.69j, 0.32 - 0.51j, 0.68 - 0.91j],
    [0.51 + 0.84j, 0.61 - 0.45j, 0.78 - 0.93j, 0.44 + 0.16j, 0.11 + 0.43j, 0.74 + 0.06j, 0.68 + 0.91j, 0.21 + 0.73j]
], dtype=np.complex128)

# Generate a random complex matrix
random_matrix = np.random.rand(8, 8) + 1j * np.random.rand(8, 8)

# Make it Hermitian (ensure it is equal to its conjugate transpose)
hermitian_matrix = (random_matrix + random_matrix.conj().T) / 2

vector = np.array([1]*8)

start_time = time.time()
hhl_solver = HHL_own(3, matrix = hermitian_sparse_matrix, vector=vector, hermitian=True)
print(f"Condition number: {np.linalg.cond(hhl_solver.matrix)}")
#print(hhl_solver.matrix)
#print(hhl_solver.vector)
solution_vector = hhl_solver.solve_hhl()
end_time = time.time()
classical_solution = hhl_solver.solve_classically()
hhl_solver.print_results()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds") """

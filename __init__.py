import csv
# Math
import numpy as np
import matplotlib.pyplot as plt
from math import e, log2, floor,sqrt
import itertools
from sklearn.preprocessing import normalize
# Quantum
from linear_solvers import NumPyLinearSolver, HHL
from linear_solvers.matrices.tridiagonal_toeplitz import TridiagonalToeplitz
from qiskit.quantum_info import Statevector
from qiskit import transpile, QuantumCircuit
from qiskit import Aer
from vqls_prototype import VQLS, VQLSLog
from qiskit.primitives import Estimator, Sampler
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA

from problem_solution import Fluid_problem
from classical_solver_class import Classical_own
from vqls_class import VQLS_own
from hhl_class import HHL_own
from scipy.interpolate import interp2d

""" Problem Statement
k = 0.6 W/(m . K)  (=tepelná vodivost)
c_p = 4180 J/(kg . K)  (=měrná teplená kapacita)
(\\rho) = 997 kg/m^3  (=hustota)
h = 0.1 m  (=polovina výšky kanálu)
(\\Delta y) = 0.01 m
L = 1 m (=délka kanálu)
(\\Delta x) = 0.1 m
q = 500 W/m (=tepelný tok (kladný pokud je kapalina ohřívána))
inlet_temp = 277 K (=vstupní teplota proudu, která udává T_2 až T_(n-1) v bodech s indexem i = 1)

(\\nu) = 1.519e-6 m^2/s (=kinematická viskozita pro T = 278 K)
(\\Delta p) = -10000 Pa (rozdíl tlaků v potrubí)

inlet_temp_up = 4 °C
inlet_temp_down = 4 °C


A = [[-3, 4, -1, 0, ..., 0],
    [-r/u(2), (2r/u(2) + 1), -r/u(2), 0, ..., 0],
    [0, -r/u(3), (2r/u(3) + 1), -r/u(3), 0, ..., 0],
    [0, ..., 0],
    [0, ..., 0, -r/u(n - 1), (2r/u(n - 1) + 1), -r/u(n - 1)],
    [0, ..., 0, 1, -4, 3]]
    
x = [T_1, T_2, T_3, ..., T_n]_(i+1)

b = [2q (\\Delta y) / k, T_2, T_3, ..., T_(n-1), -2q (\\Delta y) / k]_(i)"""

""" # Constants
k = 0.6
c_p = 4180
rho = 997
h = 1
delta_y = 0.25
l = 200
delta_x = 25
q = 50
nu = 1.519e-6
delta_p = -100

inlet_temp_up = 4
inlet_temp_down = 4

matrix_dimension = (floor(h/delta_y) * 2 + 1, floor(l/delta_x) + 1)  # (počet řádků, počet sloupců)
const_r = (k * delta_x ) / (c_p * rho * (delta_y * delta_y))
solution_vector_x = np.zeros(matrix_dimension)  # shoduje se s počtem sloupců (každý řádek je jedním sloupcem simulace)

# Zeroth layer solution depends only on the inlet temp, so it is filled with inlet temp
#solution_vector_x[0] = [inlet_temp for _ in range(matrix_dimension[0])]
solution_vector_x[0][:matrix_dimension[0]//2] = [inlet_temp_up for _ in range(matrix_dimension[0]//2)]
solution_vector_x[0][matrix_dimension[0]//2:] = [inlet_temp_down for _ in range(matrix_dimension[0]//2 + 1)]
print(solution_vector_x[0]) """

problem = Fluid_problem()
classical_solution = []
classical_thomas_solution = []
vqls_solution = []
hhl_solution = []

def show_graph(temperature_matrix, label):
    # Create a 2D array from the given data
    #temperature_matrix = np.array(data)

    # Interpolate for smoother plot
    x = np.arange(0, temperature_matrix.shape[1])
    y = np.arange(0, temperature_matrix.shape[0])
    f = interp2d(x, y, temperature_matrix, kind='cubic')

    # Generate more points
    x_new = np.linspace(0, temperature_matrix.shape[1] - 1, 200)
    y_new = np.linspace(0, temperature_matrix.shape[0] - 1, 200)
    heat_map_data_smooth = f(x_new, y_new)
    
    # Plotting the temperature matrix using imshow
    plt.figure(figsize=(8, 6))  # Adjust figure size if needed
    plt.imshow(heat_map_data_smooth, cmap='hot', interpolation='nearest', extent=[0, problem.l, -problem.h, problem.h])
    plt.colorbar(label='Temperature [°C]')  # Add color bar with label
    plt.title(f'Temperature Distribution - {label}')  # Set the plot title
    plt.xlabel('Canal length [m]')
    plt.ylabel('Canal height [m]')
    plt.gca().set_aspect(aspect=(problem.l/problem.h) / 2, adjustable='box')
    plt.grid(False)  # Optionally, turn off grid lines

for layer in range(1, problem.matrix_dimension[1]):
    A, b = problem.generate_problem_matrix(problem.solution_vector_x[layer - 1], layer)
    print("A: ", A)
    #print("Dimension A:", np.shape(A))
    print("b: ", b)

    # Solving
    print("Solving classical...")
    classical_solver = Classical_own(floor(problem.h/problem.delta_y) * 2 + 1)
    classical_x, total_steps = classical_solver.solve_linear_system(A, b)
    #classical_thomas_x, total_steps_thomas = classical_solver.thomas_algorithm_count_operations(A, b)
    
    if log2(np.shape(A)[0]) != int(log2(np.shape(A)[0])):
        A_cut = []    
        A = np.delete(A, len(A)//2, axis=0)
        A_cut.append(np.delete(A[0], len(A[0]) - 1))
        join = np.delete(A[1], len(A[1]) - 1)
        A_cut = np.vstack((A_cut[0], join))
        for i in range(2, len(A)//2):
            A_cut = np.vstack((A_cut, np.delete(A[i],len(A[i])-1)))

        for i in range(len(A)//2, len(A)):
            A_cut = np.vstack((A_cut, np.delete(A[i],0)))
        
        assert np.shape(A_cut)[0] == np.shape(A_cut)[1], "Matrix is not square"
        assert log2(np.shape(A_cut)[0]) == int(log2(np.shape(A_cut)[0])), "Matrix size is not the power of 2"
        
        b_cut = np.hstack((b[0:len(b)//2], b[len(b)//2+1:]))
        assert len(A_cut) == len(b_cut), "Matrix and vector sizes don't match"
    else:
        A_cut = A
        b_cut = b

    # Solve the system
    """print("Solving VQLS...")
    vqls_solver = VQLS_own(n_qubits=int(log2(np.shape(A_cut)[0])), matrix=A_cut, vector=b_cut, hermitian=False)
    vqls_solver.classical_solution()
    vqls_x = vqls_solver.solve_vqls(max_iteration=1000)"""
        
    print("Solving HHL...")
    hhl_solver= HHL_own(n_qubits=int(log2(np.shape(A_cut)[0])), matrix=A_cut, vector=b_cut, hermitian=False)
    hhl_x = hhl_solver.solve_hhl()
    hhl_solver.solve_classically()
    print(hhl_solver.matrix)
    print(hhl_solver.vector)

    # Print out the results
    print("RESULTS")
    print("Classical: ", classical_x)
    #print("Classical Thomas: ", classical_thomas_x)
    print("HHL: ", hhl_x)
    hhl_solver.print_results()
    coeff = classical_x[-1]/hhl_solver.solution_vector[-1]
    hhl_x = [hhl_solver.solution_vector[x] * coeff for x in range(len(hhl_solver.solution_vector))]
    print("Normalized HHL:",hhl_x)

    """print("VQLS: ", vqls_x)
    coeff = classical_x[-1]/vqls_solver.vqls_solution[-1]
    vqls_x = [vqls_solver.vqls_solution[x] * coeff for x in range(len(vqls_solver.vqls_solution))]
    print("Normalized VQLS:",[vqls_solver.vqls_solution[x] * coeff for x in range(len(vqls_solver.vqls_solution))])"""

    # Save the solution vectors
    classical_solution.append(classical_x)
    #classical_thomas_solution.append(classical_thomas_x)
    #vqls_solution.append(vqls_x[-4:])
    hhl_solution.append(hhl_x[-4:])
    problem.solution_vector_x[layer] = classical_x
    print(f"-------------Layer {layer} Finished-------------")

print("===========Final Results===========")
print("Classical solution:", classical_solution)
#print("Classical Thomas Solution:", classical_thomas_solution)
#print("VQLS_solution:", vqls_solution)
print("HHL_solution:", hhl_solution)

""" vqls = np.array([[0.56358021, 0.42705661, 0.42705661, 0.56358021],
                 [0.56023787, 0.43143195, 0.43143195, 0.56023787],
                 [0.55745009, 0.43502804, 0.43502804, 0.55745009],
                 [0.55504071, 0.43809794, 0.43809794, 0.55504071]])
for i in range(len(vqls)):
    coeff = classical_solution[i][0] / vqls[i][0]
    for j in range(len(vqls[i])):
        vqls[i][j] = vqls[i][j] * coeff 
vqls = np.transpose(vqls)
show_graph(vqls, "VQLS")"""

# Transpose the matrix so it corresponds to the canal
classical_solution = np.transpose(classical_solution)
#classical_thomas_solution = np.transpose(classical_thomas_solution)
hhl_solution = np.transpose(hhl_solution)

graph_solution = np.zeros((len(classical_solution)-1, len(classical_solution)-1))
position = 0
for i in range(len(classical_solution)):
    if i != len(classical_solution)//2:
        graph_solution[position] = classical_solution[i]
        position += 1
accuracy_list = []
for data in range(len(graph_solution)):
    for point in range(len(graph_solution[data])):
        accuracy_list.append(abs(1 - hhl_solution[data][point]/graph_solution[data][point]))
accuracy = sum(accuracy_list)/len(accuracy_list)

show_graph(graph_solution, label="Classical Solution")
#show_graph(classical_thomas_solution, label="Classical Thomas Solution")
show_graph(hhl_solution, label="HHL")
print(f"Accuracy {len(hhl_solution)}", accuracy, min(accuracy_list), max(accuracy_list))
plt.show()
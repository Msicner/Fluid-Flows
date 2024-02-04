import numpy as np
import matplotlib.pyplot as plt
from math import floor, log2

class Fluid_problem:
    def __init__(self, k = 0.6, c_p = 4180, rho = 997, h = 0.05, delta_y = 0.0125, l = 200, delta_x = 25, q = 50, nu = 1.519e-6, delta_p = -15, inlet_temp_bool=False, inlet_temp_up=4, inlet_temp_down=None):
        self.k = k
        self.c_p = c_p
        self.rho = rho
        self.h = h
        self.delta_y = delta_y
        self.l = l
        self.delta_x = delta_x
        self.q = q
        self.nu = nu
        self.delta_p = delta_p
        self.inlet_temp_up = inlet_temp_up
        self.inlet_temp_down = inlet_temp_down

        assert 2*h/delta_y == l/delta_x, "Numbers of discretization points in the two axis don't match"

        self.matrix_dimension = (floor(self.h/self.delta_y) * 2 + 1, floor(self.l/self.delta_x) + 1)  # (počet řádků, počet sloupců)
        self.const_r = (self.k * self.delta_x ) / (self.c_p * self.rho * (self.delta_y * self.delta_y))
        self.solution_vector_x = np.zeros(self.matrix_dimension)  # shoduje se s počtem sloupců (každý řádek je jedním sloupcem simulace)

        assert log2(self.matrix_dimension[0]-1) == int(log2(self.matrix_dimension[0] - 1)), "You can't pass this matrix to quantum algorithm, since its dimension is not close to the power of 2"

        if inlet_temp_bool == False:
            self.solution_vector_x[0] = [self.inlet_temp_up for _ in range(self.matrix_dimension[0])]
        else:
            # Zeroth layer solution depends only on the inlet temp, so it is filled with inlet temp
            #self.solution_vector_x[0][:self.matrix_dimension[0]//2] = [self.inlet_temp_up for _ in range(self.matrix_dimension[0]//2)]
            #self.solution_vector_x[0][self.matrix_dimension[0]//2:] = [self.inlet_temp_down for _ in range(self.matrix_dimension[0]//2 + 1)]
            temp_min = 4
            temp_max = 9
            for i in range(len(self.solution_vector_x[0])):
                self.solution_vector_x[0][i] = temp_min + (temp_max - temp_min)/len(self.solution_vector_x[0]-1)*i     


    def calculate_velocity(self, y_coordinate):
        """Returns velocity based on y_coordinate""" 
        u = - (self.delta_p * pow(self.h, 2)) / (2 * self.rho * self.nu * self.l) * (1 - pow(y_coordinate, 2)/pow(self.h, 2))
        return u

    def generate_problem_matrix(self, previous_solution_vector_x, slice_number):
        """Generates matricies A and b and returns A, b"""
        # Generating matrix A
        A = np.zeros(self.matrix_dimension)
        A[0][0:3] = [-3, 4, -1]  # Setting up the zeroth row
        A[-1][-3:] = [1, -4, 3]  # Setting up the j-th row
        for row in range(1, self.matrix_dimension[0] - 1):  # Setting up the rows from first to before j-th
            velocity = self.calculate_velocity(-self.h + row * self.delta_y)  # y = {-self.h + self.delta_y; self.h - self.delta_y}; row = {1; 9}
            r = self.const_r / velocity
            A[row][(row-1):(row+2)] = [-r, 2*r + 1, -r]
        
        #print(f'A (slice: {slice_number}): ', A)

        # Generating matrix b
        b = np.zeros(self.matrix_dimension[0])  # shoduje se s počtem řádků
        b[0] = -(2 * self.q * self.delta_y) / self.k
        b[-1] = (2 * self.q * self.delta_y) / self.k
        """ if slice_number == 0:
            b[1:-1] = [inlet_temp for _ in range(matrix_dimension[0] - 2)]
        else: """
        
        b[1:-1] = previous_solution_vector_x[1:-1]
        #print(f'b (slice: {slice_number}): ', b)

        return A, b
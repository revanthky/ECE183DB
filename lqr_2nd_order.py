import numpy as np
from scipy import linalg as la
from copy import deepcopy
import math
import random
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import time
import itertools
"""
BSD 2-Clause License
Copyright (c) 2017, Andrew Dahdouh
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CON   TRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# Author: Addison Sears-Collins
# https://automaticaddison.com
# Description: Linear Quadratic Regulator example 
#   (two-wheeled differential drive robot car)
 
######################## DEFINE CONSTANTS #####################################
# Supress scientific notation when printing NumPy arrays
np.set_printoptions(precision=3,suppress=True)
 
# Optional Variables
#max_linear_velocity = 1.565 # meters per second
max_angle = 0.610 # radians per second
mass = 682.04 #kg
max_thrust = 587.16 #N
avg_drag_force = 59.1613 #N
drag_acceleration = avg_drag_force / mass
max_acceleration = max_thrust / mass
dt = 1.0

class PathPlanner:

    def __init__(self, grid):
        """
        Constructor of the PathPlanner Class.
        :param grid: List of lists that represents the
        occupancy map/grid. List should only contain 0's
        for open nodes and 1's for obstacles/walls.
        :param visual: Boolean to determine if Matplotlib
        animation plays while path is found.
        """
        self.grid = grid
        self.heuristic = None
        self.goal_node = None

    def calc_heuristic(self):
        """
        Function will create a list of lists the same size
        of the occupancy map, then calculate the cost from the
        goal node to every other node on the map and update the
        class member variable self.heuristic.
        :return: None.
        """
        layer = len(self.grid)
        row = len(self.grid[0])
        col = len(self.grid[0][0])

        self.heuristic = [[[0 for x in range(col)] for y in range(row)] for z in range(layer)]
        
        #print(len(self.heuristic) == len(self.grid))
        #print(len(self.heuristic[0]) == len(self.grid[0]))
        #print(len(self.heuristic[0][0]) == len(self.grid[0][0]))

        for i in range(layer):
            for j in range(row):
                for k in range(col):
                    row_diff = abs(i - self.goal_node[0])
                    col_diff = abs(j - self.goal_node[1])
                    layer_diff = abs(k - self.goal_node[2])
                    self.heuristic[i][j][k] = math.sqrt(row_diff**2 + col_diff**2 + layer_diff**2)

        #print("Heuristic:")
        #print(self.heuristic)

    def a_star(self, start_cart, goal_cart):
        """
        A* Planner method. Finds a plan from a starting node
        to a goal node if one exits.
        :param init: Initial node in an Occupancy map. [x, y].
        Type: List of Ints.
        :param goal: Goal node in an Occupancy map. [x, y].
        Type: List of Ints.
        :return: Found path or -1 if it fails.
        """
        goal = [goal_cart[2], goal_cart[1], goal_cart[0]]
        self.goal_node = goal
        init = [start_cart[2], start_cart[1], start_cart[0]]
        # Calculate the Heuristic for the map
        self.calc_heuristic()
        print("Start point (z, y, x): ", init)
        print("End point (z, y, x): ", goal)

        # Different move/search direction options:

        #delta = [[-1, 0],  # go up
        #         [0, -1],  # go left
        #         [1, 0],  # go down
        #         [0, 1]]  # go right
        #delta_name = ['^ ', '< ', 'v ', '> ']

        # If you wish to use diagonals:
        # (z, y, x)
        delta = [[0, -1, 0],  # go up in plane
                 [0, 0, -1],  # go left in plane
                 [0, 1, 0],  # go down in plane
                 [0, 0, 1],  # go right in plane
                 [0, -1, -1],  # upper left in plane
                 [0, 1, -1],  # lower left in plane
                 [0, -1, 1],  # upper right in plane
                 [0, 1, 1],  # lower right in plane
                 [-1, -1, 0],  # go up and rise
                 [-1, 0, -1],  # go left and rise
                 [-1, 1, 0],  # go down and rise
                 [-1, 0, 1],  # go right and rise
                 [-1, -1, -1],  # upper left and rise
                 [-1, 1, -1],  # lower left and rise
                 [-1, -1, 1],  # upper right and rise
                 [-1, 1, 1],  # lower right and rise
                 [1, -1, 0],  # go up and sink
                 [1, 0, -1],  # go left and sink
                 [1, 1, 0],  # go down and sink
                 [1, 0, 1],  # go right and sink
                 [1, -1, -1],  # upper left and sink
                 [1, 1, -1],  # lower left and sink
                 [1, -1, 1],  # upper right and sink
                 [1, 1, 1]]  # lower right and sink
        delta_name = ['^0 ', '<0 ', 'v0 ', '>0 ', 'UL0', 'LL0', 'UR0', 'LR0', 
                    '^F ', '<F ', 'vF ', '>F ', 'ULF', 'LLF', 'URF', 'LRF', 
                    '^S ', '<S ', 'vS ', '>S ', 'ULS', 'LLS', 'URS', 'LRS'] 
                    #F is float (rise), S is sink, 0 is stationary on z axis

        # Heavily used from some of the A* Examples by Sebastian Thrun:

        closed = [[[0 for col in range(len(self.grid[0][0]))] for row in range(len(self.grid[0]))] for layer in range(len(self.grid))]
        shortest_path = [[['X' for _ in range(len(self.grid[0][0]))] for _ in range(len(self.grid[0]))] for _ in range(len(self.grid))]
        for x in range(len(self.grid[0][0])):
            for y in range(len(self.grid[0])):
                for z in range(len(self.grid)):
                    if self.grid[z][y][x] == 0:
                        shortest_path[z][y][x] = ' '
        closed[init[0]][init[1]][init[2]] = 1

        expand = [[[-1 for col in range(len(self.grid[0][0]))] for row in range(len(self.grid[0]))] for layer in range(len(self.grid))]
        delta_tracker = [[[-1 for _ in range(len(self.grid[0][0]))] for _ in range(len(self.grid[0]))] for _ in range(len(self.grid))]

        cost = 1
        z = init[0]
        y = init[1]
        x = init[2]
        g = 0
        f = g + self.heuristic[z][y][x]
        open = [[f, g, z, y, x]]

        found = False  # flag that is set when search is complete
        resign = False  # flag set if we can't find expand
        count = 0
        while not found and not resign:
            if len(open) == 0:
                resign = True
                return -1
            else:
                open.sort()
                open.reverse()
                next = open.pop()
                z = next[2]
                y = next[3]
                x = next[4]
                g = next[1]
                expand[z][y][x] = count
                count += 1

                if z == goal[0] and y == goal[1] and x == goal[2]:
                    found = True
                else:
                    for i in range(len(delta)):
                        z2 = z + delta[i][0]
                        y2 = y + delta[i][1]
                        x2 = x + delta[i][2]
                        if len(self.grid) > z2 >= 0 and len(self.grid[0]) > y2 >= 0 and len(self.grid[0][0]) > x2 >= 0:
                            #print(z2, y2, x2)
                            if closed[z2][y2][x2] == 0 and self.grid[z2][y2][x2] == 0:
                                g2 = g + cost
                                f = g2 + self.heuristic[z2][y2][x2]
                                open.append([f, g2, z2, y2, x2])
                                closed[z2][y2][x2] = 1
                                delta_tracker[z2][y2][x2] = i

        current_z = goal[0]
        current_y = goal[1]
        current_x = goal[2]
        shortest_path[current_z][current_y][current_x] = '* '
        full_path = []
        deltas = []
        while current_z != init[0] or current_y != init[1] or current_x != init[2]:
            print(delta_tracker[current_z][current_y][current_x])
            print(current_z, current_y, current_x)
            previous_z = current_z - delta[delta_tracker[current_z][current_y][current_x]][0]
            previous_y = current_y - delta[delta_tracker[current_z][current_y][current_x]][1]
            previous_x = current_x - delta[delta_tracker[current_z][current_y][current_x]][2]
            shortest_path[previous_z][previous_y][previous_x] = delta_name[delta_tracker[current_z][current_y][current_x]]
            full_path.append((current_z, current_y, current_x))
            deltas.append(delta_tracker[current_z][current_y][current_x])
            current_z = previous_z
            current_y = previous_y
            current_x = previous_x
        full_path.reverse()
        print("Found the goal in {} iterations.".format(count))
        print("full_path (z, y, x): ", full_path)
        #print("shortest path: ")
        #for a in range(len(shortest_path)):
            #print("\n")
            #for b in range(len(shortest_path[0])):
                #print(shortest_path[a][b])

        return init, full_path, deltas

def plan_path(sx, sy, sz, ex, ey, ez):
    sz = -sz
    ez = -ez
    width = abs(ex-sx)+1
    height = abs(ey-sy)+1
    depth = abs(ez-sz)+1
    test_grid = [[[0 for _ in range(width)] for _ in range(height)] for _ in range(depth)]
    #test_start = [random.randint(0,width-1), random.randint(0,height-1), random.randint(0,depth-1)]  # [x, y, z]
    #while test_grid[test_start[2]][test_start[1]][test_start[0]] == 1:
    #    test_start = [random.randint(0,width-1), random.randint(0,height-1), random.randint(0,depth-1)]  # [x, y, z]
    #test_goal = [random.randint(0,width-1), random.randint(0,height-1), random.randint(0,depth-1)]   # [x, y, z]
    #while test_grid[test_goal[2]][test_goal[1]][test_goal[0]] == 1:
    #    test_goal = [random.randint(0,width-1), random.randint(0,height-1), random.randint(0,depth-1)]   # [x, y, z]
    test_start = [sx,sy,sz]
    test_goal = [ex,ey,ez]
    xs = [test_start[0]]
    ys = [test_start[1]]
    zs = [-test_start[2]]
    obstacle_xs = []
    obstacle_ys = []
    obstacle_zs = []
    for y in range(1*height//10,4*height//10):
        for x in range(1*width//10,4*width//10):
            for z in range(0*depth//10,10*depth//10):
                test_grid[z][y][x] = 1
                if x >= 2*width//10 and x <= 3*width//10 and y>= 2*height//10 and y <= 3*height//10:
                    obstacle_xs.append(x)
                    obstacle_ys.append(y)
                    obstacle_zs.append(-z)
    for y in range(5*height//10,8*height//10):
        for x in range(6*width//10,9*width//10):
            for z in range(0*depth//10,10*depth//10):
                test_grid[z][y][x] = 1
                if x >= 7*width//10 and x <= 8*width//10 and y>= 6*height//10 and y <= 7*height//10:
                    obstacle_xs.append(x)
                    obstacle_ys.append(y)
                    obstacle_zs.append(-z)
    for y in range(5*height//10,8*height//10):
        for x in range(2*width//10,5*width//10):
            for z in range(0*depth//10,10*depth//10):
                test_grid[z][y][x] = 1
                if x >= 3*width//10 and x <= 4*width//10 and y>= 6*height//10 and y <= 7*height//10:
                    obstacle_xs.append(x)
                    obstacle_ys.append(y)
                    obstacle_zs.append(-z)
    #for y in range(2*height//10,5*height//10):
        #for x in range(5*width//10,8*width//10):
            #for z in range(0*depth//10,10*depth//10):
                #test_grid[z][y][x] = 1
                #if x >= 6*width//10 and x <= 7*width//10 and y>= 3*height//10 and y <= 4*height//10:
                    #obstacle_xs.append(x)
                    #obstacle_ys.append(y)
                    #obstacle_zs.append(-z)
    
    # Create an instance of the PathPlanner class:
    test_planner = PathPlanner(test_grid)

    # Plan a path.
    _, path, delts = test_planner.a_star(test_start, test_goal)
    for i in range(len(path)):
        xs.append(path[i][2])
        ys.append(path[i][1])
        zs.append(-path[i][0])
    #xs.append(test_goal[0])
    #ys.append(test_goal[1])
    #zs.append(-test_goal[2])

    waypoints = []
    delts.reverse()
    for i in range(1, len(path)):
        if delts[i] != delts[i-1]:
            waypoints.append((path[i-1][2], path[i-1][1], -path[i-1][0]))
    
    waypoints.append((test_goal[0], test_goal[1], -test_goal[2]))


    return waypoints, xs, ys, zs, obstacle_xs, obstacle_ys, obstacle_zs, test_start, test_goal

def getB(alpha, beta, gamma):

    B = np.zeros((9,4))
    yaw_rotation = np.array([[np.cos(alpha), -np.sin(alpha), 0.0],
                        [np.sin(alpha), np.cos(alpha), 0.0],
                        [0.0, 0.0, 1.0]])
    pitch_rotation = np.array([[np.cos(beta), 0.0, -np.sin(beta)],
                        [0.0, 1.0, 0.0],
                        [np.sin(beta), 0.0, np.cos(beta)]])
    roll_rotation = np.array([[1.0, 0.0, 0.0],
                        [0.0, np.cos(gamma), -np.sin(gamma)],
                        [0.0, np.sin(gamma), np.cos(gamma)]])
    rotation_matrix = yaw_rotation @ pitch_rotation @ roll_rotation
    #print(rotation_matrix[0][0])
    #print(np.cos(beta)*np.cos(alpha))
    #print(rotation_matrix[1][0])
    #print(np.cos(beta)*np.sin(alpha))
    #print(rotation_matrix[2][0])
    #print(np.sin(beta))
    #rotation_matrix[0][0] = np.cos(beta)*np.cos(alpha)
    #rotation_matrix[1][0] = np.cos(beta)*np.sin(alpha)
    #rotation_matrix[2][0] = np.sin(beta)
    B[0][0] = 0.5*rotation_matrix[0][0]*dt*dt
    B[1][0] = rotation_matrix[0][0]*dt
    B[2][0] = 0.5*rotation_matrix[1][0]*dt*dt
    B[3][0] = rotation_matrix[1][0]*dt
    B[4][0] = 0.5*rotation_matrix[2][0]*dt*dt
    B[5][0] = rotation_matrix[2][0]*dt
    B[6][1] = 0.2*dt #0.07
    B[7][2] = 0.1*dt #0.035
    B[7][3] = 0.1*dt #0.035
    B[8][2] = 0.2*dt
    B[8][3] = -0.2*dt
    return B
 
def clip_and_noise(control_input_t_minus_1):

    fin_noise = np.random.normal(0, 0.05*max_angle, 1)
    thrust_noise = np.random.normal(0, 0.05*max_acceleration, 1)
    control_input_t_minus_1[0] += thrust_noise
    control_input_t_minus_1[1] += fin_noise
    control_input_t_minus_1[2] += fin_noise
    control_input_t_minus_1[3] += fin_noise
    clipped = [0, 0, 0, 0]
    clipped[0] = np.clip(control_input_t_minus_1[0], -max_acceleration, max_acceleration)
    clipped[1] = np.clip(control_input_t_minus_1[1], -max_angle, max_angle)
    clipped[2] = np.clip(control_input_t_minus_1[2], -max_angle, max_angle)
    clipped[3] = np.clip(control_input_t_minus_1[3], -max_angle, max_angle)
    clipped[0] -= drag_acceleration
    return np.asarray(clipped)

def update_state_with_noise(A, state_t_minus_1, B, control_input_t_minus_1):
    
    position_noise = np.random.normal(0, 0.01, 1) #meters
    orientation_noise = np.random.normal(0, 0.005, 1) #radians
    state_estimate_t = (A @ state_t_minus_1) + (B @ control_input_t_minus_1) 
    state_estimate_t[0] += position_noise
    state_estimate_t[2] += position_noise
    state_estimate_t[4] += position_noise
    state_estimate_t[6] += orientation_noise
    state_estimate_t[7] += orientation_noise
    state_estimate_t[8] += orientation_noise
    return state_estimate_t

def lqr(actual_state_x, desired_state_xf, Q, R, A, B):
    """
    Discrete-time linear quadratic regulator for a nonlinear system.
 
    Compute the optimal control inputs given a nonlinear system, cost matrices, 
    current state, and a final state.
     
    Compute the control variables that minimize the cumulative cost.
 
    Solve for P using the dynamic programming method.
 
    :param actual_state_x: The current state of the system 
        3x1 NumPy Array given the state is [x,y,yaw angle] --->
        [meters, meters, radians]
    :param desired_state_xf: The desired state of the system
        3x1 NumPy Array given the state is [x,y,yaw angle] --->
        [meters, meters, radians]   
    :param Q: The state cost matrix
        3x3 NumPy Array
    :param R: The input cost matrix
        2x2 NumPy Array
    :param dt: The size of the timestep in seconds -> float
 
    :return: u_star: Optimal action u for the current state 
        2x1 NumPy Array given the control input vector is
        [linear velocity of the car, angular velocity of the car]
        [meters per second, radians per second]
    """
    # We want the system to stabilize at desired_state_xf.
    x_error = actual_state_x - desired_state_xf
 
    # Solutions to discrete LQR problems are obtained using the dynamic 
    # programming method.
    # The optimal solution is obtained recursively, starting at the last 
    # timestep and working backwards.
    # You can play with this number
    N = 50
 
    # Create a list of N + 1 elements
    P = [None] * (N + 1)
     
    Qf = Q
 
    # LQR via Dynamic Programming
    P[N] = Qf
 
    # For i = N, ..., 1
    for i in range(N, 0, -1):
 
        # Discrete-time Algebraic Riccati equation to calculate the optimal 
        # state cost matrix
        P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
            R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)
        #print("P[i]: ")
        #print(P[i-1])
 
    # Create a list of N elements
    K = [None] * N
    u = [None] * N
 
    # For i = 0, ..., N - 1
    for i in range(N):
 
        # Calculate the optimal feedback gain K
        K[i] = -np.linalg.pinv(R + B.T @ P[i+1] @ B) @ B.T @ P[i+1] @ A
        #print("K[i]: ")
        #print(K[i])
        u[i] = K[i] @ x_error
        #print("u[i]: ")
        #print(u[i])
    #print("x_error: ")
    #print(x_error)
 
    # Optimal control input is u_star
    u_star = u[N-1]
 
    #P = la.solve_discrete_are(A, B, Q, R)
    #K = la.solve(R + B.T.dot(P).dot(B), B.T.dot(P).dot(A))
    #u_star = K.dot(x_error)
    #K, S, E = control.lqr(A, B, Q, R)
    #u_star = K.dot(x_error)
    return u_star
 
def go_to_waypoint(start, end_x, end_y, end_z, Q_costs, eval=False):
     
     
    # Actual state
    actual_state_x = start 
 
    # Desired state [x, dx, y, dy, z, dz, azimuth angle, elevation angle, tilt angle]
    desx = end_x
    desy = end_y
    desz = end_z
    x_diff = desx - actual_state_x[0]
    y_diff = desy - actual_state_x[2]
    z_diff = desz - actual_state_x[4]
    desaz = np.arctan(y_diff/x_diff)
    desel = np.arctan(z_diff/(np.sqrt(x_diff**2+y_diff**2)))
    desired_state_xf = np.array([desx, 0.0, desy, 0.0, desz, 0.0, desaz, desel, 0.0])  
     
    # A matrix
    # 3x3 matrix -> number of states x number of states matrix
    # Expresses how the state of the system [x,y,yaw] changes 
    # from t-1 to t when no control command is executed.
    # Typically a robot on wheels only drives when the wheels are told to turn.
    # For this case, A is the identity matrix.
    # Note: A is sometimes F in the literature.
    A = np.array([[1.0, dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, dt, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, dt, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
 
    # R matrix
    # The control input cost matrix
    # Experiment with different R matrices
    # This matrix penalizes actuator effort (i.e. rotation of the 
    # motors on the wheels that drive the linear velocity and angular velocity).
    # The R matrix has the same number of rows as the number of control
    # inputs and same number of columns as the number of 
    # control inputs.
    # This matrix has positive values along the diagonal and 0s elsewhere.
    # We can target control inputs where we want low actuator effort 
    # by making the corresponding value of R large. 
    R = np.array([[750.00, 0.00, 0.00, 0.00],  # Penalty for thrust effort
                  [0.00, 0.01, 0.00, 0.00],  # Penalty for rudder effort
                  [0.00, 0.00, 0.01, 0.00],  # Penalty for left fin effort
                  [0.00, 0.00, 0.00, 0.01]]) # Penalty for right fin effort
 
    # Q matrix
    # The state cost matrix.
    # Experiment with different Q matrices.
    # Q helps us weigh the relative importance of each state in the 
    # state vector (X, Y, YAW ANGLE). 
    # Q is a square matrix that has the same number of rows as 
    # there are states.
    # Q penalizes bad performance.
    # Q has positive values along the diagonal and zeros elsewhere.
    # Q enables us to target states where we want low error by making the 
    # corresponding value of Q large.
    #Q = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Penalize X position error 
    #              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # Penalize dX error
    #              [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Penalize Y position error 
    #              [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Penalize dY error 
    #              [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0], # Penalize Z position error
    #              [0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0], # Penalize dZ error
    #              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0],  # Penalize AZIMUTH heading error
    #              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0], # Penalize ELEVATION heading error
    #              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]]) # Penalize TILT heading error
    Q = np.diag(Q_costs)

    # Launch the robot, and have it move to the desired goal destination
    #controls = []
    xpositions = []
    ypositions = []
    zpositions = []
    controls = []
    old_mag = 300.0
    for i in range(360):
        print(f'iteration = {i} seconds')
        print(f'Current State = {actual_state_x}')
        print(f'Desired State = {desired_state_xf}')
         
        state_error = actual_state_x - desired_state_xf
        concerns = np.array([state_error[0], state_error[2], state_error[4]])
        state_error_magnitude = np.linalg.norm(state_error)     
        print(f'State Error Magnitude = {state_error_magnitude}')
        concerns_magnitude = np.linalg.norm(concerns)
        print(f'Position Error Magnitude = {concerns_magnitude}')
        print(f'Q: {Q_costs}')
        B = getB(actual_state_x[6], actual_state_x[7], actual_state_x[8])
         
        # LQR returns the optimal control input
        optimal_control_input = lqr(actual_state_x, desired_state_xf, Q, R, A, B) 
        clipped_control_input = clip_and_noise(optimal_control_input)
        print(f'Control Input = {optimal_control_input}') 
        print(f'Clipped Control = {clipped_control_input}')      
        controls.append(clipped_control_input.tolist())
        
        # We apply the optimal control to the robot
        # so we can get a new actual (estimated) state.
        actual_state_x = update_state_with_noise(A, actual_state_x, B, clipped_control_input)  
        
        if eval:
            xpositions.append(actual_state_x[0])
            ypositions.append(actual_state_x[2])
            zpositions.append(actual_state_x[4])

        # Stop as soon as we reach the goal
        # Feel free to change this threshold value.
        if state_error_magnitude < 1.0 or concerns_magnitude < 3.8:
            print("\nGoal Has Been Reached Successfully!")
            break
        if state_error_magnitude > old_mag+0.2:
            break
        old_mag = state_error_magnitude
        print()
    #return actual_state_x, xpositions, ypositions, zpositions, linvels, ruddervels, lfvels, rfvels
    if not eval:
        return actual_state_x, concerns_magnitude
    return actual_state_x, xpositions, ypositions, zpositions, controls

def obstacle1():
    for y in range(1*height//10,4*height//10):
        for x in range(1*width//10,4*width//10):
            for z in range(0*depth//10,10*depth//10):
                if x >= 2*width//10 and x <= 3*width//10 and y>= 2*height//10 and y <= 3*height//10:
                    obstacle_xs.append(x)
                    obstacle_ys.append(y)
                    obstacle_zs.append(-z)
def obstacle2():
    for y in range(5*height//10,8*height//10):
        for x in range(6*width//10,9*width//10):
            for z in range(0*depth//10,10*depth//10):
                if x >= 7*width//10 and x <= 8*width//10 and y>= 6*height//10 and y <= 7*height//10:
                    obstacle_xs.append(x)
                    obstacle_ys.append(y)
                    obstacle_zs.append(-z)
def obstacle3():
    for y in range(5*height//10,8*height//10):
        for x in range(2*width//10,5*width//10):
            for z in range(0*depth//10,10*depth//10):
                if x >= 3*width//10 and x <= 4*width//10 and y>= 6*height//10 and y <= 7*height//10:
                    obstacle_xs.append(x)
                    obstacle_ys.append(y)
                    obstacle_zs.append(-z)
def obstacle4():
    for y in range(2*height//10,5*height//10):
        for x in range(5*width//10,8*width//10):
            for z in range(0*depth//10,10*depth//10):
                if x >= 6*width//10 and x <= 7*width//10 and y>= 3*height//10 and y <= 4*height//10:
                    obstacle_xs.append(x)
                    obstacle_ys.append(y)
                    obstacle_zs.append(-z)

# Entry point for the program

obstacles = [[2,3,4], [3,4], [4], [1,2,4], [1,4], [1,4], [1,3,4], [2], [2,3,4], [2,3], [2,3], [1,2,3], [1,3], [3], [3,4], [1,4], [1,2,4], [1,3,4], [1], [1,2]]
ways = [[(89, 89, -89), (89, 119, -119), (149, 149, -124)], 
[(124, 124, -124), (149, 149, -124)],
[(124, 124, -124), (149, 149, -124)],
[(59, 14, -59), (74, 29, -74), (89, 119, -124), (119, 149, -124), (149, 149, -124)],
[(14, 14, -14), (74, 29, -74), (74, 74, -119), (149, 149, -124)],
[(7, 7, -7), (7, 27, -27), (49, 69, -69), (79, 69, -99)],
[(7, 27, -27),  (39, 34, -59),  (79, 69, -99)],
[(47, 47, -47), (47, 55, -55), (79, 69, -99)],
[(47, 47, -47), (47, 55, -55), (79, 69, -99)],
[(47, 47, -47), (47, 55, -55), (79, 69, -99)],
[(24, 24, -24), (39, 28, -39), (39, 48, -59), (39, 49, -69)],
[(4, 4, -4), (15, 4, -15), (39, 28, -39),  (39, 49, -69)],
[(4, 4, -4), (15, 4, -15), (39, 28, -39), (39, 49, -69)],
[(39, 39, -39), (39, 49, -49), (39, 48, -68), (39, 49, -69)],
[(19, 19, -19), (19, 24, -24), (39, 44, -44), (39, 49, -69)],
[(10, 10, -10), (10, 35, -35), (64, 89, -89), (74, 89, -99), (109, 89, -99)],
[(10, 10, -10), (10, 35, -35), (64, 89, -89), (74, 89, -99), (109, 89, -99)],
[(10, 35, -35), (20, 45, -45),  (54, 44, -79), (109, 89, -99)],
[(10, 10, -10), (10, 35, -35), (64, 89, -89), (74, 89, -99), (109, 89, -99)],
[(10, 10, -10), (10, 35, -35), (64, 89, -89), (74, 89, -99), (109, 89, -99)]]

assert(len(obstacles)==len(ways))
Nu = len(ways)

for n in range(Nu):
    print(f'STARTING TRIAL {n} RIGHT NOW\n\n')
    time.sleep(1)
    o = obstacles[n]
    w = ways[n]
    start = [0, 0, 0]
    xs = [start[0]]+[v[0] for v in w]
    ys = [start[1]]+[v[1] for v in w]
    zs = [start[2]]+[v[2] for v in w]
    end = [w[-1][0], w[-1][1], w[-1][2]]
    startpoint = np.array([start[0], 0, start[1], 0, start[2], 0, 0, 0, 0])
    #waypoints, xs, ys, zs, obstacle_xs, obstacle_ys, obstacle_zs, test_start, test_goal = plan_path(start[0],start[1],start[2],end[0],end[1],end[2])
    width = abs(end[0]-start[0])+1
    height = abs(end[1]-start[1])+1
    depth = abs(end[2]-start[2])+1
    obstacle_xs = []
    obstacle_ys = []
    obstacle_zs = []
    if 1 in o:
        obstacle1()
    if 2 in o:
        obstacle2()
    if 3 in o:
        obstacle3()
    if 4 in o:
        obstacle4()
    test_start = start
    test_goal = end
    waypoints = w
    realxs = []
    realys = []
    realzs = []
    realcontrol = []
    trials = [0.1, 1.0, 10.0]
    costs = []
    best_costs = []
    min_error = 300.0
    for a in trials:
        for b in trials:
            for c in trials:
                for d in trials:
                    for e in trials:
                        for f in trials:
                            costs = [a,b,c,d,e,f,0.1,0.1,0.1]
                            start = [0, 0, 0]
                            startpoint = np.array([start[0], 0, start[1], 0, start[2], 0, 0, 0, 0])
                            error = 0.0
                            totalweight = 1.0
                            for way in waypoints:
                                startpoint, mid_error = go_to_waypoint(startpoint, way[0], way[1], way[2], costs)
                                if mid_error > 3.8:
                                    error += 0.15*mid_error
                                    totalweight -= 0.15
                            error += totalweight*mid_error
                            if error < min_error:
                                min_error = error
                                best_costs = costs
    start = [0, 0, 0]
    startpoint = np.array([start[0], 0, start[1], 0, start[2], 0, 0, 0, 0])
    for way in waypoints:
        startpoint, x_s, y_s, z_s, control = go_to_waypoint(startpoint, way[0], way[1], way[2], best_costs, eval=True)
        realxs.extend(x_s)
        realys.extend(y_s)
        realzs.extend(z_s)
        realcontrol.extend(control)
    
    f = open(f'trial{n+1}.txt', "w")
    f.write(f'Q:{best_costs}, Error:{min_error}')
    f.close()

    ax = plt.axes(projection='3d')
    ax.view_init(azim=45, elev=30)
    ax.plot3D(obstacle_xs, obstacle_ys, obstacle_zs, 'ro', alpha=0.3, label='obstacles')
    ax.plot3D(xs,ys,zs, color='black', label='planned path (waypoints)')
    ax.plot3D([test_start[0]], [test_start[1]], [test_start[2]], 'b*', label='plan start')
    ax.plot3D([test_goal[0]], [test_goal[1]], [test_goal[2]], 'g*', label='plan end')
    ax.plot3D(realxs,realys,realzs,'g',label='2nd order path')
    ax.plot3D([realxs[0]], [realys[0]], [realzs[0]], 'r*', label = '2nd order start')
    ax.plot3D([realxs[-1]], [realys[-1]], [realzs[-1]], 'm*', label = '2nd order end')
    ax.set_xlabel('x (meters)')
    ax.set_ylabel('y (meters)')
    ax.set_zlabel('z (meters)')
    plt.title(f'2nd order Waypoint Control')
    ax.legend(loc='center right')
    st = f'trial{n+1}_path_updated'
    plt.savefig(st)
    plt.clf()

    tim = [i for i in range(len(realcontrol))]
    accs = [realcontrol[i][0] for i in range(len(realcontrol))]
    st2 = f'trial{n+1}_acc_control_updated'
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration control (m/s^2)')
    plt.title('LQR Acceleration Control Input')
    plt.plot(tim, accs)
    plt.savefig(st2)
    plt.clf()

    #tim = [i for i in range(len(realcontrol))]
    ruds = [realcontrol[i][1] for i in range(len(realcontrol))]
    st3 = f'trial{n+1}_rud_control_updated'
    plt.xlabel('Time (seconds)')
    plt.ylabel('Rudder control (radians)')
    plt.title('LQR Rudder Control Input')
    plt.plot(tim, ruds)
    plt.savefig(st3)
    plt.clf()

    #tim = [i for i in range(len(realcontrol))]
    lfs = [realcontrol[i][2] for i in range(len(realcontrol))]
    st4 = f'trial{n+1}_lf_control_updated'
    plt.xlabel('Time (seconds)')
    plt.ylabel('Left Fin control (radians)')
    plt.title('LQR Left Fin Control Input')
    plt.plot(tim, lfs)
    plt.savefig(st4)
    plt.clf()

    tim = [i for i in range(len(realcontrol))]
    rfs = [realcontrol[i][3] for i in range(len(realcontrol))]
    st5 = f'trial{n+1}_rf_control_updated'
    plt.xlabel('Time (seconds)')
    plt.ylabel('Right Fin control (radians)')
    plt.title('LQR Right Fin Control Input')
    plt.plot(tim, rfs)
    plt.savefig(st5)
    plt.clf()
#velocities = np.asarray(vels)
#accelerations = np.diff(velocities)
#print(np.mean(np.absolute(velocities)))
#time = np.arange(len(velocities))
#plt.scatter(time, velocities)
#plt.title("Control Velocity Input")
#plt.xlabel('time (seconds)')
#plt.ylabel('velocity (meters/second)')
#plt.show()
#
#time = np.arange(len(accelerations))
#plt.scatter(time, accelerations)
#plt.title("Control Acceleration Input (from Velocity)")
#plt.xlabel('time (seconds)')
#plt.ylabel('acceleration (meters/second^2)')
#plt.show()
#
## Test 1
#rob = Robot(0, 0, 0, 0, 0, 0, 0, 0)
#xs = []
#ys = []
#zs = []
#dxs = []
#dys = []
#dzs = []
#azs = []
#els = []
#len_ = []
#for i in range(0,200,1):
    #len_.append(i)
#
    ##cur = rob.getCurState()
    ##goal = rob.getGoalState()
#
    ##print(f'iteration = {i} seconds')
    ##print(f'Current State = {cur}')
    ##print(f'Desired State = {goal}')
#         
    ##state_error = cur - goal
    ##state_error_magnitude = np.linalg.norm(state_error)     
    ##print(f'State Error Magnitude = {state_error_magnitude}')
#
    ##rob.update(rob.lqr())
    #if i < 100:
        #rob.update(np.array([0.05,0,-0.4,-0.4])) #second and third inputs should be same
    #if i >= 100:
        #rob.update(np.array([0.05,0.2,0,0])) #second and third inputs should be same
#
    #xs.append(rob.state[0])
    #dxs.append(rob.state[1])
    #ys.append(rob.state[2])
    #dys.append(rob.state[3])
    #zs.append(rob.state[4])
    #dzs.append(rob.state[5])
    #azs.append(rob.state[6])
    #els.append(rob.state[7])
#ax = plt.axes(projection='3d')
#ax.plot3D(xs,ys,zs, 'bo')
#ax.set_xlabel('x (meters)')
#ax.set_ylabel('y (meters)')
#ax.set_zlabel('z (meters)')
#plt.title("Simulation Output (Simple Fake Control)")
#plt.show()
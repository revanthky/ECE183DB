import numpy as np
import math
import random

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
#from scipy import linalg as la

# Author: Addison Sears-Collins
# https://automaticaddison.com
# Description: Linear Quadratic Regulator example 
#   (two-wheeled differential drive robot car)
 
######################## DEFINE CONSTANTS #####################################
# Supress scientific notation when printing NumPy arrays
np.set_printoptions(precision=3,suppress=True)
 
# Optional Variables
#max_linear_velocity = 3.0 # meters per second
#max_angular_velocity = 1.5708 # radians per second

mass = 1.0 #kilograms
length = 3.81 #meters
dt = 1.0

class Robot:
    def __init__(self, x, dx, y, dy, z, dz, az, el, desx=0, desdx=0, desy=0, desdy=0, desz=0, desdz=0, desaz=0, desel=0):
        self.state = np.array([x, dx, y, dy, z, dz, az, el])
        self.goal_state = np.array([desx, desdx, desy, desdy, desz, desdz, desaz, desel])
        self.A = np.identity(8)
        self.A[0][1] = dt
        self.A[2][3] = dt
        self.A[4][5] = dt
        self.R = np.array([[0.01, 0.00, 0.00, 0.00],  # Penalty for propeller force effort
                           [0.00, 0.01, 0.00, 0.00],  # Penalty for rudder effort
                           [0.00, 0.00, 0.01, 0.00],  # Penalty for left fin effort
                           [0.00, 0.00, 0.00, 0.01]]) # Penalty for right fin effort
        self.Q = np.diag(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1]))
        self.B = self.getB(self.state[6], self.state[7], dt)

    def getCurState(self):
        return self.state
    def getGoalState(self):
        return self.goal_state

    def getB(self, azimuth, elevation, deltat):
        """
        Expresses how the state of the system changes
        from t-1 to t due to the control commands (i.e. control inputs).
        """
        B = np.zeros((8,4))
        B[1][0] = np.cos(elevation)*np.cos(azimuth)*deltat/mass
        B[3][0] = np.cos(elevation)*np.sin(azimuth)*deltat/mass
        B[5][0] = np.sin(elevation)*deltat/mass
        #1 to 1 linear relationship between fin angle, drift angle, add listed multiple for more realism
        B[6][1] = -0.07*deltat #0.07
        B[7][2] = 0.035*deltat #0.035
        B[7][3] = 0.035*deltat #0.035
        return B

    def update(self, input_):
        self.state = (self.A @ self.state) + (self.B @ input_) 
        self.B = self.getB(self.state[6], self.state[7], dt)

    def lqr(self):
        """
        Discrete-time linear quadratic regulator for a nonlinear system.
    
        Compute the optimal control inputs given a nonlinear system, cost matrices, 
        current state, and a final state.

        Compute the control variables that minimize the cumulative cost.
    
        Solve for P using the dynamic programming method.
    
        """
        # We want the system to stabilize at desired_state_xf.
        x_error = self.state - self.goal_state
    
        # Solutions to discrete LQR problems are obtained using the dynamic 
        # programming method.
        # The optimal solution is obtained recursively, starting at the last 
        # timestep and working backwards.
        # You can play with this number
        N = 50
    
        # Create a list of N + 1 elements
        P = [None] * (N + 1)

        Qf = self.Q
    
        # LQR via Dynamic Programming
        P[N] = Qf
    
        # For i = N, ..., 1
        for i in range(N, 0, -1):
        
            # Discrete-time Algebraic Riccati equation to calculate the optimal 
            # state cost matrix
            P[i-1] = self.Q + self.A.T @ P[i] @ self.A - (self.A.T @ P[i] @ self.B) @ np.linalg.pinv(
                self.R + self.B.T @ P[i] @ self.B) @ (self.B.T @ P[i] @ self.A)
            #print("P[i]: ")
            #print(P[i-1])
    
        # Create a list of N elements
        K = [None] * N
        u = [None] * N
    
        # For i = 0, ..., N - 1
        for i in range(N):
            # Calculate the optimal feedback gain K
            K[i] = -np.linalg.pinv(self.R + self.B.T @ P[i+1] @ self.B) @ self.B.T @ P[i+1] @ self.A
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
        return u_star

#def main():
    ## Launch the robot, and have it move to the desired goal destination
    #for i in range(100):
        #print(f'iteration = {i} seconds')
        #print(f'Current State = {actual_state_x}')
        #print(f'Desired State = {desired_state_xf}')
#         
        #state_error = actual_state_x - desired_state_xf
        #state_error_magnitude = np.linalg.norm(state_error)     
        #print(f'State Error Magnitude = {state_error_magnitude}')
#         
        #B = getB(actual_state_x[6], actual_state_x[7], dt)
#         
        ## LQR returns the optimal control input
        #optimal_control_input = lqr(actual_state_x, desired_state_xf, Q, R, A, B, dt) 
        #print(f'Control Input = {optimal_control_input}')
#                                     
        ## We apply the optimal control to the robot
        ## so we can get a new actual (estimated) state.
        #actual_state_x = state_space_model(A, actual_state_x, B, optimal_control_input)  
# 
        ## Stop as soon as we reach the goal
        ## Feel free to change this threshold value.
        #if state_error_magnitude < 0.1:
            #print("\nGoal Has Been Reached Successfully!")
            #break
        #print()

# Test 1
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
#for i in range(0,50,1):
    #len_.append(i)
    #rob.state_dynamics(np.array([0.1,0,-np.pi/8,-np.pi/8])) #second and third inputs should be same
    #xs.append(rob.state[0])
    #dxs.append(rob.state[1])
    #ys.append(rob.state[2])
    #dys.append(rob.state[3])
    #zs.append(rob.state[4])
    #dzs.append(rob.state[5])
    #azs.append(rob.state[6])
    #els.append(rob.state[7])
#for i in range(50,100,1):
    #len_.append(i)
    #rob.state_dynamics(np.array([-0.05,-np.pi/8,0,0])) #second and third inputs should be same
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
#plt.title("Simulation Output (Fake Control)")
#plt.show()

# Test 1
rob = Robot(0, 0, 0, 0, 0, 0, 0, 0)
xs = []
ys = []
zs = []
dxs = []
dys = []
dzs = []
azs = []
els = []
len_ = []
for i in range(0,200,1):
    len_.append(i)
    cur = rob.getCurState()
    goal = rob.getGoalState()

    print(f'iteration = {i} seconds')
    print(f'Current State = {cur}')
    print(f'Desired State = {goal}')
         
    state_error = cur - goal
    state_error_magnitude = np.linalg.norm(state_error)     
    print(f'State Error Magnitude = {state_error_magnitude}')

    rob.update(rob.lqr())
    #rob.update(np.array([0.1,0,0,0])) #second and third inputs should be same
    xs.append(rob.state[0])
    dxs.append(rob.state[1])
    ys.append(rob.state[2])
    dys.append(rob.state[3])
    zs.append(rob.state[4])
    dzs.append(rob.state[5])
    azs.append(rob.state[6])
    els.append(rob.state[7])
ax = plt.axes(projection='3d')
ax.plot3D(xs,ys,zs, 'bo')
ax.set_xlabel('x (meters)')
ax.set_ylabel('y (meters)')
ax.set_zlabel('z (meters)')
plt.title("Simulation Output (Simple Fake Control)")
plt.show()

#main()
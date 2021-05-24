import numpy as np
from scipy import linalg as la
import time
import matplotlib.pyplot as plt

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
mass = 1.0

def getB(azimuth, elevation, tilt, deltat):
        """
        Expresses how the state of the system changes
        from t-1 to t due to the control commands (i.e. control inputs).
        """
        B = np.zeros((9,4))
        B[0][0] = 0.5*np.cos(elevation)*np.cos(azimuth)*(deltat**2)/mass
        B[1][0] = np.cos(elevation)*np.cos(azimuth)*deltat/mass
        B[2][0] = 0.5*np.cos(elevation)*np.sin(azimuth)*(deltat**2)/mass
        B[3][0] = np.cos(elevation)*np.sin(azimuth)*deltat/mass
        B[4][0] = 0.5*np.sin(elevation)*(deltat**2)/mass
        B[5][0] = np.sin(elevation)*deltat/mass
        B[6][1] = 0.2*deltat #0.07
        B[7][2] = 0.1*deltat #0.035
        B[7][3] = 0.1*deltat #0.035
        B[8][2] = 0.2*deltat
        B[8][3] = -0.2*deltat
        return B
 
 
def state_space_model(A, state_t_minus_1, B, control_input_t_minus_1):
    """
    Calculates the state at time t given the state at time t-1 and
    the control inputs applied at time t-1
     
    :param: A   The A state transition matrix
        3x3 NumPy Array
    :param: state_t_minus_1     The state at time t-1  
        3x1 NumPy Array given the state is [x,y,yaw angle] ---> 
        [meters, meters, radians]
    :param: B   The B state transition matrix
        3x2 NumPy Array
    :param: control_input_t_minus_1     Optimal control inputs at time t-1  
        2x1 NumPy Array given the control input vector is 
        [linear velocity of the car, angular velocity of the car]
        [meters per second, radians per second]
         
    :return: State estimate at time t
        3x1 NumPy Array given the state is [x,y,yaw angle] --->
        [meters, meters, radians]
    """
    # These next 6 lines of code which place limits on the angular and linear 
    # velocities of the robot car can be removed if you desire.
    #control_input_t_minus_1[0] = np.clip(control_input_t_minus_1[0],
    #                                                                        -max_linear_velocity,
    #                                                                        max_linear_velocity)
    #control_input_t_minus_1[1] = np.clip(control_input_t_minus_1[1],
    #                                                                        -max_angular_velocity,
    #                                                                        max_angular_velocity)
    state_estimate_t = (A @ state_t_minus_1) + (B @ control_input_t_minus_1) 
             
    return state_estimate_t
     
def lqr(actual_state_x, desired_state_xf, Q, R, A, B, dt):
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
 
def go_to_waypoint(start, end):
     
    # Let the time interval be 1.0 seconds
    dt = 1.0
     
    # Actual state
    #actual_state_x = np.array([0,0,0,0,0,0,0,0,0]) 
    actual_state_x = start
 
    # Desired state [x, y, z, azimuth angle, elevation angle, tilt angle]
    # [meters, meters, meters, radians, radians, radians]
    desx = 100.0
    desdx = 0.0
    desy = 50.0
    desdy = 0.0
    desz = -100.0
    desdz = 0.0
    desaz = np.arctan(desy/desx)
    desel = np.arctan(desz/(np.sqrt(desx**2+desy**2)))
    destl = 0.0
    #desired_state_xf = np.array([desx,desdx,desy,desdy,desz,desdz,desaz,desel,destl])  
    desired_state_xf = end
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
    Q = np.array([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Penalize X position error 
                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # Penalize dX error
                  [0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Penalize Y position error 
                  [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  # Penalize dY error 
                  [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0], # Penalize Z position error
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0], # Penalize dZ error
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0],  # Penalize AZIMUTH heading error
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0], # Penalize ELEVATION heading error
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]]) # Penalize TILT heading error
                   
    # Launch the robot, and have it move to the desired goal destination
    old_mag = 300.0
    a = []
    b = []
    c = []
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
        B = getB(actual_state_x[6], actual_state_x[7], actual_state_x[8], dt)
         
        # LQR returns the optimal control input
        optimal_control_input = lqr(actual_state_x, desired_state_xf, Q, R, A, B, dt) 
        print(f'Control Input = {optimal_control_input}')
                                     
         
        # We apply the optimal control to the robot
        # so we can get a new actual (estimated) state.
        actual_state_x = state_space_model(A, actual_state_x, B, optimal_control_input)  
        a.append(actual_state_x[0])
        b.append(actual_state_x[2])
        c.append(actual_state_x[4])
        # Stop as soon as we reach the goal
        # Feel free to change this threshold value.
        if state_error_magnitude < 1.0 or concerns_magnitude < 3.8:
            print("\nGoal Has Been Reached Successfully!")
            break
        if state_error_magnitude > old_mag+0.2:
            break
        old_mag = state_error_magnitude
        print()
    return actual_state_x, a, b, c

# Entry point for the program
ways = []
true_state = np.array([0,0,0,0,0,0,0,0,0])
desx = 10.0
desy = 20.0
desz = -50.0
desaz = np.arctan(desy/desx)
desel = np.arctan(desz/(np.sqrt(desx**2+desy**2)))
desired_state_xf = np.array([desx,0.0,desy,0.0,desz,0.0,desaz,desel,0.0])
ways.append(desired_state_xf)

desx = 50.0
desy = 50.0
desz = -100.0
desaz = np.arctan(desy/desx)
desel = np.arctan(desz/(np.sqrt(desx**2+desy**2)))
desired_state_xf = np.array([desx,0.0,desy,0.0,desz,0.0,desaz,desel,0.0])
ways.append(desired_state_xf)
xs = []
ys = []
zs = []
for way in ways:
    true_state, x, y, z = go_to_waypoint(true_state, way)
    xs.extend(x)
    ys.extend(y)
    zs.extend(z)
    time.sleep(2)

ax = plt.axes(projection='3d')
ax.plot3D(xs, ys, zs)
ax.plot3D([10], [20], [-50], 'r*')
ax.plot3D([50], [50], [-100], 'b*')
ax.set_xlabel('x (meters)')
ax.set_ylabel('y (meters)')
ax.set_zlabel('z (meters)')
plt.show()

import numpy as np
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
mass = 1.0

def getB(azimuth, elevation, deltat):
        """
        Expresses how the state of the system changes
        from t-1 to t due to the control commands (i.e. control inputs).
        """
        B = np.zeros((8,4))
        B[1][0] = np.cos(elevation)*np.cos(azimuth)*deltat/mass
        B[3][0] = np.cos(elevation)*np.sin(azimuth)*deltat/mass
        B[5][0] = np.sin(elevation)*deltat/mass
        B[6][1] = -0.07*deltat #0.07
        B[7][2] = 0.035*deltat #0.035
        B[7][3] = 0.035*deltat #0.035
        return B

def getA(z, dt):
        multiple = 0.0
        if z < 0 and z > -50:
            multiple = np.exp(z)/z

        A = np.array([[1.0, dt, 0.0, 0.0, multiple, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, dt, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, dt, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        return A
 
 
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
 
def main():
     
    # Let the time interval be 1.0 seconds
    dt = 1.0
     
    # Actual state
    actual_state_x = np.array([0,0.1,0,0.1,0,-0.1,0,0])
 
    # Desired state [x, y, z, azimuth angle, elevation angle, tilt angle]
    # [meters, meters, meters, radians, radians, radians]
    desx = 40.0
    desdx = 0.0
    desy = 40.0
    desdy = 0.0
    desz = -100.0
    desdz = 0.0
    desaz = np.arctan(desy/desx)
    desel = np.arctan(desz/(np.sqrt(desx**2+desy**2)))
    desired_state_xf = np.array([desx,desdx,desy,desdy,desz,desdz,desaz,desel])
     
    # A matrix
    # 3x3 matrix -> number of states x number of states matrix
    # Expresses how the state of the system [x,y,yaw] changes
    # from t-1 to t when no control command is executed.
    # Typically a robot on wheels only drives when the wheels are told to turn.
    # For this case, A is the identity matrix.
    # Note: A is sometimes F in the literature.
    #x, dx, y, dy, ... azimuth, elevation
    #direction of water flow, waves move from left to right
    #
#    A = np.array([[1.0, dt, 0.0, 0.0, (1/100), 0.0, 0.0, 0.0],
#                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                  [0.0, 0.0, 1.0, dt, 0.0, 0.0, 0.0, 0.0],
#                  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#                  [0.0, 0.0, 0.0, 0.0, 1.0, dt, 0.0, 0.0],
#                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
#                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
#                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])\

    
 
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
    R = np.array([[200.00, 0.00, 0.00, 0.00],  # Penalty for linear velocity effort
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
    Q = np.array([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Penalize X position error
                  [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # Penalize dX error
                  [0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Penalize Y position error
                  [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],  # Penalize dY error
                  [0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0], # Penalize Z position error
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0], # Penalize dZ error
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],  # Penalize AZIMUTH heading error
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]]) # Penalize ELEVATION heading error
                   
    # Launch the robot, and have it move to the desired goal destination
    for i in range(200):
        print(f'iteration = {i} seconds')
        print(f'Current State = {actual_state_x}')
        print(f'Desired State = {desired_state_xf}')
         
        state_error = actual_state_x - desired_state_xf
        state_error_magnitude = np.linalg.norm(state_error)
        print(f'State Error Magnitude = {state_error_magnitude}')
         
        A = getA(actual_state_x[4], dt)
        B = getB(actual_state_x[6], actual_state_x[7], dt)
         
        # LQR returns the optimal control input
        optimal_control_input = lqr(actual_state_x, desired_state_xf, Q, R, A, B, dt)
        print(f'Control Input = {optimal_control_input}')
                                     
         
        # We apply the optimal control to the robot
        # so we can get a new actual (estimated) state.
        actual_state_x = state_space_model(A, actual_state_x, B, optimal_control_input)
 
        # Stop as soon as we reach the goal
        # Feel free to change this threshold value.
        if state_error_magnitude < 0.01:
            print("\nGoal Has Been Reached Successfully!")
            break
        print()
 
# Entry point for the program
main()
